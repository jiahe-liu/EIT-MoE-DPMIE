import os
import argparse
import numpy as np
import pandas as pd
import joblib
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import torch
import matplotlib.pyplot as plt


# data split function
def split_data(voltageData, imageData, labels, start_idx, end_idx):
    # 1. Extract the data
    voltage_part = voltageData[start_idx:end_idx]
    image_part = imageData[start_idx:end_idx]
    labels_part = labels[start_idx:end_idx]

    # 2. Split into 80% majority and 20% minority
    voltage_80, voltage_20, image_80, image_20, labels_80, labels_20 = train_test_split(
        voltage_part, image_part, labels_part, test_size=0.2, random_state=42
    )

    # 3. Split the 80% part into 80% training set and 20% temporary data
    voltage_train_1, voltage_temp, image_train_1, image_temp, labels_train_1, labels_temp = train_test_split(
        voltage_80, image_80, labels_80, test_size=0.2, random_state=42
    )

    # 4. Split the temporary data into 50% training set and 50% validation set
    voltage_train_2, voltage_val, image_train_2, image_val, labels_train_2, labels_val = train_test_split(
        voltage_temp, image_temp, labels_temp, test_size=0.5, random_state=42
    )

    # 5. Split the 20% minority part into two test sets
    voltage_test_for_classifier, voltage_test_for_moe, image_test_for_classifier, image_test_for_moe, labels_test_for_classifier, labels_test_for_moe = train_test_split(
        voltage_20, image_20, labels_20, test_size=0.5, random_state=42
    )

    # 6. Merge the two training sets
    voltage_train = np.concatenate([voltage_train_1, voltage_train_2])
    image_train = np.concatenate([image_train_1, image_train_2])
    labels_train = np.concatenate([labels_train_1, labels_train_2])

    return voltage_train, voltage_val, voltage_test_for_classifier, voltage_test_for_moe, \
        image_train, image_val, image_test_for_classifier, image_test_for_moe, \
        labels_train, labels_val, labels_test_for_classifier, labels_test_for_moe


# normalize function
def normalize_images(images):
    min_val = np.min(images, axis=1, keepdims=True)
    max_val = np.max(images, axis=1, keepdims=True)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1  # Avoid dividing by zero
    norm_images = (images - min_val) / range_val
    return norm_images


def create_gating_network():
    model = Sequential()
    model.add(Dense(256, input_shape=(256,)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(4, activation='softmax'))  # output 4 expert
    return model


# Load the expert model and the weighted average function
def load_expert_models(expert_paths, weights):
    experts = [joblib.load(path) for path in expert_paths]

    def predict_with_weighted_average(data):
        predictions = [expert.predict(data) for expert in experts]
        combined_prediction = np.sum([w * pred for w, pred in zip(weights, predictions)], axis=0)
        return combined_prediction

    return predict_with_weighted_average


def select_expert_and_predict(voltage_data, gating_network, expert_predictors):
    gating_predictions = gating_network.predict(voltage_data)
    selected_experts = np.clip(np.argmax(gating_predictions, axis=1), 0, len(expert_predictors) - 1)

    predicted_images = np.zeros((voltage_data.shape[0], image_test_for_moe.shape[1]))

    for expert_idx in range(len(expert_predictors)):
        mask = (selected_experts == expert_idx)
        if np.any(mask):
            predicted_images[mask] = expert_predictors[expert_idx](voltage_data[mask])

    return predicted_images, gating_predictions, selected_experts


def save_history(filepath, history):
    with open(filepath, 'w') as f:
        for value in history:
            f.write(f"{value}\n")


def plot_comparison(predicted_images, ground_truth_images, filename, rows=5, cols=5):
    fig, axes = plt.subplots(rows, cols * 2, figsize=(20, 10))
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(predicted_images):
                axes[i, j * 2].imshow(predicted_images[idx].reshape((24, 24)), cmap='gray')
                axes[i, j * 2].axis('off')
                axes[i, j * 2 + 1].imshow(ground_truth_images[idx].reshape((24, 24)), cmap='gray')
                axes[i, j * 2 + 1].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    # Parameter configuration
    parser = argparse.ArgumentParser("MoE")
    parser.add_argument('--name', type=str, default='temp', help='experiment name')
    args = parser.parse_args()

    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct relative paths to dataset files
    image_path = os.path.join(script_dir, "dataset", "24x24_NormImages_11Cond_2024-08-07.csv")
    voltage_path = os.path.join(script_dir, "dataset", "24x24_CalibVoltage_11Cond_2024-08-07.csv")

    # Load data using the constructed paths
    imageData = pd.read_csv(image_path).values
    voltageData = pd.read_csv(voltage_path).values
    imageData = torch.tensor(imageData, dtype=torch.float)
    voltageData = torch.tensor(voltageData, dtype=torch.float)

    labels = np.zeros(len(voltageData), dtype=int)
    labels[40000:79999] = 1
    labels[80000:119999] = 2
    labels[120000:159999] = 3
    one_hot_labels = np.eye(4)[labels]

    # Data splitting intervals
    splits = [(0, 40000), (40000, 80000), (80000, 120000), (120000, 160000)]

    # Split and merge multiple data intervals based on splits
    x_train, x_valid, voltage_test_for_classifier_list, voltage_test_for_moe_list = [], [], [], []
    image_train, image_val, image_test_for_classifier_list, image_test_for_moe_list = [], [], [], []
    train_labels, valid_labels, labels_test_for_classifier_list, labels_test_for_moe_list = [], [], [], []

    for start, end in splits:
        xt, xv, x_test_cls, x_test_moe, yt, yv, y_test_cls, y_test_moe, lt, lv, l_test_cls, l_test_moe = \
            split_data(voltageData, imageData, one_hot_labels, start, end)

        x_train.append(xt)
        x_valid.append(xv)
        voltage_test_for_classifier_list.append(x_test_cls)
        voltage_test_for_moe_list.append(x_test_moe)

        image_train.append(yt)
        image_val.append(yv)
        image_test_for_classifier_list.append(y_test_cls)
        image_test_for_moe_list.append(y_test_moe)

        train_labels.append(lt)
        valid_labels.append(lv)
        labels_test_for_classifier_list.append(l_test_cls)
        labels_test_for_moe_list.append(l_test_moe)

    # Merge data
    x_train = np.concatenate(x_train)
    x_valid = np.concatenate(x_valid)
    voltage_test_for_classifier = np.concatenate(voltage_test_for_classifier_list)
    voltage_test_for_moe = np.concatenate(voltage_test_for_moe_list)

    image_train = np.concatenate(image_train)
    image_val = np.concatenate(image_val)
    image_test_for_classifier = np.concatenate(image_test_for_classifier_list)
    image_test_for_moe = np.concatenate(image_test_for_moe_list)

    train_labels = np.concatenate(train_labels)
    valid_labels = np.concatenate(valid_labels)
    labels_test_for_classifier = np.concatenate(labels_test_for_classifier_list)
    labels_test_for_moe = np.concatenate(labels_test_for_moe_list)


    # Function to shuffle data
    def shuffle_data(data, labels, targets):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        return data[indices], labels[indices], targets[indices]


    x_train, train_labels, y_train = shuffle_data(x_train, train_labels, image_train)
    x_valid, valid_labels, y_valid = shuffle_data(x_valid, valid_labels, image_val)

    voltage_test_for_classifier, labels_test_for_classifier, image_test_for_classifier = shuffle_data(
        voltage_test_for_classifier, labels_test_for_classifier, image_test_for_classifier
    )
    voltage_test_for_moe, labels_test_for_moe, image_test_for_moe = shuffle_data(
        voltage_test_for_moe, labels_test_for_moe, image_test_for_moe
    )

    # Create the gating network
    gating_network = create_gating_network()
    optimizer = Adam(learning_rate=0.001, decay=1e-6)  # 初始学习率 0.001，逐渐降低
    gating_network.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    early_stopping = EarlyStopping(patience=30, restore_best_weights=True)

    # Train the gating network
    hist = gating_network.fit(
        x_train, train_labels, epochs=20000, batch_size=256,
        validation_data=(x_valid, valid_labels), callbacks=[early_stopping]
    )

    # Compute training loss and validation loss
    train_loss, train_accuracy = gating_network.evaluate(x_train, train_labels, verbose=0)
    val_loss, val_accuracy = gating_network.evaluate(x_valid, valid_labels, verbose=0)

    loss, accuracy = gating_network.evaluate(voltage_test_for_classifier, labels_test_for_classifier, verbose=0)
    print(f"Classifier Test Loss: {loss}, Classifier Test Accuracy: {accuracy}")

    # Create a directory to save results
    path = os.path.join("checkpoints/combined_mlp_5models/", args.name)
    os.makedirs(path, exist_ok=True)
    # Save the gating network model
    gating_network.save(os.path.join(path, "gating_model.h5"))

    # Add the trained model path here
    expert_model_paths = {
        0: ['checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib'],
        1: ['checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib'],
        2: ['checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib'],
        3: ['checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib',
            'checkpoints/saveModel/....../model.joblib'],
    }

    expert_weights = {
        0: [0.37418062, 0.161722199, 0.157971697,0.153748245,0.15237724],
        1: [0.21879315, 0.195749522, 0.195006058, 0.194921607, 0.195529664],
        2: [0.206132084, 0.197366263, 0.199339518, 0.197521761, 0.199640374],
        3: [0.203666555, 0.200465388,  0.197881888,0.198659009, 0.19932716]
    }

    expert_predictors = {
        idx: load_expert_models(expert_model_paths[idx], expert_weights[idx]) for idx in expert_model_paths.keys()
    }

    # Gating network selects experts and makes predictions
    predicted_images, predictions, selected_experts = select_expert_and_predict(voltage_test_for_moe, gating_network,
                                                                                expert_predictors)

    # Compute the ratio of expert selection by the gating network
    expert_counts = np.bincount(selected_experts, minlength=4)
    total_samples = np.sum(expert_counts)
    expert_ratios = expert_counts / total_samples

    print("\n===== Gating Network Expert Selection Ratios =====")
    for i in range(4):
        print(f"Expert {i + 1}: {expert_ratios[i] * 100:.2f}%")

    # Save expert selection ratios to a file
    with open(os.path.join(path, "expert_selection_ratios.txt"), "w") as f:
        for i in range(4):
            f.write(f"Expert {i + 1}: {expert_ratios[i] * 100:.2f}%\n")

    # Compute MSE for misclassified samples by the gating network
    true_expert_labels = np.argmax(labels_test_for_moe, axis=1)  # labels are one-hot
    incorrect_classifications = selected_experts != true_expert_labels

    if np.any(incorrect_classifications):  # Ensure there are misclassified samples
        incorrect_mse = mean_squared_error(image_test_for_moe[incorrect_classifications].flatten(),
                                           predicted_images[incorrect_classifications].flatten())
        print(f"\nMSE of misclassified samples: {incorrect_mse:.6f}")

        with open(os.path.join(path, "test_result.txt"), "a") as f:
            f.write(f"\nMSE of misclassified samples: {incorrect_mse:.6f}\n")
    else:
        print("\nNo misclassified samples found. Unable to compute MSE.")

    # # Compute MoE's MSE
    mse = mean_squared_error(image_test_for_moe.flatten(), predicted_images.flatten())
    rmse = np.sqrt(mse)
    norm_predicted_images = normalize_images(predicted_images)
    norm_y_test = normalize_images(image_test_for_moe)

    ssim_values = []
    for i in range(len(norm_y_test)):
        true_img = norm_y_test[i].reshape((24, 24))
        pred_img = norm_predicted_images[i].reshape((24, 24))
        ssim_value = ssim(true_img, pred_img, data_range=1.0) if np.any(pred_img) and np.any(true_img) else 0
        ssim_values.append(ssim_value)

    avg_ssim = np.mean(ssim_values)

    print(f"Train Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")
    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")
    print(f"Test SSIM: {avg_ssim}")

    # Generate a visual comparison of predictions
    test_result_path = os.path.join(path, "test_result.txt")
    with open(test_result_path, "w") as f:
        f.write("===== Gating Network (Classifier) Results =====\n")
        f.write(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}\n")
        f.write(f"Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.6f}\n")
        f.write(f"Test Loss: {loss:.6f}, Test Accuracy: {accuracy:.6f}\n")

        f.write("\n===== MoE Network (Expert Model) Results =====\n")
        f.write(f"MoE Test MSE: {mse:.6f}\n")
        f.write(f"MoE Test RMSE: {rmse:.6f}\n")
        f.write(f"MoE Test SSIM: {avg_ssim:.6f}\n")

    print(f"Test results saved to {test_result_path}")

    # Quickly view the results in the terminal
    print("\n===== Gating Network (Classifier) Results =====")
    print(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}")
    print(f"Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.6f}")
    print(f"Test Loss: {loss:.6f}, Test Accuracy: {accuracy:.6f}")

    print("\n===== MoE Network (Expert Model) Results =====")
    print(f"MoE Test MSE: {mse:.6f}")
    print(f"MoE Test RMSE: {rmse:.6f}")
    print(f"MoE Test SSIM: {avg_ssim:.6f}")

    # Generate a visual comparison of predicted results
    plot_comparison(predicted_images[:25], image_test_for_moe[:25], os.path.join(path, "comparison_images.png"))
