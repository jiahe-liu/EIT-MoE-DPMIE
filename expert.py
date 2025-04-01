import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import save_history
from network import Network_fine_tune
from keras import optimizers
from keras.callbacks import EarlyStopping
import optuna

# Hardcoded arguments for Google Colab

args = {
    'expert_index': 4,  # Set the expert index (1, 2, 3, or 4)
    'obj': 4,  # Number of objects
    'epochs': 20000,  # Number of epochs
    'input_size': 256,  # Input size
    'image_size': 24,  # Image size
    'noise_level': 0,  # Noise level
    'lambda_cond': 0.8  # Weight factor, used for condition number constraints

}

EXPERT_RANGES = {

    1: (0, 39999),      # Expert 1: 1 object
    2: (40000, 79999),  # Expert 2: 2 objects
    3: (80000, 119999),  # Expert 3: 3 objects
    4: (120000, 159999)  # Expert 4: 4 objects
}


def shuffle(voltage, image, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(len(voltage))
    np.random.shuffle(indices)
    return voltage[indices], image[indices]


def split_data(voltageData, imageData, start_idx, end_idx):
    voltage_part = voltageData[start_idx:end_idx]

    image_part = imageData[start_idx:end_idx]

    voltage_80, voltage_20, image_80, image_20 = train_test_split(
        voltage_part, image_part, test_size=0.2, random_state=42)

    voltage_train, voltage_temp, image_train, image_temp = train_test_split(
        voltage_80, image_80, test_size=0.2, random_state=42)

    voltage_val, voltage_test, image_val, image_test = train_test_split(
        voltage_temp, image_temp, test_size=0.5, random_state=42)

    return voltage_train, voltage_val, voltage_test, image_train, image_val, image_test

# Load the sensitivity matrix CSV file (576x576) and convert it to a Tensor
kernel = pd.read_csv("./sensitivity_matrix.csv", header=None).values
kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)

# Define a custom loss function with a condition number constraint based on the sensitivity matrix
def cond_mse_loss(y_true, y_pred):
    epsilon = 1e-6
    # Compute the singular values of the sensitivity matrix
    s = tf.linalg.svd(kernel, compute_uv=False)
    sigma_max = s[0]
    sigma_min = s[-1]
    # Calculate the condition number of the sensitivity matrix
    cond_S = sigma_max / (sigma_min + epsilon)
    # Compute the conventional Mean Squared Error (MSE) loss
    mse_loss = K.mean(K.square(y_true - y_pred))
    # Use the condition number as a weighting factor; lambda_cond can be adjusted
    return args['lambda_cond'] * cond_S * mse_loss

def evaluate_model(model, voltage_val, image_val, save_path):
    predicted_images = model.predict(voltage_val)
    predicted_images = np.nan_to_num(predicted_images, nan=0.0, posinf=1.0, neginf=0.0)
    image_val = np.nan_to_num(image_val, nan=0.0, posinf=1.0, neginf=0.0)
    norm_predicted_images = (predicted_images - np.min(predicted_images)) / (
            np.max(predicted_images) - np.min(predicted_images))
    norm_image_val = (image_val - np.min(image_val)) / (np.max(image_val) - np.min(image_val))
    mse = np.mean((predicted_images - image_val) ** 2)
    rmse = np.sqrt(mse)
    ssim_values = []

    for i in range(len(image_val)):
        pred_img = norm_predicted_images[i].reshape((24, 24))
        true_img = norm_image_val[i].reshape((24, 24))
        if np.all(pred_img == pred_img[0, 0]) or np.all(true_img == true_img[0, 0]):
            ssim_value = 0
        else:
            ssim_value = ssim(true_img, pred_img, data_range=1.0)
        ssim_values.append(ssim_value)

    avg_ssim = np.mean(ssim_values)
    first_pred_img = predicted_images[0].reshape((24, 24))
    first_true_img = image_val[0].reshape((24, 24))
    df_first = pd.DataFrame({
        'Predicted': first_pred_img.flatten(),
        'True': first_true_img.flatten()
    })

    output_csv_path_first = os.path.join(save_path, "first_image_comparison.csv")
    df_first.to_csv(output_csv_path_first, index=False)
    second_pred_img = predicted_images[1].reshape((24, 24))
    second_true_img = image_val[1].reshape((24, 24))
    df_second = pd.DataFrame({
        'Predicted': second_pred_img.flatten(),
        'True': second_true_img.flatten()
    })

    output_csv_path_second = os.path.join(save_path, "second_image_comparison.csv")
    df_second.to_csv(output_csv_path_second, index=False)
    return mse, rmse, avg_ssim, predicted_images


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


def plot_loss(train_loss, val_loss, save_path):
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    plt.close()


def objective(trial):
    # Define hyperparameters to optimize
    num_layers = trial.suggest_int('num_layers', 1, 5)
    units_per_layer = [trial.suggest_int(f'units_layer_{i}', 64, 512) for i in range(num_layers)]

    # Let Optuna choose whether to use L1 or L2
    reg_type = trial.suggest_categorical('reg_type', ['L1', 'L2'])
    l1_lambda = trial.suggest_float('l1_lambda', 1e-6, 1e-2, log=True) if reg_type == 'L1' else 0.0
    l2_lambda = trial.suggest_float('l2_lambda', 1e-6, 1e-2, log=True) if reg_type == 'L2' else 0.0

    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    # Initialize model with the suggested hyperparameters

    model = Network_fine_tune(256, 24, units_per_layer, dropout_rate, l1_lambda, l2_lambda)
    optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9,
                                beta_2=0.999)  # Fixed: Use learning_rate instead of lr

    model.compile(loss=cond_mse_loss, optimizer=optimizer, metrics=['mse'])
    early_stopping = EarlyStopping(patience=30, restore_best_weights=True, min_delta=1e-3)
    hist = model.fit(x_train, y_train, epochs=args['epochs'], batch_size=batch_size,

                     validation_data=(x_val, y_val), callbacks=[early_stopping])
    val_loss = mean_squared_error(y_val, model.predict(x_val))

    # Save hyperparameters and loss to a file
    trials_path = "checkpoints/mlp/expert4/best_model/optuna_trials.txt"
    os.makedirs(os.path.dirname(trials_path), exist_ok=True)

    with open(trials_path, "a") as f:
        f.write(f"Trial {trial.number}:\n")
        f.write(f"  num_layers: {num_layers}\n")
        f.write(f"  units_per_layer: {units_per_layer}\n")
        f.write(f"  dropout_rate: {dropout_rate}\n")
        f.write(f"  learning_rate: {learning_rate}\n")
        f.write(f"  batch_size: {batch_size}\n")
        f.write(f"  reg_type: {reg_type}\n")
        f.write(f"  l1_lambda: {l1_lambda}\n")
        f.write(f"  l2_lambda: {l2_lambda}\n")
        f.write(f"  val_loss: {val_loss:.6f}\n")
        f.write("-" * 40 + "\n")

    return val_loss


# Main execution

if __name__ == '__main__':
    # Set start and end indices based on expert index

    args['start_index'], args['end_index'] = EXPERT_RANGES[args['expert_index']]

    # Load data
    script_dir = os.getcwd()  # Use current working directory in Colab

    image_path = os.path.join("./dataset", "", "24x24_NormImages_11Cond_2024-08-07.csv")
    voltage_path = os.path.join("./dataset", "", "24x24_CalibVoltage_11Cond_2024-08-07.csv")
    imageData = pd.read_csv(image_path).values

    voltageData = pd.read_csv(voltage_path).values

    x_train, x_val, x_test, y_train, y_val, y_test = zip(*[
        split_data(voltageData, imageData, args['start_index'], args['end_index'])
    ])

    x_train = np.concatenate(x_train)
    x_val = np.concatenate(x_val)
    x_test = np.concatenate(x_test)
    y_train = np.concatenate(y_train)
    y_val = np.concatenate(y_val)
    y_test = np.concatenate(y_test)

    # Optuna optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    # Best hyperparameters
    best_params = study.best_params
    best_val_loss = study.best_trial.value

    print("Best Hyperparameters:", best_params)

    # Train the final model with the best hyperparameters
    num_layers = best_params['num_layers']
    units_per_layer = [best_params[f'units_layer_{i}'] for i in range(num_layers)]
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    l1_lambda = best_params.get('l1_lambda', 0.0)
    l2_lambda = best_params.get('l2_lambda', 0.0)

    model = Network_fine_tune(256, 24, units_per_layer, dropout_rate, l1_lambda, l2_lambda)

    optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    early_stopping = EarlyStopping(patience=30, restore_best_weights=True, min_delta=1e-3)

    hist = model.fit(x_train, y_train, epochs=args['epochs'], batch_size=batch_size,

                     validation_data=(x_val, y_val), callbacks=[early_stopping])

    # Save model and training history
    path = os.path.join("checkpoints/mlp", f"expert{args['expert_index']}", f"best_model")

    os.makedirs(path, exist_ok=True)
    save_history(os.path.join(path, "train_loss.txt"), hist.history["loss"])
    save_history(os.path.join(path, "val_loss.txt"), hist.history["val_loss"])
    from joblib import dump
    dump(model, os.path.join(path, "model.joblib"))

    mse, rmse, avg_ssim, predicted_images = evaluate_model(model, x_test, y_test, save_path=path)

    print(f"Test MSE: {mse}")
    print(f"Test RMSE: {rmse}")
    print(f"Test SSIM: {avg_ssim}")

    with open(os.path.join(path, "test_result.txt"), "w") as f:
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
        f.write("\n")
        f.write(f"Test MSE: {mse}\n")
        f.write(f"Test RMSE: {rmse}\n")
        f.write(f"Test SSIM: {avg_ssim}\n")

plot_comparison(predicted_images[:25], y_test[:25], os.path.join(path, "comparison_images.png"))