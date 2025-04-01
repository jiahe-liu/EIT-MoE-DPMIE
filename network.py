from keras.regularizers import L1L2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape
from keras import Input


def Network_fine_tune(input_size, output_size, units_per_layer, dropout_rate, l1_lambda, l2_lambda):
    """
    Create an MLP network for fine-tuning EIT image reconstruction.

    Parameters:
    - input_size: The feature dimension of the input data (e.g., 256).
    - output_size: The size of the output image (e.g., 24, representing a 24x24 image).
    - units_per_layer: A list specifying the number of neurons in each hidden layer, e.g., [256, 128, 64].
    - dropout_rate: Dropout regularization rate (e.g., 0.2).
    - l1_lambda: L1 regularization coefficient (0 means not used).
    - l2_lambda: L2 regularization coefficient (0 means not used).

    Returns:
    - model: Keras Sequential model
    """

    model = Sequential()

    # Input
    model.add(Input(shape=(input_size,)))

    # Hidden layer
    for units in units_per_layer:
        model.add(Dense(units, activation='relu',
                        kernel_regularizer=L1L2(l1=l1_lambda, l2=l2_lambda)))  # 加入 L1/L2 正则化
        model.add(Dropout(dropout_rate))  # Dropout 以防止过拟合

    # Output
    model.add(Dense(output_size * output_size, activation='linear'))
    model.add(Reshape((output_size * output_size,)))

    return model