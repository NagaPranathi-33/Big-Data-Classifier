import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression

def process(Data, Label, o1):
    # ---------------------- Check Input Dimensions ------------------------
    if Data.ndim != 2:
        raise ValueError(f"Data should be 2D, got shape: {Data.shape}")
    if Label.ndim == 1:
        Label = Label.reshape(-1, 1)
    elif Label.ndim != 2 or Label.shape[1] != 1:
        raise ValueError(f"Label should be a column vector, got shape: {Label.shape}")

    # ---------------------- Weight Calculation using RMSprop ------------------------
    opt = tf.keras.optimizers.RMSprop()  # ✅ FIXED: Removed legacy
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(Data.shape[1],)),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer=opt, loss='mse')
    model.fit(Data, Label, epochs=5, verbose=0)
    weights = model.get_weights()[0]  # Shape: (features, 2)

    # ---------------------- Linear Combination using Weights ------------------------
    y1 = np.sum(Data @ weights[:, 0])  # Weighted sum for 1st neuron output

    # ---------------------- Fusion with DRN Output (o1) -----------------------------
    alpha = 2
    o2 = alpha * y1 + 0.5 * alpha * np.sum(o1)

    # ---------------------- Feature Augmentation ------------------------------------
    o2_expanded = np.full((len(Data), 1), o2)
    fused_data = np.concatenate((Data, o2_expanded), axis=1)

    # ---------------------- Linear Regression Prediction ----------------------------
    reg = LinearRegression().fit(fused_data, Label)
    prediction = reg.predict(fused_data).astype(int).reshape(-1, 1)

    # ---------------------- Final Feature Set ---------------------------------------
    final_output = np.concatenate((fused_data, prediction), axis=1)

    return final_output
