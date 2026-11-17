"""Fixed CNN architecture from Notebook 146."""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def prepare_cnn_features(X):
    if X.ndim!=2 or X.shape[1]!=50: raise ValueError(f"Expected (N, 50), got {X.shape}")
    return X.reshape(-1,10,5,1).astype(np.float32)

def build_final_cnn():
    inputs=keras.Input(shape=(10,5,1),name="50_Feature_Input")
    x=layers.Conv2D(32,(3,3),padding="same",activation="relu",name="Conv2D_32")(inputs)
    x=layers.MaxPooling2D((2,2),padding="same",name="MaxPool_1")(x)
    x=layers.Conv2D(64,(3,3),padding="same",activation="relu",name="Conv2D_64")(x)
    x=layers.MaxPooling2D((2,2),padding="same",name="MaxPool_2")(x)
    x=layers.Conv2D(128,(3,3),padding="same",activation="relu",name="Conv2D_128")(x)
    x=layers.GlobalAveragePooling2D(name="GlobalAveragePooling")(x)
    x=layers.Dense(128,activation="relu",name="Dense_128")(x)
    x=layers.Dropout(0.50,name="Dropout_50")(x)
    x=layers.Dense(64,activation="relu",name="Dense_64")(x)
    outputs=layers.Dense(1,activation="sigmoid",name="Nodule_Output")(x)
    model=keras.Model(inputs,outputs,name="CNN_SFO_SBS_50Features")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),keras.metrics.AUC(name="auc")])
    return model
