import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from data_generator import DataGenerator
from unet import UNet


def IoUMetric(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((intersection + 1e-5) / (union - intersection + 1e-5))


def train(batch_size=10, data_dir='./data', epochs=2, test_size=0.1, usesDataGenerator=True):
    # List of callbacks
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    ]

    history = None

    if usesDataGenerator:
        # Initialize data generator
        train_generator = DataGenerator(batch_size, data_dir, shuffle=True, phase='train', test_size=test_size)
        validation_generator = DataGenerator(batch_size, data_dir, shuffle=True, phase='train', test_size=test_size)

        # Initialize model
        model = initialize_model()

        # Train model
        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callback_list
        )

    else:
        # Get training and validation data
        data = np.load("./preprocessed_data.npz")
        X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

        # Initialize model
        model = initialize_model()

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            callbacks=callback_list
        )

    # Save model
    model.save_weights('trained_unet_weights.h5')

    # Plot Training History
    plot_training_stats(history)
    return history


def initialize_model():
    model = UNet(n_classes=1)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', IoUMetric]
    )

    return model


def plot_training_stats(history):
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.subplot(121)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    acc, val_acc = history.history['IoUMetric'], history.history['val_IoUMetric']
    plt.subplot(122)
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    hist = train()
    hist_df = pd.DataFrame(hist.history)
    hist_json_file = 'history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
