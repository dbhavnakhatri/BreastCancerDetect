import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

def create_model():
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_synthetic_data(num_samples=200):
    np.random.seed(42)
    
    X_benign = np.random.rand(num_samples // 2, 224, 224, 3) * 0.6
    y_benign = np.zeros(num_samples // 2)
    
    X_malignant = np.random.rand(num_samples // 2, 224, 224, 3) * 0.4 + 0.5
    noise = np.random.rand(num_samples // 2, 224, 224, 3) * 0.3
    X_malignant = np.clip(X_malignant + noise, 0, 1)
    y_malignant = np.ones(num_samples // 2)
    
    X = np.concatenate([X_benign, X_malignant], axis=0)
    y = np.concatenate([y_benign, y_malignant], axis=0)
    
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]
    
    return X.astype('float32'), y.astype('float32')

if __name__ == "__main__":
    print("Creating model...")
    model = create_model()
    
    print("\nGenerating synthetic training data...")
    X_train, y_train = generate_synthetic_data(200)
    X_val, y_val = generate_synthetic_data(40)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    print("\nTraining model (quick training for demonstration)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16,
        verbose=1
    )
    
    print("\nSaving model...")
    os.makedirs("models", exist_ok=True)
    model.save("models/breast_cancer_model.keras")
    
    print("\nModel saved successfully to models/breast_cancer_model.keras")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
