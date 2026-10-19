import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ChurnANN:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.history = None
        os.makedirs(model_dir, exist_ok=True)
    
    def build_ann(self, input_shape):
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        print("ANN model built successfully.")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        print("Training ANN...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("ANN training completed.")
        return self.history
    
    def evaluate(self, X_test, y_test):
        print("Evaluating ANN...")
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\nANN Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def plot_training_history(self):
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'ann_training_history.png'))
        plt.close()
        print("Training history plot saved.")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - ANN')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(self.model_dir, 'ann_confusion_matrix.png'))
        plt.close()
        print("Confusion matrix saved for ANN.")
    
    def save_model(self, model_name='ann_model'):
        self.model.save(os.path.join(self.model_dir, f'{model_name}.keras'))
        print(f"ANN model saved: {model_name}.keras")
    
    def load_model(self, model_name='ann_model'):
        self.model = keras.models.load_model(os.path.join(self.model_dir, f'{model_name}.keras'))
        print(f"ANN model loaded: {model_name}.keras")
        return self.model


class ChurnRNN:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.history = None
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_sequences(self, X, sequence_length=10):
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        return np.array(sequences)
    
    def build_rnn(self, input_shape):
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        print("RNN model built successfully.")
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        print("Training RNN...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("RNN training completed.")
        return self.history
    
    def evaluate(self, X_test, y_test):
        print("Evaluating RNN...")
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\nRNN Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def save_model(self, model_name='rnn_model'):
        self.model.save(os.path.join(self.model_dir, f'{model_name}.keras'))
        print(f"RNN model saved: {model_name}.keras")


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    from sklearn.model_selection import train_test_split
    
    preprocessor = DataPreprocessor('Customer-Churn_Records.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    preprocessor.save_preprocessor()
    
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    ann = ChurnANN()
    ann.build_ann(X_train.shape[1])
    ann.train(X_train_final, y_train_final, X_val, y_val, epochs=50)
    metrics, y_pred, y_pred_proba = ann.evaluate(X_test, y_test)
    ann.plot_training_history()
    ann.plot_confusion_matrix(y_test, y_pred)
    ann.save_model()
    
    print("\nDeep learning models trained and evaluated successfully!")
