import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


class ChurnPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.models = {}
        self.metrics = {}
        os.makedirs(model_dir, exist_ok=True)
    
    def train_logistic_regression(self, X_train, y_train):
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        print("Logistic Regression trained successfully.")
        return lr_model
    
    def train_random_forest(self, X_train, y_train):
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10,
            min_samples_split=5
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        print("Random Forest trained successfully.")
        return rf_model
    
    def train_xgboost(self, X_train, y_train):
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight= (len(y_train) - sum(y_train)) / sum(y_train)
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        print("XGBoost trained successfully.")
        return xgb_model
    
    def train_lightgbm(self, X_train, y_train):
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced'
        )
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        print("LightGBM trained successfully.")
        return lgb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        self.metrics[model_name] = metrics
        
        print(f"\n{model_name} Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(self.model_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()
        print(f"Confusion matrix saved for {model_name}")
    
    def plot_feature_importance(self, model, feature_names, model_name):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_imp.head(15))
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f'{model_name}_feature_importance.png'))
            plt.close()
            print(f"Feature importance plot saved for {model_name}")
    
    def save_model(self, model, model_name):
        joblib.dump(model, os.path.join(self.model_dir, f'{model_name}.pkl'))
        print(f"Model saved: {model_name}.pkl")
    
    def load_model(self, model_name):
        model = joblib.load(os.path.join(self.model_dir, f'{model_name}.pkl'))
        return model
    
    def compare_models(self):
        comparison_df = pd.DataFrame(self.metrics).T
        print("\nModel Comparison:")
        print(comparison_df)
        comparison_df.to_csv(os.path.join(self.model_dir, 'model_comparison.csv'))
        print("Model comparison saved to model_comparison.csv")
        return comparison_df
    
    def train_all_models(self, X_train, y_train, X_test, y_test, feature_names):
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        
        for model_name, model in self.models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
            y_pred = model.predict(X_test)
            self.plot_confusion_matrix(y_test, y_pred, model_name)
            self.plot_feature_importance(model, feature_names, model_name)
            self.save_model(model, model_name)
        
        comparison = self.compare_models()
        return comparison


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor('Customer-Churn_Records.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    preprocessor.save_preprocessor()
    
    predictor = ChurnPredictor()
    comparison = predictor.train_all_models(
        X_train, y_train, X_test, y_test, preprocessor.feature_columns
    )
    print("\nAll models trained and evaluated successfully!")
