import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
        # Handle relative paths
        if not os.path.isabs(data_path):
            self.data_path = os.path.join(os.path.dirname(__file__), '..', data_path)
            self.data_path = os.path.abspath(self.data_path)
        
    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def handle_missing_values(self):
        if self.df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        else:
            print("No missing values found.")
        return self.df
    
    def encode_categorical_variables(self):
        categorical_cols = ['Location', 'Gender', 'Card Type']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"Encoded {col}")
        
        return self.df
    
    def feature_engineering(self):
        self.df['CreditScore_per_Age'] = self.df['CreditScore'] / self.df['Age']
        self.df['Balance_per_Product'] = self.df['Account Balance'] / (self.df['NumOfProducts'] + 1)
        self.df['Tenure_per_Age'] = self.df['Tenure'] / self.df['Age']
        self.df['EstimatedSalary_per_Age'] = self.df['EstimatedSalary'] / self.df['Age']
        
        self.df['Engagement_Score'] = (
            self.df['IsActiveMember'] * 0.3 +
            self.df['HasCrCard'] * 0.2 +
            (self.df['Tenure'] / 10) * 0.3 +
            (self.df['NumOfProducts'] / 4) * 0.2
        )
        
        print("Feature engineering completed.")
        return self.df
    
    def prepare_features(self):
        feature_cols = [
            'CreditScore', 'Age', 'Tenure', 'Account Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score',
            'Point Earned', 'Location_encoded', 'Gender_encoded', 'Card Type_encoded',
            'CreditScore_per_Age', 'Balance_per_Product', 'Tenure_per_Age',
            'EstimatedSalary_per_Age', 'Engagement_Score'
        ]
        
        self.feature_columns = [col for col in feature_cols if col in self.df.columns]
        X = self.df[self.feature_columns]
        y = self.df['Exited']
        
        return X, y
    
    def normalize_features(self, X, fit=True):
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_preprocessor(self, save_dir='models'):
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'label_encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(save_dir, 'feature_columns.pkl'))
        print(f"Preprocessor saved to {save_dir}/")
    
    def load_preprocessor(self, load_dir='models'):
        self.scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(load_dir, 'label_encoders.pkl'))
        self.feature_columns = joblib.load(os.path.join(load_dir, 'feature_columns.pkl'))
        print(f"Preprocessor loaded from {load_dir}/")
    
    def preprocess_pipeline(self):
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical_variables()
        self.feature_engineering()
        X, y = self.prepare_features()
        X_scaled = self.normalize_features(X, fit=True)
        X_train, X_test, y_train, y_test = self.split_data(X_scaled, y)
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor = DataPreprocessor('Customer-Churn_Records.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    preprocessor.save_preprocessor()
    print("Preprocessing pipeline completed successfully!")
