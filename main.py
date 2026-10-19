"""
Customer Churn Prediction and Sales Dashboard - Main Script

This script runs the complete pipeline:
1. Data Preprocessing
2. Churn Prediction Models (Logistic Regression, Random Forest, XGBoost, LightGBM)
3. Deep Learning Models (ANN)
4. Customer Segmentation (K-Means, DBSCAN)
5. Sales Trend Analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from churn_models import ChurnPredictor
from deep_learning_models import ChurnANN
from customer_segmentation import CustomerSegmentation
from sales_analysis import SalesAnalyzer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def run_preprocessing():
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor('Customer-Churn_Records.csv')
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline()
    preprocessor.save_preprocessor()
    
    print("\n[OK] Preprocessing completed successfully!")
    return preprocessor, X_train, X_test, y_train, y_test


def run_churn_models(X_train, X_test, y_train, y_test, feature_columns):
    print("\n" + "="*60)
    print("STEP 2: CHURN PREDICTION MODELS")
    print("="*60)
    
    predictor = ChurnPredictor()
    comparison = predictor.train_all_models(
        X_train, y_train, X_test, y_test, feature_columns
    )
    
    print("\n[OK] Churn prediction models trained and evaluated successfully!")
    return predictor, comparison


def run_deep_learning(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("STEP 3: DEEP LEARNING MODELS (ANN)")
    print("="*60)
    
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
    
    print("\n[OK] Deep learning models trained and evaluated successfully!")
    return ann, metrics


def run_segmentation(df):
    print("\n" + "="*60)
    print("STEP 4: CUSTOMER SEGMENTATION")
    print("="*60)
    
    segmentation = CustomerSegmentation()
    
    print("\n--- K-Means Clustering ---")
    kmeans_labels, kmeans_silhouette, kmeans_summary = segmentation.segment_customers(
        df, method='kmeans'
    )
    
    print("\n--- DBSCAN Clustering ---")
    dbscan_labels, dbscan_silhouette, dbscan_summary = segmentation.segment_customers(
        df, method='dbscan', eps=0.5, min_samples=5
    )
    
    print("\n[OK] Customer segmentation completed successfully!")
    return segmentation, kmeans_summary, dbscan_summary


def run_sales_analysis(df):
    print("\n" + "="*60)
    print("STEP 5: SALES TREND ANALYSIS")
    print("="*60)
    
    analyzer = SalesAnalyzer()
    analyzer.load_customer_data(df=df)
    
    transactions_df = analyzer.generate_transaction_data(
        num_customers=1000,
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    
    report = analyzer.generate_sales_report(transactions_df)
    
    print(f"\nTotal Transactions: {len(transactions_df):,}")
    print(f"Total Sales Volume: ${transactions_df['Amount'].sum():,.2f}")
    print(f"Average Transaction Value: ${transactions_df['Amount'].mean():,.2f}")
    
    print("\n[OK] Sales trend analysis completed successfully!")
    return analyzer, report


def main():
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION AND SALES DASHBOARD")
    print("Complete Pipeline Execution")
    print("="*60)
    
    try:
        preprocessor, X_train, X_test, y_train, y_test = run_preprocessing()
        
        predictor, comparison = run_churn_models(
            X_train, X_test, y_train, y_test, preprocessor.feature_columns
        )
        
        ann, ann_metrics = run_deep_learning(X_train, X_test, y_train, y_test)
        
        segmentation, kmeans_summary, dbscan_summary = run_segmentation(preprocessor.df)
        
        analyzer, sales_report = run_sales_analysis(preprocessor.df)
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nAll models have been trained and saved to the 'models/' directory.")
        print("Visualization plots have been saved to the 'models/' directory.")
        print("\nTo launch the interactive dashboard, run:")
        print("  streamlit run app.py")
        
    except Exception as e:
        print(f"\n[ERROR] Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
