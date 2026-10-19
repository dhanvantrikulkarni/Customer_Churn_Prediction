import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os


class SalesAnalyzer:
    def __init__(self, output_dir='models'):
        self.output_dir = output_dir
        self.df = None
        os.makedirs(output_dir, exist_ok=True)
    
    def load_customer_data(self, data_path=None, df=None):
        if df is not None:
            self.df = df
            print(f"Customer data loaded from dataframe. Shape: {self.df.shape}")
        elif data_path is not None:
            self.df = pd.read_csv(data_path)
            print(f"Customer data loaded from file. Shape: {self.df.shape}")
        else:
            raise ValueError("Either data_path or df must be provided")
        return self.df
    
    def generate_transaction_data(self, num_customers=None, start_date='2023-01-01', end_date='2024-12-31'):
        if self.df is None:
            raise ValueError("Load customer data first.")
        
        if num_customers is None:
            num_customers = len(self.df)
        
        customers = self.df.sample(min(num_customers, len(self.df)), random_state=42)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_range = (end_dt - start_dt).days
        
        transactions = []
        
        for _, customer in customers.iterrows():
            num_transactions = np.random.randint(1, 20)
            customer_id = customer['CustomerId']
            
            for _ in range(num_transactions):
                random_days = np.random.randint(0, date_range)
                transaction_date = start_dt + timedelta(days=random_days)
                
                amount = np.random.uniform(10, 1000)
                if customer['Card Type'] == 'DIAMOND':
                    amount *= 1.5
                elif customer['Card Type'] == 'PLATINUM':
                    amount *= 1.3
                elif customer['Card Type'] == 'GOLD':
                    amount *= 1.1
                
                transactions.append({
                    'CustomerId': customer_id,
                    'TransactionDate': transaction_date,
                    'Amount': amount,
                    'Location': customer['Location'],
                    'CardType': customer['Card Type'],
                    'NumProducts': customer['NumOfProducts']
                })
        
        transactions_df = pd.DataFrame(transactions)
        transactions_df['Month'] = transactions_df['TransactionDate'].dt.to_period('M')
        transactions_df['Quarter'] = transactions_df['TransactionDate'].dt.to_period('Q')
        transactions_df['Year'] = transactions_df['TransactionDate'].dt.year
        
        print(f"Generated {len(transactions_df)} transactions.")
        return transactions_df
    
    def analyze_monthly_sales(self, transactions_df):
        monthly_sales = transactions_df.groupby('Month')['Amount'].agg(['sum', 'count', 'mean']).reset_index()
        monthly_sales.columns = ['Month', 'TotalSales', 'TransactionCount', 'AvgTransactionValue']
        monthly_sales['Month'] = monthly_sales['Month'].astype(str)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        axes[0].plot(monthly_sales['Month'], monthly_sales['TotalSales'], marker='o', linewidth=2)
        axes[0].set_title('Monthly Total Sales', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Total Sales ($)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(monthly_sales['Month'], monthly_sales['TransactionCount'], marker='s', color='orange', linewidth=2)
        axes[1].set_title('Monthly Transaction Count', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Number of Transactions')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(monthly_sales['Month'], monthly_sales['AvgTransactionValue'], marker='^', color='green', linewidth=2)
        axes[2].set_title('Average Transaction Value', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Month')
        axes[2].set_ylabel('Average Value ($)')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'monthly_sales_trends.png'), dpi=300)
        plt.close()
        print("Monthly sales trends plot saved.")
        
        return monthly_sales
    
    def analyze_quarterly_sales(self, transactions_df):
        quarterly_sales = transactions_df.groupby('Quarter')['Amount'].agg(['sum', 'count', 'mean']).reset_index()
        quarterly_sales.columns = ['Quarter', 'TotalSales', 'TransactionCount', 'AvgTransactionValue']
        quarterly_sales['Quarter'] = quarterly_sales['Quarter'].astype(str)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].bar(quarterly_sales['Quarter'], quarterly_sales['TotalSales'], color='steelblue')
        axes[0].set_title('Quarterly Total Sales', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Quarter')
        axes[0].set_ylabel('Total Sales ($)')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(quarterly_sales['Quarter'], quarterly_sales['TransactionCount'], color='coral')
        axes[1].set_title('Quarterly Transaction Count', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Quarter')
        axes[1].set_ylabel('Number of Transactions')
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(quarterly_sales['Quarter'], quarterly_sales['AvgTransactionValue'], color='mediumseagreen')
        axes[2].set_title('Quarterly Average Transaction Value', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Quarter')
        axes[2].set_ylabel('Average Value ($)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quarterly_sales_trends.png'), dpi=300)
        plt.close()
        print("Quarterly sales trends plot saved.")
        
        return quarterly_sales
    
    def analyze_sales_by_location(self, transactions_df):
        location_sales = transactions_df.groupby('Location')['Amount'].agg(['sum', 'count', 'mean']).reset_index()
        location_sales.columns = ['Location', 'TotalSales', 'TransactionCount', 'AvgTransactionValue']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].bar(location_sales['Location'], location_sales['TotalSales'], color='steelblue')
        axes[0].set_title('Total Sales by Location', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Location')
        axes[0].set_ylabel('Total Sales ($)')
        
        axes[1].bar(location_sales['Location'], location_sales['TransactionCount'], color='coral')
        axes[1].set_title('Transaction Count by Location', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Location')
        axes[1].set_ylabel('Number of Transactions')
        
        axes[2].bar(location_sales['Location'], location_sales['AvgTransactionValue'], color='mediumseagreen')
        axes[2].set_title('Average Transaction Value by Location', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Location')
        axes[2].set_ylabel('Average Value ($)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sales_by_location.png'), dpi=300)
        plt.close()
        print("Sales by location plot saved.")
        
        return location_sales
    
    def analyze_sales_by_card_type(self, transactions_df):
        card_sales = transactions_df.groupby('CardType')['Amount'].agg(['sum', 'count', 'mean']).reset_index()
        card_sales.columns = ['CardType', 'TotalSales', 'TransactionCount', 'AvgTransactionValue']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        colors = {'DIAMOND': '#00CED1', 'PLATINUM': '#E5E4E2', 'GOLD': '#FFD700', 'SILVER': '#C0C0C0'}
        bar_colors = [colors.get(ct, 'gray') for ct in card_sales['CardType']]
        
        axes[0].bar(card_sales['CardType'], card_sales['TotalSales'], color=bar_colors)
        axes[0].set_title('Total Sales by Card Type', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Card Type')
        axes[0].set_ylabel('Total Sales ($)')
        
        axes[1].bar(card_sales['CardType'], card_sales['TransactionCount'], color=bar_colors)
        axes[1].set_title('Transaction Count by Card Type', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Card Type')
        axes[1].set_ylabel('Number of Transactions')
        
        axes[2].bar(card_sales['CardType'], card_sales['AvgTransactionValue'], color=bar_colors)
        axes[2].set_title('Average Transaction Value by Card Type', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Card Type')
        axes[2].set_ylabel('Average Value ($)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sales_by_card_type.png'), dpi=300)
        plt.close()
        print("Sales by card type plot saved.")
        
        return card_sales
    
    def correlate_churn_with_sales(self, transactions_df):
        customer_sales = transactions_df.groupby('CustomerId')['Amount'].agg(['sum', 'count', 'mean']).reset_index()
        customer_sales.columns = ['CustomerId', 'TotalSales', 'TransactionCount', 'AvgTransactionValue']
        
        merged_df = self.df.merge(customer_sales, on='CustomerId', how='left')
        merged_df.fillna(0, inplace=True)
        
        churn_sales = merged_df.groupby('Exited').agg({
            'TotalSales': ['mean', 'sum'],
            'TransactionCount': ['mean', 'sum'],
            'AvgTransactionValue': 'mean'
        }).reset_index()
        
        churn_sales.columns = ['Exited', 'AvgTotalSales', 'TotalSales', 'AvgTransactionCount', 
                              'TotalTransactionCount', 'AvgTransactionValue']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        churn_labels = ['Retained', 'Churned']
        
        axes[0, 0].bar(churn_labels, churn_sales['AvgTotalSales'], color=['green', 'red'])
        axes[0, 0].set_title('Average Total Sales by Churn Status', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Average Total Sales ($)')
        
        axes[0, 1].bar(churn_labels, churn_sales['AvgTransactionCount'], color=['green', 'red'])
        axes[0, 1].set_title('Average Transaction Count by Churn Status', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Average Transaction Count')
        
        axes[1, 0].bar(churn_labels, churn_sales['TotalSales'], color=['green', 'red'])
        axes[1, 0].set_title('Total Sales by Churn Status', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Total Sales ($)')
        
        axes[1, 1].bar(churn_labels, churn_sales['AvgTransactionValue'], color=['green', 'red'])
        axes[1, 1].set_title('Average Transaction Value by Churn Status', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Average Transaction Value ($)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'churn_sales_correlation.png'), dpi=300)
        plt.close()
        print("Churn-sales correlation plot saved.")
        
        return churn_sales, merged_df
    
    def generate_sales_report(self, transactions_df):
        monthly_sales = self.analyze_monthly_sales(transactions_df)
        quarterly_sales = self.analyze_quarterly_sales(transactions_df)
        location_sales = self.analyze_sales_by_location(transactions_df)
        card_sales = self.analyze_sales_by_card_type(transactions_df)
        churn_sales, merged_df = self.correlate_churn_with_sales(transactions_df)
        
        report = {
            'monthly_sales': monthly_sales,
            'quarterly_sales': quarterly_sales,
            'location_sales': location_sales,
            'card_sales': card_sales,
            'churn_sales': churn_sales
        }
        
        return report


if __name__ == "__main__":
    analyzer = SalesAnalyzer()
    analyzer.load_customer_data('Customer-Churn_Records.csv')
    
    transactions_df = analyzer.generate_transaction_data(
        num_customers=1000,
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    
    report = analyzer.generate_sales_report(transactions_df)
    
    print("\n=== Sales Analysis Report ===")
    print(f"Total Transactions: {len(transactions_df)}")
    print(f"Total Sales Volume: ${transactions_df['Amount'].sum():,.2f}")
    print(f"Average Transaction Value: ${transactions_df['Amount'].mean():,.2f}")
    
    print("\nSales analysis completed successfully!")
