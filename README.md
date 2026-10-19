# Customer Churn Prediction and Sales Dashboard

A comprehensive machine learning project that analyzes customer behavior, predicts churn probability, and visualizes sales trends using advanced analytics and interactive dashboards.

## 📋 Project Overview

This project leverages Data Analytics and Machine Learning to analyze customer behavior, predict churn probability, and visualize sales trends. This system is essential for businesses, e-commerce platforms, and service providers to enhance customer retention and optimize sales strategies.

## ✨ Key Features

### 1. Data Acquisition & Preprocessing
- Collect customer data from CRM systems and databases
- Perform ETL (Extract, Transform, Load) operations
- Handle missing values, normalize numerical data, and encode categorical variables
- Feature engineering to extract purchase frequency, engagement scores, and subscription tenure

### 2. Churn Prediction Models
- **Supervised Learning Models:**
  - Logistic Regression
  - Decision Trees & Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
- **Deep Learning Models:**
  - Artificial Neural Networks (ANNs)
  - Recurrent Neural Networks (RNNs) for time-based churn trends
- **Unsupervised Learning for Customer Segmentation:**
  - K-Means Clustering
  - DBSCAN for anomaly detection

### 3. Sales Trend Analysis
- Track monthly, quarterly, and yearly sales growth
- Identify top-performing products/services
- Correlate churn trends with revenue impact

### 4. Interactive Dashboard & Insights
- Real-time customer retention and churn analysis
- Sales visualization using heatmaps, bar charts, and line graphs
- Customer segmentation insights for targeted marketing

## 🛠️ Technology Stack

### Data Processing & ML Models
- **Python** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **TensorFlow/Keras** - Deep learning models
- **XGBoost/LightGBM** - Gradient boosting frameworks
- **Imbalanced-learn** - Handling imbalanced datasets

### Visualization & Dashboard
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical data visualization
-**Plotly** - Interactive visualizations
- **Streamlit** - Interactive web dashboard 

## 📁 Project Structure

```
Customer_Churn_Prediction/
├── Customer-Churn_Records.csv          # Customer dataset
├── app.py                              # Streamlit dashboard
├── main.py                             # Main pipeline script
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── src/                                # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data preprocessing module
│   ├── churn_models.py                 # Churn prediction models
│   ├── deep_learning_models.py         # Deep learning models (ANN, RNN)
│   ├── customer_segmentation.py       # Customer segmentation (K-Means, DBSCAN)
│   └── sales_analysis.py               # Sales trend analysis
├── models/                             # Saved models and artifacts
├── data/                               # Data storage
└── notebooks/                          # Jupyter notebooks (optional)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd Customer_Churn_Prediction
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the `Customer-Churn_Records.csv` file containing customer information with the following columns:

- **CustomerId**: Unique customer identifier
- **Surname**: Customer surname
- **CreditScore**: Customer's credit score
- **Location**: Customer's location (France, Spain, Germany)
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years with the bank
- **Account Balance**: Account balance
- **NumOfProducts**: Number of products held
- **HasCrCard**: Has credit card (0/1)
- **IsActiveMember**: Is active member (0/1)
- **EstimatedSalary**: Estimated salary
- **Exited**: Churn status (0/1) - Target variable
- **Complain**: Customer complaint (0/1)
- **Satisfaction Score**: Satisfaction score (1-5)
- **Card Type**: Card type (DIAMOND, PLATINUM, GOLD, SILVER)
- **Point Earned**: Points earned

## 🎯 Usage

### Running the Complete Pipeline

To run the entire pipeline (preprocessing, model training, segmentation, and sales analysis):

```bash
python main.py
```

This will:
1. Preprocess the data
2. Train all churn prediction models
3. Train deep learning models
4. Perform customer segmentation
5. Analyze sales trends
6. Save all models and visualizations to the `models/` directory

### Running Individual Components

#### Data Preprocessing
```bash
python src/data_preprocessing.py
```

#### Churn Prediction Models
```bash
python src/churn_models.py
```

#### Deep Learning Models
```bash
python src/deep_learning_models.py
```

#### Customer Segmentation
```bash
python src/customer_segmentation.py
```

#### Sales Analysis
```bash
python src/sales_analysis.py
```

### Launching the Interactive Dashboard

After training the models, launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

#### Dashboard Features

1. **Overview Page**: Key metrics and high-level statistics
2. **Customer Analysis Page**: Customer demographics and behavior analysis
3. **Churn Prediction Page**: Interactive churn prediction for individual customers
4. **Sales Trends Page**: Sales analysis and visualization
5. **Customer Segmentation Page**: Customer segments and characteristics

## 📈 Model Performance

The project implements multiple machine learning models for churn prediction:

- **Logistic Regression**: Baseline model with interpretability
- **Random Forest**: Ensemble method with good performance
- **XGBoost**: Gradient boosting with high accuracy
- **LightGBM**: Fast gradient boosting framework
- **ANN (Artificial Neural Network)**: Deep learning approach

Model evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

## 🎨 Customer Segmentation

The project uses unsupervised learning for customer segmentation:

- **K-Means Clustering**: Groups customers into distinct segments
- **DBSCAN**: Density-based clustering for anomaly detection

Segments include:
- High-value customers
- Loyal customers
- At-risk customers
- Churned customers

## 💰 Sales Analysis

The sales analysis module provides:
- Monthly, quarterly, and yearly sales trends
- Sales by location
- Sales by card type
- Correlation between churn and sales
- Transaction value analysis

## 🔧 Configuration

### Model Parameters

Model parameters can be adjusted in the respective source files:

- `src/churn_models.py`: Adjust ML model hyperparameters
- `src/deep_learning_models.py`: Adjust neural network architecture
- `src/customer_segmentation.py`: Adjust clustering parameters

### Data Preprocessing

Preprocessing steps can be customized in `src/data_preprocessing.py`:
- Feature engineering
- Normalization methods
- Encoding strategies

## 📝 Notes

- The dataset is simulated for demonstration purposes
- Sales transaction data is generated based on customer profiles
- Models are saved in the `models/` directory for reuse
- All visualizations are saved as PNG files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is for educational and demonstration purposes.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

## 🙏 Acknowledgments

- Dataset source: Customer Churn Records
- Built with Python, Scikit-learn, TensorFlow, and Streamlit
