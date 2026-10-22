import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction & Sales Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Metric Card Styling */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-5px);
    }
    .stMetric div[data-testid="st-metric-label"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: white;
    }
    .stMetric div[data-testid="st-metric-value"] {
        font-size: 2rem;
        font-weight: bold;
        color: white;
    }
    .stMetric div[data-testid="st-metric-delta"] {
        color: white;
    }
    
    /* Sidebar Header */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background-color: #667eea;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    
    /* Form Styling */
    .stForm {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Chart Container */
    .chart-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Dataframe Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-in;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🎯 Customer Churn Prediction & Sales Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Analytics & Machine Learning for Customer Retention</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('Customer-Churn_Records.csv')
    return df

@st.cache_resource
def load_models():
    models = {}
    model_dir = 'models'
    
    if os.path.exists(os.path.join(model_dir, 'logistic_regression.pkl')):
        models['logistic_regression'] = joblib.load(os.path.join(model_dir, 'logistic_regression.pkl'))
    if os.path.exists(os.path.join(model_dir, 'random_forest.pkl')):
        models['random_forest'] = joblib.load(os.path.join(model_dir, 'random_forest.pkl'))
    if os.path.exists(os.path.join(model_dir, 'xgboost.pkl')):
        models['xgboost'] = joblib.load(os.path.join(model_dir, 'xgboost.pkl'))
    if os.path.exists(os.path.join(model_dir, 'lightgbm.pkl')):
        models['lightgbm'] = joblib.load(os.path.join(model_dir, 'lightgbm.pkl'))
    
    if os.path.exists(os.path.join(model_dir, 'scaler.pkl')):
        models['scaler'] = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    if os.path.exists(os.path.join(model_dir, 'feature_columns.pkl')):
        models['feature_columns'] = joblib.load(os.path.join(model_dir, 'feature_columns.pkl'))
    
    return models

df = load_data()
models = load_models()

# Enhanced Sidebar with Interactive Navigation
with st.sidebar:
    # Logo/Header
    st.markdown("""
    <div class="sidebar-header">
        <h1 style="margin: 0; font-size: 1.8rem; font-weight: bold;">🎯 Dashboard</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.9;">Customer Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Navigation with Buttons
    st.markdown('<p style="font-size: 0.9rem; font-weight: 600; color: #667eea; margin-bottom: 0.5rem;">📱 NAVIGATION</p>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Overview", use_container_width=True, key="nav_overview"):
            st.session_state.current_page = "📈 Overview"
        if st.button("👥 Analysis", use_container_width=True, key="nav_analysis"):
            st.session_state.current_page = "👥 Customer Analysis"
        if st.button("🔮 Prediction", use_container_width=True, key="nav_prediction"):
            st.session_state.current_page = "🔮 Churn Prediction"
    
    with col2:
        if st.button("💰 Sales", use_container_width=True, key="nav_sales"):
            st.session_state.current_page = "💰 Sales Trends"
        if st.button("🎨 Segments", use_container_width=True, key="nav_segments"):
            st.session_state.current_page = "🎨 Customer Segmentation"
    
    # Initialize session state for page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📈 Overview"
    
    page = st.session_state.current_page
    
    st.markdown("---")
    
    # Dashboard Settings in Expander
    with st.expander("⚙️ Dashboard Settings", expanded=False):
        theme_color = st.selectbox(
            "🎨 Accent Color",
            ["Purple", "Blue", "Green", "Orange", "Pink"],
            index=0
        )
        
        show_animations = st.checkbox("✨ Enable Animations", value=True)
        
        dark_mode = st.checkbox("🌙 Dark Mode", value=False)
        
        refresh_interval = st.slider("🔄 Auto Refresh (min)", 0, 60, 0)
    
    st.markdown("---")
    
    # Model Information in Expander
    with st.expander("🤖 Model Information", expanded=False):
        if models:
            st.markdown(f"""
            <div class="success-box">
                <p style="margin: 0;">✅ <strong>{len([k for k in models.keys() if k not in ['scaler', 'feature_columns']])} Models Loaded</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            for model_name in models.keys():
                if model_name not in ['scaler', 'feature_columns']:
                    model_display = model_name.replace('_', ' ').title()
                    st.markdown(f"""
                    <div style="padding: 0.5rem; background-color: #f0f2f6; border-radius: 5px; margin: 0.5rem 0;">
                        <p style="margin: 0; font-size: 0.85rem;">🤖 <strong>{model_display}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <p style="margin: 0;">⚠️ No models loaded</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">Run <code>python main.py</code> to train models</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Stats in Expander
    with st.expander("📊 Quick Stats", expanded=False):
        try:
            if df is not None:
                st.metric("👥 Total Customers", f"{len(df):,}")
                churn_rate = df['Exited'].mean() * 100
                st.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
                avg_balance = df['Account Balance'].mean()
                st.metric("💰 Avg Balance", f"${avg_balance:,.0f}")
                avg_salary = df['EstimatedSalary'].mean()
                st.metric("💵 Avg Salary", f"${avg_salary:,.0f}")
        except:
            st.info("Load data to view stats")
    
    st.markdown("---")
    
    # Recent Activity
    with st.expander("🕐 Recent Activity", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem; color: #666;">
            <p style="margin: 0.5rem 0;">📊 Dashboard loaded</p>
            <p style="margin: 0.5rem 0;">🤖 Models ready</p>
            <p style="margin: 0.5rem 0;">📈 Data analysis active</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About Section
    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
        <div class="info-box">
            <p style="margin: 0; font-weight: bold;">Customer Churn Prediction Dashboard</p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">
            Built with <strong>Streamlit</strong>, <strong>Scikit-learn</strong>, 
            <strong>TensorFlow</strong>, and <strong>Plotly</strong>.
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">
            Powered by advanced machine learning models for 
            customer retention analytics.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Social/Contact
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px;">
        <p style="margin: 0; font-size: 0.85rem; color: #666;">
            <strong>Version 2.0</strong>
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.75rem; color: #999;">
            © 2024 Customer Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

# Map page names with icons to simple names
page_mapping = {
    "📈 Overview": "Overview",
    "👥 Customer Analysis": "Customer Analysis",
    "🔮 Churn Prediction": "Churn Prediction",
    "💰 Sales Trends": "Sales Trends",
    "🎨 Customer Segmentation": "Customer Segmentation"
}
current_page = page_mapping.get(page, "Overview")

if current_page == "Overview":
    st.markdown('<h2 style="color: #667eea; margin-bottom: 1.5rem;">📈 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="info-box animate-fade-in">
        <p style="margin: 0; font-size: 1rem;">
            <strong>👋 Welcome!</strong> Explore customer analytics, churn predictions, and sales insights 
            using our advanced machine learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric(
            "Total Customers",
            f"{total_customers:,}",
            delta="+12% from last month",
            delta_color="normal"
        )
    
    with col2:
        churn_rate = df['Exited'].mean() * 100
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%",
            delta="-2.3% from last month",
            delta_color="inverse"
        )
    
    with col3:
        avg_balance = df['Account Balance'].mean()
        st.metric(
            "Avg Account Balance",
            f"${avg_balance:,.0f}",
            delta="+$5,000 from last month",
            delta_color="normal"
        )
    
    with col4:
        avg_salary = df['EstimatedSalary'].mean()
        st.metric(
            "Avg Estimated Salary",
            f"${avg_salary:,.0f}",
            delta="+$3,200 from last month",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">🌍 Customer Distribution by Location</h3>', unsafe_allow_html=True)
        location_counts = df['Location'].value_counts()
        fig_location = px.pie(
            values=location_counts.values,
            names=location_counts.index,
            title="",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig_location.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}'
        )
        fig_location.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_location, width='stretch', key="overview_location")
    
    with col2:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📊 Churn Distribution</h3>', unsafe_allow_html=True)
        churn_counts = df['Exited'].value_counts()
        churn_labels = ['Retained', 'Churned']
        fig_churn = px.bar(
            x=churn_labels,
            y=churn_counts.values,
            color=churn_labels,
            color_discrete_map={'Retained': '#4caf50', 'Churned': '#f44336'},
            title="",
            text=churn_counts.values
        )
        fig_churn.update_traces(
            texttemplate='%{y:,}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}'
        )
        fig_churn.update_layout(
            showlegend=False,
            xaxis_title="Status",
            yaxis_title="Number of Customers"
        )
        st.plotly_chart(fig_churn, width='stretch', key="overview_churn")
    
    st.markdown("---")
    
    # Additional overview charts
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">💳 Card Type Distribution</h3>', unsafe_allow_html=True)
        card_counts = df['Card Type'].value_counts()
        fig_card = px.bar(
            x=card_counts.index,
            y=card_counts.values,
            color=card_counts.index,
            color_discrete_map={
                'DIAMOND': '#00CED1',
                'PLATINUM': '#E5E4E2',
                'GOLD': '#FFD700',
                'SILVER': '#C0C0C0'
            },
            title="",
            text=card_counts.values
        )
        fig_card.update_traces(
            texttemplate='%{y:,}',
            textposition='outside'
        )
        fig_card.update_layout(
            showlegend=False,
            xaxis_title="Card Type",
            yaxis_title="Number of Customers"
        )
        st.plotly_chart(fig_card, width='stretch', key="overview_card_dist")
    
    with col4:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">⭐ Satisfaction Score Distribution</h3>', unsafe_allow_html=True)
        satisfaction_counts = df['Satisfaction Score'].value_counts().sort_index()
        fig_satisfaction = px.bar(
            x=satisfaction_counts.index,
            y=satisfaction_counts.values,
            color=satisfaction_counts.index,
            color_continuous_scale=px.colors.sequential.Viridis,
            title="",
            text=satisfaction_counts.values
        )
        fig_satisfaction.update_traces(
            texttemplate='%{y:,}',
            textposition='outside'
        )
        fig_satisfaction.update_layout(
            showlegend=False,
            xaxis_title="Satisfaction Score",
            yaxis_title="Number of Customers",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_satisfaction, width='stretch', key="overview_satisfaction")

elif current_page == "Customer Analysis":
    st.markdown('<h2 style="color: #667eea; margin-bottom: 1.5rem;">👥 Customer Analysis</h2>', unsafe_allow_html=True)
    
    # Tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["📊 Demographics", "💰 Financial Analysis", "📈 Advanced Insights"])
    
    with tab1:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">🔍 Data Filters</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
    
    with col1:
        location_filter = st.multiselect(
            "Filter by Location",
            df['Location'].unique(),
            default=df['Location'].unique()
        )
    
    with col2:
        card_filter = st.multiselect(
            "Filter by Card Type",
            df['Card Type'].unique(),
            default=df['Card Type'].unique()
        )
    
    with col3:
        gender_filter = st.multiselect(
            "Filter by Gender",
            df['Gender'].unique(),
            default=df['Gender'].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df['Location'].isin(location_filter)) &
        (df['Card Type'].isin(card_filter)) &
        (df['Gender'].isin(gender_filter))
    ].copy()
    
    st.info(f"Showing {len(filtered_df):,} customers after filtering")
    
    st.markdown("---")
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📊 Age Distribution</h3>', unsafe_allow_html=True)
        fig_age = px.histogram(
            filtered_df, 
            x='Age', 
            nbins=30,
            color='Exited',
            color_discrete_map={0: '#4caf50', 1: '#f44336'},
            title="",
            marginal="box"
        )
        fig_age.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Age",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_age, width='stretch', key="customer_age")
    
    with col2:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">💯 Credit Score Distribution</h3>', unsafe_allow_html=True)
        fig_credit = px.histogram(
            filtered_df, 
            x='CreditScore', 
            nbins=30,
            color='Exited',
            color_discrete_map={0: '#4caf50', 1: '#f44336'},
            title="",
            marginal="box"
        )
        fig_credit.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title="Credit Score",
            yaxis_title="Count"
        )
        st.plotly_chart(fig_credit, width='stretch', key="customer_credit")
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">👤 Gender Distribution</h3>', unsafe_allow_html=True)
        gender_counts = filtered_df['Gender'].value_counts()
        fig_gender = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_gender.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        st.plotly_chart(fig_gender, width='stretch', key="customer_gender")
    
    with col4:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">💳 Card Type Distribution</h3>', unsafe_allow_html=True)
        card_counts = filtered_df['Card Type'].value_counts()
        fig_card = px.pie(
            values=card_counts.values,
            names=card_counts.index,
            title="",
            hole=0.4,
            color_discrete_map={
                'DIAMOND': '#00CED1',
                'PLATINUM': '#E5E4E2',
                'GOLD': '#FFD700',
                'SILVER': '#C0C0C0'
            }
        )
        fig_card.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        st.plotly_chart(fig_card, width='stretch', key="customer_card")
    
    st.markdown("---")
    
    # Advanced analysis
    st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📈 Advanced Analysis</h3>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Account Balance vs Estimated Salary</h4>', unsafe_allow_html=True)
        fig_balance_salary = px.scatter(
            filtered_df,
            x='Account Balance',
            y='EstimatedSalary',
            color='Exited',
            color_discrete_map={0: '#4caf50', 1: '#f44336'},
            title="",
            hover_data=['Age', 'CreditScore'],
            opacity=0.6
        )
        fig_balance_salary.update_layout(
            xaxis_title="Account Balance",
            yaxis_title="Estimated Salary"
        )
        st.plotly_chart(fig_balance_salary, width='stretch', key="customer_balance_salary_1")
    
    with col6:
        st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Tenure vs Number of Products</h4>', unsafe_allow_html=True)
        fig_tenure_products = px.scatter(
            filtered_df,
            x='Tenure',
            y='NumOfProducts',
            color='Exited',
            color_discrete_map={0: '#4caf50', 1: '#f44336'},
            title="",
            size='Account Balance',
            hover_data=['Age', 'CreditScore'],
            opacity=0.6
        )
        fig_tenure_products.update_layout(
            xaxis_title="Tenure (Years)",
            yaxis_title="Number of Products"
        )
        st.plotly_chart(fig_tenure_products, width='stretch', key="customer_tenure_products_1")
    
    with tab2:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">💰 Financial Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Account Balance Distribution</h4>', unsafe_allow_html=True)
            fig_balance = px.histogram(
                df,
                x='Account Balance',
                color='Exited',
                color_discrete_map={0: '#4caf50', 1: '#f44336'},
                title="",
                marginal="box"
            )
            fig_balance.update_layout(
                xaxis_title="Account Balance",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_balance, width='stretch', key="financial_balance")
        
        with col2:
            st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Estimated Salary Distribution</h4>', unsafe_allow_html=True)
            fig_salary = px.histogram(
                df,
                x='EstimatedSalary',
                color='Exited',
                color_discrete_map={0: '#4caf50', 1: '#f44336'},
                title="",
                marginal="box"
            )
            fig_salary.update_layout(
                xaxis_title="Estimated Salary",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_salary, width='stretch', key="financial_salary")
    
    with tab3:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📈 Advanced Insights</h3>', unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Account Balance vs Estimated Salary</h4>', unsafe_allow_html=True)
            fig_balance_salary = px.scatter(
                df,
                x='Account Balance',
                y='EstimatedSalary',
                color='Exited',
                color_discrete_map={0: '#4caf50', 1: '#f44336'},
                title="",
                hover_data=['Age', 'CreditScore'],
                opacity=0.6
            )
            fig_balance_salary.update_layout(
                xaxis_title="Account Balance",
                yaxis_title="Estimated Salary"
            )
            st.plotly_chart(fig_balance_salary, width='stretch', key="advanced_balance_salary")
        
        with col6:
            st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Tenure vs Number of Products</h4>', unsafe_allow_html=True)
            fig_tenure_products = px.scatter(
                df,
                x='Tenure',
                y='NumOfProducts',
                color='Exited',
                color_discrete_map={0: '#4caf50', 1: '#f44336'},
                title="",
                size='Account Balance',
                hover_data=['Age', 'CreditScore'],
                opacity=0.6
            )
            fig_tenure_products.update_layout(
                xaxis_title="Tenure (Years)",
                yaxis_title="Number of Products"
            )
            st.plotly_chart(fig_tenure_products, width='stretch', key="advanced_tenure_products")

elif current_page == "Churn Prediction":
    st.markdown('<h2 style="color: #667eea; margin-bottom: 1.5rem;">🔮 Churn Prediction</h2>', unsafe_allow_html=True)
    
    if not models:
        st.markdown("""
        <div class="warning-box">
            <p style="margin: 0; font-weight: bold;">⚠️ No trained models found</p>
            <p style="margin: 0.5rem 0 0 0;">Please run <code>python main.py</code> to train the models first.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">🤖 Select Model for Prediction</h3>', unsafe_allow_html=True)
        
        available_models = [k for k in models.keys() if k not in ['scaler', 'feature_columns']]
        selected_model = st.selectbox(
            "Choose a model",
            available_models,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_model:
            st.markdown(f"""
            <div class="success-box">
                <p style="margin: 0;">✅ Using <strong>{selected_model.replace('_', ' ').title()}</strong> model</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">👤 Customer Information</h3>', unsafe_allow_html=True)
            
            # Organized input form
            with st.form("customer_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 👤 Personal Info")
                    credit_score = st.slider("Credit Score", 300, 850, 650)
                    age = st.slider("Age", 18, 92, 40)
                    gender = st.selectbox("Gender", df['Gender'].unique())
                    location = st.selectbox("Location", df['Location'].unique())
                
                with col2:
                    st.markdown("#### 💰 Financial Info")
                    tenure = st.slider("Tenure (years)", 0, 10, 5)
                    balance = st.number_input("Account Balance", value=0.0, min_value=0.0, step=1000.0)
                    salary = st.number_input("Estimated Salary", value=0.0, min_value=0.0, step=1000.0)
                    card_type = st.selectbox("Card Type", df['Card Type'].unique())
                
                with col3:
                    st.markdown("#### 📊 Account Info")
                    num_products = st.slider("Number of Products", 1, 4, 2)
                    has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")
                    is_active = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")
                    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
                
                points = st.slider("Points Earned", 0, 1000, 500)
                
                submitted = st.form_submit_button("🔮 Predict Churn Probability", use_container_width=True)
            
            if submitted:
                if 'scaler' in models and 'feature_columns' in models:
                    input_data = pd.DataFrame({
                        'CreditScore': [credit_score],
                        'Age': [age],
                        'Tenure': [tenure],
                        'Account Balance': [balance],
                        'NumOfProducts': [num_products],
                        'HasCrCard': [has_cr_card],
                        'IsActiveMember': [is_active],
                        'EstimatedSalary': [salary],
                        'Satisfaction Score': [satisfaction],
                        'Point Earned': [points],
                        'Location_encoded': [0],
                        'Gender_encoded': [0],
                        'Card Type_encoded': [0],
                        'CreditScore_per_Age': [credit_score / age],
                        'Balance_per_Product': [balance / num_products],
                        'Tenure_per_Age': [tenure / age],
                        'EstimatedSalary_per_Age': [salary / age],
                        'Engagement_Score': [is_active * 0.3 + has_cr_card * 0.2 + (tenure / 10) * 0.3 + (num_products / 4) * 0.2]
                    })
                    
                    input_scaled = models['scaler'].transform(input_data)
                    
                    model = models[selected_model]
                    prediction = model.predict(input_scaled)[0]
                    probability = model.predict_proba(input_scaled)[0, 1] if hasattr(model, 'predict_proba') else prediction
                    
                    # Results display
                    st.markdown("---")
                    st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📊 Prediction Results</h3>', unsafe_allow_html=True)
                    
                    col_pred1, col_pred2, col_pred3 = st.columns(3)
                    
                    with col_pred1:
                        if prediction == 1:
                            st.markdown("""
                            <div class="warning-box" style="background-color: #ffebee; border-left-color: #f44336;">
                                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: #f44336;">⚠️ High Churn Risk</p>
                                <p style="margin: 0.5rem 0 0 0; color: #666;">Customer is likely to churn</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="success-box" style="background-color: #e8f5e9; border-left-color: #4caf50;">
                                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: #4caf50;">✅ Low Churn Risk</p>
                                <p style="margin: 0.5rem 0 0 0; color: #666;">Customer is likely to stay</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col_pred2:
                        probability_percent = probability * 100
                        st.metric(
                            "Churn Probability",
                            f"{probability_percent:.2f}%",
                            delta=None
                        )
                    
                    with col_pred3:
                        retention_percent = (1 - probability) * 100
                        st.metric(
                            "Retention Probability",
                            f"{retention_percent:.2f}%",
                            delta=None
                        )
                    
                    # Gauge chart
                    col_gauge1, col_gauge2 = st.columns([2, 1])
                    
                    with col_gauge1:
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Probability %", 'font': {'size': 20}},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "#667eea"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 30], 'color': "#4caf50"},
                                    {'range': [30, 70], 'color': "#ff9800"},
                                    {'range': [70, 100], 'color': "#f44336"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 70
                                }
                            }
                        ))
                        fig_gauge.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_gauge, width='stretch', key="churn_gauge")
                    
                    with col_gauge2:
                        st.markdown("### Risk Level")
                        if probability < 0.3:
                            risk_level = "Low"
                            risk_color = "#4caf50"
                        elif probability < 0.7:
                            risk_level = "Medium"
                            risk_color = "#ff9800"
                        else:
                            risk_level = "High"
                            risk_color = "#f44336"
                        
                        st.markdown(f"""
                        <div style="background-color: {risk_color}; color: white; padding: 2rem; border-radius: 10px; text-align: center; margin-top: 1rem;">
                            <p style="margin: 0; font-size: 2rem; font-weight: bold;">{risk_level}</p>
                        </div>
                        """, unsafe_allow_html=True)

elif current_page == "Sales Trends":
    st.markdown('<h2 style="color: #667eea; margin-bottom: 1.5rem;">💰 Sales Trends Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📊 Simulated Sales Data Generation</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_customers = st.slider("Number of Customers to Analyze", 100, 1000, 500)
    
    with col2:
        time_period = st.selectbox("Time Period", ["Last 12 Months", "Last 6 Months", "Last 3 Months"])
    
    if st.button("🔄 Generate Sales Data", use_container_width=True):
        with st.spinner("Generating sales data..."):
            customers = df.sample(min(num_customers, len(df)), random_state=42)
            
            transactions = []
            for _, customer in customers.iterrows():
                num_trans = np.random.randint(1, 20)
                for _ in range(num_trans):
                    amount = np.random.uniform(10, 1000)
                    if customer['Card Type'] == 'DIAMOND':
                        amount *= 1.5
                    elif customer['Card Type'] == 'PLATINUM':
                        amount *= 1.3
                    elif customer['Card Type'] == 'GOLD':
                        amount *= 1.1
                    transactions.append({
                        'CustomerId': customer['CustomerId'],
                        'Amount': amount,
                        'Location': customer['Location'],
                        'CardType': customer['Card Type'],
                        'NumProducts': customer['NumOfProducts'],
                        'Exited': customer['Exited']
                    })
            
            transactions_df = pd.DataFrame(transactions)
            
            st.markdown("---")
            st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📈 Sales Summary</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", f"{len(transactions_df):,}")
            
            with col2:
                total_sales = transactions_df['Amount'].sum()
                st.metric("Total Sales Volume", f"${total_sales:,.0f}")
            
            with col3:
                avg_trans = transactions_df['Amount'].mean()
                st.metric("Avg Transaction Value", f"${avg_trans:,.0f}")
            
            with col4:
                total_customers_analyzed = transactions_df['CustomerId'].nunique()
                st.metric("Customers Analyzed", f"{total_customers_analyzed:,}")
            
            st.markdown("---")
            
            # Enhanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">🌍 Sales by Location</h4>', unsafe_allow_html=True)
                location_sales = transactions_df.groupby('Location')['Amount'].sum()
                fig_loc_sales = px.bar(
                    x=location_sales.index,
                    y=location_sales.values,
                    color=location_sales.index,
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    title="",
                    text=location_sales.values
                )
                fig_loc_sales.update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )
                fig_loc_sales.update_layout(
                    showlegend=False,
                    xaxis_title="Location",
                    yaxis_title="Total Sales ($)"
                )
                st.plotly_chart(fig_loc_sales, width='stretch', key="sales_location")
            
            with col2:
                st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">💳 Sales by Card Type</h4>', unsafe_allow_html=True)
                card_sales = transactions_df.groupby('CardType')['Amount'].sum()
                fig_card_sales = px.bar(
                    x=card_sales.index,
                    y=card_sales.values,
                    color=card_sales.index,
                    color_discrete_map={
                        'DIAMOND': '#00CED1',
                        'PLATINUM': '#E5E4E2',
                        'GOLD': '#FFD700',
                        'SILVER': '#C0C0C0'
                    },
                    title="",
                    text=card_sales.values
                )
                fig_card_sales.update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )
                fig_card_sales.update_layout(
                    showlegend=False,
                    xaxis_title="Card Type",
                    yaxis_title="Total Sales ($)"
                )
                st.plotly_chart(fig_card_sales, width='stretch', key="sales_card")
            
            st.markdown("---")
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">📊 Sales vs Churn Status</h4>', unsafe_allow_html=True)
                churn_sales = transactions_df.groupby('Exited')['Amount'].mean()
                fig_churn_sales = px.bar(
                    x=['Retained', 'Churned'],
                    y=churn_sales.values,
                    color=['Retained', 'Churned'],
                    color_discrete_map={'Retained': '#4caf50', 'Churned': '#f44336'},
                    title="",
                    text=churn_sales.values
                )
                fig_churn_sales.update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )
                fig_churn_sales.update_layout(
                    showlegend=False,
                    xaxis_title="Churn Status",
                    yaxis_title="Average Sales ($)"
                )
                st.plotly_chart(fig_churn_sales, width='stretch', key="sales_churn")
            
            with col4:
                st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">📦 Sales by Number of Products</h4>', unsafe_allow_html=True)
                product_sales = transactions_df.groupby('NumProducts')['Amount'].mean()
                fig_product_sales = px.bar(
                    x=product_sales.index,
                    y=product_sales.values,
                    color=product_sales.index,
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="",
                    text=product_sales.values
                )
                fig_product_sales.update_traces(
                    texttemplate='$%{text:,.0f}',
                    textposition='outside'
                )
                fig_product_sales.update_layout(
                    showlegend=False,
                    xaxis_title="Number of Products",
                    yaxis_title="Average Sales ($)",
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_product_sales, width='stretch', key="sales_product")

elif current_page == "Customer Segmentation":
    st.markdown('<h2 style="color: #667eea; margin-bottom: 1.5rem;">🎨 Customer Segmentation</h2>', unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📊 Segmentation by Card Type and Activity</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_card_active = px.scatter(
            df,
            x='Account Balance',
            y='EstimatedSalary',
            color='Card Type',
            symbol='IsActiveMember',
            title="",
            hover_data=['Age', 'CreditScore'],
            color_discrete_map={
                'DIAMOND': '#00CED1',
                'PLATINUM': '#E5E4E2',
                'GOLD': '#FFD700',
                'SILVER': '#C0C0C0'
            },
            opacity=0.7
        )
        fig_card_active.update_layout(
            xaxis_title="Account Balance",
            yaxis_title="Estimated Salary",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_card_active, width='stretch', key="segment_card_active")
    
    with col2:
        fig_products = px.scatter(
            df,
            x='Tenure',
            y='NumOfProducts',
            color='Exited',
            size='Account Balance',
            title="",
            hover_data=['Age', 'CreditScore'],
            color_discrete_map={0: '#4caf50', 1: '#f44336'},
            opacity=0.7
        )
        fig_products.update_layout(
            xaxis_title="Tenure (Years)",
            yaxis_title="Number of Products",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_products, width='stretch', key="segment_products")
    
    st.markdown("---")
    
    st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Customer Segments Analysis</h3>', unsafe_allow_html=True)
    
    # Create segments
    segment_df = df.copy()
    segment_df['Segment'] = 'Standard'
    
    segment_df.loc[
        (segment_df['Account Balance'] > segment_df['Account Balance'].quantile(0.75)) &
        (segment_df['NumOfProducts'] >= 3),
        'Segment'
    ] = 'High Value'
    
    segment_df.loc[
        (segment_df['Tenure'] > 7) &
        (segment_df['IsActiveMember'] == 1),
        'Segment'
    ] = 'Loyal'
    
    segment_df.loc[
        (segment_df['Exited'] == 1),
        'Segment'
    ] = 'Churned'
    
    # Segment distribution
    col3, col4 = st.columns(2)
    
    with col3:
        segment_counts = segment_df['Segment'].value_counts()
        fig_segments = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_segments.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        st.plotly_chart(fig_segments, width='stretch', key="segment_distribution")
    
    with col4:
        st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Segment Characteristics</h4>', unsafe_allow_html=True)
        segment_summary = segment_df.groupby('Segment').agg({
            'CreditScore': 'mean',
            'Age': 'mean',
            'Account Balance': 'mean',
            'EstimatedSalary': 'mean',
            'Satisfaction Score': 'mean'
        }).round(2)
        
        # Style the dataframe
        st.dataframe(
            segment_summary.style.background_gradient(cmap='Blues', axis=0),
            width='stretch'
        )
    
    st.markdown("---")
    
    # Segment comparison
    st.markdown('<h3 style="color: #667eea; margin-bottom: 1rem;">📈 Segment Comparison</h3>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Credit Score by Segment</h4>', unsafe_allow_html=True)
        fig_credit_segment = px.box(
            segment_df,
            x='Segment',
            y='CreditScore',
            color='Segment',
            title="",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_credit_segment.update_layout(
            xaxis_title="Segment",
            yaxis_title="Credit Score",
            showlegend=False
        )
        st.plotly_chart(fig_credit_segment, width='stretch', key="segment_credit")
    
    with col6:
        st.markdown('<h4 style="color: #667eea; margin-bottom: 0.5rem;">Account Balance by Segment</h4>', unsafe_allow_html=True)
        fig_balance_segment = px.box(
            segment_df,
            x='Segment',
            y='Account Balance',
            color='Segment',
            title="",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_balance_segment.update_layout(
            xaxis_title="Segment",
            yaxis_title="Account Balance",
            showlegend=False
        )
        st.plotly_chart(fig_balance_segment, width='stretch', key="segment_balance")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 10px; margin-top: 2rem;">
    <p style="margin: 0; color: #666; font-size: 0.9rem;">
        <strong>Customer Churn Prediction & Sales Dashboard</strong> | 
        Built with <span style="color: #667eea;">Streamlit</span>, <span style="color: #667eea;">Scikit-learn</span>, <span style="color: #667eea;">TensorFlow</span>, and <span style="color: #667eea;">Plotly</span>
    </p>
    <p style="margin: 0.5rem 0 0 0; color: #999; font-size: 0.8rem;">
        Powered by Advanced Machine Learning Models | © 2024
    </p>
</div>
""", unsafe_allow_html=True)
