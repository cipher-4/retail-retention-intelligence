import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
import os

# Set page config
st.set_page_config(
    page_title="E-Commerce Churn & CLV Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2d324d;
    }
    .stMetric > div {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 12px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Data loading with caching for instant rendering
@st.cache_data
def load_all_data():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    rfm = pd.read_parquet(os.path.join(base_dir, "rfm_customers.parquet"))
    churn = pd.read_parquet(os.path.join(base_dir, "churn_features.parquet"))
    preds = pd.read_parquet(os.path.join(base_dir, "churn_predictions.parquet"))
    clean = pd.read_parquet(os.path.join(base_dir, "clean_retail.parquet"))
    clean['InvoiceDate'] = pd.to_datetime(clean['InvoiceDate'])
    return rfm, churn, preds, clean

@st.cache_resource
def load_churn_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    model_path = os.path.join(base_dir, "xgb_churn_model.json")
    model = XGBClassifier()
    model.load_model(model_path)
    return model

# Load data & model
try:
    rfm_df, churn_df, preds_df, clean_df = load_all_data()
    model = load_churn_model()
except Exception as e:
    st.error(f"Error loading project artifacts: {e}")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/clouds/200/shopping-cart.png", width=120)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Select Module:",
    [
        "📊 Executive Overview",
        "🎯 RFM Segmentation",
        "⚠️ Churn Risk & Retention Prioritization",
        "🔍 Model Performance & Feature Drivers",
        "🧪 Live Customer Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project:** E-Commerce Churn & CLV Pipeline  
**Dataset:** 1M+ Transactions (UK Gift Retailer)  
**Algorithm:** XGBoost Classifier (ROC-AUC: 0.83)  
""")

# ==========================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================================
if menu == "📊 Executive Overview":
    st.title("🛒 E-Commerce Customer Intelligence & Retention Platform")
    st.markdown("Executive summary of customer lifetime activity, segmentation health, and revenue exposure.")

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(rfm_df)
    total_rev = rfm_df['Monetary'].sum()
    avg_order_value = clean_df['TotalPrice'].mean()
    high_risk_rev = preds_df[(preds_df['Predicted_Churn'] == 1) & (preds_df['monetary'] > 1000)]['monetary'].sum()

    col1.metric("Total Profiled Customers", f"{total_customers:,}", "2-Year Cohort")
    col2.metric("Total Gross Revenue", f"£{total_rev:,.0f}", "100% Retained Sales")
    col3.metric("Avg Basket Value", f"£{avg_order_value:.2f}", "Across 805k line items")
    col4.metric("At-Risk Revenue Identified", f"£{high_risk_rev:,.0f}", "Test Sample Exposure", delta_color="inverse")

    st.markdown("---")

    # Monthly Trends
    st.subheader("📈 Monthly Revenue & Active Customers Trend")
    clean_df['YearMonth'] = clean_df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly = clean_df.groupby('YearMonth').agg(
        Monthly_Revenue=('TotalPrice', 'sum'),
        Active_Customers=('Customer ID', 'nunique')
    ).reset_index()

    fig_rev = px.line(
        monthly, x='YearMonth', y='Monthly_Revenue',
        title="Monthly Revenue Trend (£)",
        labels={'YearMonth': 'Month', 'Monthly_Revenue': 'Revenue (£)'},
        markers=True,
        line_shape='spline',
        color_discrete_sequence=['#00CC96']
    )
    fig_rev.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_rev, use_container_width=True)

    # Segment Summary Table
    st.subheader("👥 Customer Segments Breakdown")
    summary = rfm_df.groupby('Segment').agg(
        Count=('CustomerID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Orders=('Frequency', 'mean'),
        Avg_Spend=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum')
    ).round(1).reset_index()
    summary['Revenue_Share_%'] = ((summary['Total_Revenue'] / summary['Total_Revenue'].sum()) * 100).round(1)
    summary = summary.sort_values(by='Total_Revenue', ascending=False)
    st.dataframe(summary, use_container_width=True)

# ==========================================================
# PAGE 2: RFM SEGMENTATION
# ==========================================================
elif menu == "🎯 RFM Segmentation":
    st.title("🎯 RFM (Recency, Frequency, Monetary) Segmentation")
    st.markdown("Deep dive into customer behavioral buckets based on purchasing velocity and monetary scale.")

    col1, col2 = st.columns(2)
    with col1:
        fig_donut = px.pie(
            rfm_df, names='Segment', title="Customer Distribution by Segment",
            hole=0.45, color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_donut.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        seg_rev = rfm_df.groupby('Segment')['Monetary'].sum().reset_index()
        fig_bar = px.bar(
            seg_rev.sort_values(by='Monetary', ascending=True),
            x='Monetary', y='Segment', orientation='h',
            title="Total Revenue Generated by Segment (£)",
            color='Monetary', color_continuous_scale='Viridis',
            labels={'Monetary': 'Revenue (£)'}
        )
        fig_bar.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Interactive 3D Customer Behavior Scatter")
    scatter_sample = rfm_df.sample(min(2000, len(rfm_df)), random_state=42)
    fig_3d = px.scatter_3d(
        scatter_sample, x='Recency', y='Frequency', z='Monetary',
        color='Segment', size='Monetary', size_max=25, opacity=0.7,
        title="3D RFM Clusters (Recency vs Frequency vs Spend)",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig_3d.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig_3d, use_container_width=True)

# ==========================================================
# PAGE 3: CHURN RISK & RETENTION PRIORITIZATION
# ==========================================================
elif menu == "⚠️ Churn Risk & Retention Prioritization":
    st.title("⚠️ High-Risk Churn Identification & Retention Targets")
    st.markdown("Actionable lists of customers predicted to churn in the next 90 days, prioritized by historical financial value.")

    # Filter controls
    col_a, col_b = st.columns([1, 2])
    with col_a:
        threshold = st.slider("Churn Probability Threshold (%)", min_value=50, max_value=95, value=70, step=5)
    with col_b:
        min_spend = st.number_input("Minimum Historical Spend (£) Filter", min_value=0, value=500, step=100)

    # Filtered dataset
    filtered = preds_df[
        (preds_df['Churn_Probability'] >= (threshold / 100.0)) &
        (preds_df['monetary'] >= min_spend)
    ].sort_values(by='monetary', ascending=False)

    # High-level KPIs for this filter
    f_count = len(filtered)
    f_rev = filtered['monetary'].sum()

    st.warning(f"🚨 **{f_count} High-Value Customers** flagged with >={threshold}% churn probability, exposing **£{f_rev:,.2f}** in at-risk revenue.")

    # CSV download button for Marketing Team
    csv = filtered[['CustomerID', 'Churn_Probability', 'monetary', 'recency', 'frequency']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Targeted Retention Campaign List (CSV)",
        data=csv,
        file_name=f"churn_retention_targets_threshold_{threshold}.csv",
        mime="text/csv"
    )

    # Table of targets
    display_cols = ['CustomerID', 'Churn_Probability', 'monetary', 'recency', 'frequency', 'Actual_Churn']
    st.dataframe(
        filtered[display_cols].rename(columns={
            'CustomerID': 'Customer ID',
            'Churn_Probability': 'Predicted Churn Risk',
            'monetary': 'Total Past Spend (£)',
            'recency': 'Days Since Last Order',
            'frequency': 'Past Orders',
            'Actual_Churn': 'True Churned Status'
        }).style.format({
            'Predicted Churn Risk': '{:.1%}',
            'Total Past Spend (£)': '£{:,.2f}'
        }),
        use_container_width=True,
        height=400
    )

# ==========================================================
# PAGE 4: MODEL PERFORMANCE & FEATURE DRIVERS
# ==========================================================
elif menu == "🔍 Model Performance & Feature Drivers":
    st.title("🔍 Model Architecture & Explainability")
    st.markdown("XGBoost Classifier evaluation metrics and behavioral feature importance breakdown.")

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC Score", "0.8301", "Target > 0.80 Met")
    col2.metric("Recall (Churn Detection)", "81%", "Caught 8/10 churners")
    col3.metric("Model Precision", "77%", "Accurate positive calls")

    st.markdown("---")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("Feature Importance (Information Gain)")
        feat_names = [
            'recency', 'purchase_frequency_rate', 'frequency', 'monetary',
            'total_items', 'monthly_spend_rate', 'unique_products',
            'std_order_value', 'max_order_value', 'customer_tenure_days',
            'avg_order_value', 'avg_items_per_order'
        ]
        importances = model.feature_importances_
        imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values(by='Importance', ascending=True)

        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            title="Top Behavioral Factors Driving Customer Churn",
            color='Importance', color_continuous_scale='Plasma'
        )
        fig_imp.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_m2:
        st.subheader("Test Set Confusion Matrix")
        cm = [[327, 142], [108, 473]]
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=['Predicted Retained', 'Predicted Churned'],
            y=['Actual Retained', 'Actual Churned'],
            color_continuous_scale='Blues',
            title="Confusion Matrix (1,050 Holdout Customers)"
        )
        fig_cm.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_cm, use_container_width=True)

# ==========================================================
# PAGE 5: LIVE CUSTOMER PREDICTOR
# ==========================================================
elif menu == "🧪 Live Customer Predictor":
    st.title("🧪 Real-Time Customer Churn Risk Simulator")
    st.markdown("Adjust customer behavioral parameters to calculate real-time churn probability using the trained XGBoost model.")

    col_in1, col_in2, col_in3 = st.columns(3)

    with col_in1:
        recency = st.slider("Days Since Last Purchase (Recency)", min_value=1, max_value=700, value=90)
        frequency = st.slider("Number of Historical Orders (Frequency)", min_value=1, max_value=50, value=4)
        monetary = st.number_input("Total Lifetime Spend (£)", min_value=10.0, max_value=50000.0, value=1200.0, step=50.0)

    with col_in2:
        tenure = st.slider("Customer Tenure (Days since 1st purchase)", min_value=1, max_value=730, value=300)
        total_items = st.number_input("Total Items Purchased", min_value=1, max_value=10000, value=150)
        unique_products = st.number_input("Unique SKUs Purchased", min_value=1, max_value=500, value=25)

    with col_in3:
        avg_order = monetary / max(frequency, 1)
        max_order = st.number_input("Maximum Single Order Spend (£)", min_value=10.0, value=float(avg_order * 1.5), step=20.0)
        std_order = st.number_input("Order Spend Standard Deviation", min_value=0.0, value=float(avg_order * 0.3), step=10.0)

    # Derived rates
    avg_items_per_order = total_items / max(frequency, 1)
    purchase_rate = frequency / (max(tenure, 1) / 30.4)
    monthly_spend_rate = monetary / (max(tenure, 1) / 30.4)

    input_data = pd.DataFrame([{
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'total_items': total_items,
        'unique_products': unique_products,
        'customer_tenure_days': tenure,
        'avg_order_value': avg_order,
        'max_order_value': max_order,
        'std_order_value': std_order,
        'avg_items_per_order': avg_items_per_order,
        'purchase_frequency_rate': purchase_rate,
        'monthly_spend_rate': monthly_spend_rate
    }])

    st.markdown("---")
    if st.button("🚀 Calculate Churn Probability", type="primary"):
        churn_prob = model.predict_proba(input_data)[0, 1]

        st.subheader("Prediction Result:")
        prob_pct = churn_prob * 100

        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            if churn_prob >= 0.70:
                st.error(f"### High Churn Risk: {prob_pct:.1f}%")
                st.write("🚨 **Recommendation:** Immediate proactive intervention! Offer personalized 15% discount or dedicated account outreach.")
            elif churn_prob >= 0.40:
                st.warning(f"### Moderate Churn Risk: {prob_pct:.1f}%")
                st.write("⚠️ **Recommendation:** Re-engagement email sequence with popular new products.")
            else:
                st.success(f"### Low Churn Risk: {prob_pct:.1f}%")
                st.write("✅ **Recommendation:** Healthy active account. Enrol in VIP loyalty reward program.")

        with col_res2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF4B4B" if churn_prob >= 0.7 else "#FFA500" if churn_prob >= 0.4 else "#00CC96"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(0, 204, 150, 0.2)"},
                        {'range': [40, 70], 'color': "rgba(255, 165, 0, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                    ]
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)
