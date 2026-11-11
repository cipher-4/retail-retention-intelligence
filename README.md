# 🛒 Retail Retention Intelligence
### *End-to-End Customer Lifetime Value (CLV) & Churn Prediction Platform*

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v3.4-orange.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.8-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Parquet-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

In non-contractual e-commerce, customers silently churn without explicit cancellation signals. This project establishes an enterprise-grade retention intelligence system built on **1,067,371 real commercial retail transactions** from a UK-based e-commerce merchant (2009–2011).

The platform transforms raw transactional logs into **5,878 behavioral customer profiles**, applies a **leakage-free 90-day time-split**, and trains an **XGBoost Classifier achieving an ROC-AUC of 0.8301** with **81% churn recall**. By linking predictive churn probabilities with customer lifetime financial spend, the system flags **£350,748.58 in high-value at-risk revenue** in the holdout test set alone, enabling targeted retention campaigns.

---

## 📸 Platform Dashboard Preview

| Executive Overview | 3D RFM Segmentation |
| :---: | :---: |
| ![Executive Overview](docs/images/01_overview.png) | ![RFM 3D Clusters](docs/images/02_rfm.png) |

| Churn Risk & Retention Prioritization | Live Customer Predictor |
| :---: | :---: |
| ![Retention Targets](docs/images/03_churn_targets.png) | ![Live Simulator](docs/images/04_simulator.png) |

---

## 🎯 Key Project Highlights & Business Impact

* **Large-Scale Data Engineering:** Ingested and cleaned **1.06M+ raw transactional rows**; resolved guest checkouts, credit cancellations (`Invoice` starting with `C`), bookkeeping fees, and negative quantities.
* **Optimized Columnar Storage:** Converted multi-sheet Excel logs into Apache Parquet (`.parquet`), reducing disk footprint by **85%** and speeding load times from **45 seconds to 0.2 seconds**.
* **10-Tier RFM Behavioral Segmentation:** Profiled 5,878 customers across Recency, Frequency, and Monetary dimensions. Uncovered that **Champions (25.2% of customers) generate 69.3% (£12.29M) of total revenue**.
* **Leakage-Free Predictive Target:** Formulated a realistic 90-day forward prediction window (Sep 1 – Dec 9, 2011) evaluated against 21 months of prior observation history.
* **High-Performance ML Model:** Engineered 12 velocity, frequency, and spend-variance features. Trained an XGBoost model achieving **0.8301 ROC-AUC** and **81% recall** on churners.
* **Actionable ROI Targeting:** Flagged **£350K+ in at-risk revenue** within the test cohort, with a one-click export engine for CRM/Email retention marketing.

---

## 🏗️ System Architecture

```
Raw Transaction Logs (1.06M+ rows, 2 Years)
                     │
                     ▼
  [Phase 1] Data Cleaning & Parquet Ingestion
   • Drop anonymous guest checkouts (243k rows)
   • Filter cancellations (starts with 'C') & negative quantities
   • Clean mixed-type StockCodes & invalid price adjustments
   • Output: 805,549 clean rows (clean_retail.parquet)
                     │
                     ▼
  [Phase 2] RFM Customer Segmentation
   • Aggregation: Recency (days), Frequency (orders), Monetary (total £)
   • Quintile Scoring (1-5) across R, F, M dimensions
   • 10 Business Segments (Champions, Loyal, At Risk, Lost)
                     │
                     ▼
  [Phase 3] Time-Split & Behavioral Feature Engineering
   • Observation Horizon (< Sep 1, 2011) vs 90-Day Target Horizon (>= Sep 1, 2011)
   • 12 Predictors: Purchase velocity, spend volatility, tenure, basket size
   • Binary Churn Label: 1 = No purchases in next 90 days (55.3%), 0 = Repurchased (44.7%)
                     │
                     ▼
  [Phase 4] Predictive Machine Learning (XGBoost)
   • 80/20 Stratified Train/Test Split (4,199 Train / 1,050 Test)
   • Hyperparameter-tuned XGBoost Classifier (ROC-AUC: 0.8301, Recall: 81%)
   • Feature importance analysis (Recency & Frequency Rate drive 41% of decisions)
                     │
                     ▼
  [Phase 5] Streamlit Web Platform
   • Executive KPI cards, interactive 3D clusters, dynamic threshold filtering
   • One-click marketing campaign CSV export & real-time churn risk simulator
```

---

## 📊 Model Performance & Evaluation

The XGBoost model was evaluated on a strictly held-out test set of **1,050 customers** using stratified sampling:

```
======================================================
  ROC-AUC SCORE: 0.8301  |  ACCURACY: 76.0%
======================================================

Classification Report:
              precision    recall  f1-score   support
 Retained (0)      0.75      0.70      0.72       469
  Churned (1)      0.77      0.81      0.79       581

Confusion Matrix:
                    Predicted Retained    Predicted Churned
  Actual Retained           327                  142
  Actual Churned            108                  473
```

### Top Behavioral Churn Drivers (Feature Importance)

| Rank | Feature | Relative Importance | Business Interpretation |
|:---:|---|:---:|---|
| **1** | `recency` | **20.96%** | Number of days since latest order is the strongest churn indicator. |
| **2** | `purchase_frequency_rate` | **19.99%** | Decay in monthly order velocity signals disengagement before complete churn. |
| **3** | `frequency` | **13.11%** | Single-purchase shoppers exhibit a 3x higher churn hazard vs repeat buyers. |
| **4** | `monetary` | **7.83%** | Total lifetime financial commitment. |
| **5** | `total_items` | **6.85%** | Total item volume across all historical carts. |
| **6** | `monthly_spend_rate` | **5.95%** | Average monthly capital contribution. |

---

## 👥 Customer Segmentation Breakdown

| Segment | Customers | Avg Recency | Avg Orders | Avg Spend | Total Revenue (£) | Rev Share |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Champions** 🏆 | 1,482 | 20 days | 15.7 | £8,295.0 | £12,293,138.7 | **69.3%** |
| **Loyal Customers** | 1,772 | 148 days | 5.3 | £2,099.6 | £3,720,502.0 | **21.0%** |
| **Potential Loyalists** | 468 | 61 days | 1.7 | £986.7 | £461,785.9 | 2.6% |
| **Lost / Hibernating** | 902 | 557 days | 1.2 | £431.9 | £389,556.0 | 2.2% |
| **At Risk (High Value)** ⚠️ | 256 | 504 days | 3.6 | £1,382.7 | £353,978.5 | 2.0% |
| **Need Attention** | 318 | 316 days | 1.6 | £519.6 | £165,217.1 | 0.9% |
| **About to Sleep** | 303 | 319 days | 1.0 | £370.8 | £112,348.9 | 0.6% |
| **Can't Lose Them** 🚨 | 17 | 473 days | 16.3 | £6,378.4 | £108,432.7 | 0.6% |
| **Promising** | 192 | 105 days | 1.0 | £430.1 | £82,579.0 | 0.5% |
| **New Customers** | 168 | 30 days | 1.0 | £332.7 | £55,890.6 | 0.3% |
| **Total** | **5,878** | — | — | — | **£17,743,429.0** | **100%** |

---

## 📂 Repository Structure

```text
retail-retention-intelligence/
├── app/
│   └── main.py                     # 5-page interactive Streamlit dashboard
├── data/
│   ├── online+retail+ii/           # Raw Excel dataset (ignored in git)
│   └── processed/                  # Cleaned and engineered Parquet datasets
│       ├── clean_retail.parquet
│       ├── rfm_customers.parquet
│       ├── churn_features.parquet
│       ├── churn_predictions.parquet
│       └── xgb_churn_model.json    # Serialized XGBoost model artifact
├── docs/
│   └── images/                     # Dashboard screenshots for documentation
├── notebooks/
│   ├── data_cleaning_eda.ipynb     # Phase 1: Ingestion, cleaning & sanity checks
│   ├── rfm_segmentaion.ipynb       # Phase 2: RFM calculation & customer segmentation
│   └── 03_churn_model.ipynb        # Phase 3 & 4: Feature engineering & XGBoost modeling
├── .gitignore                      # Git exclusion rules
├── README.md                       # Enterprise project documentation
└── requirements.txt                # Production environment dependencies
```

---

## ⚡ Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/cipher-4/retail-retention-intelligence.git
cd retail-retention-intelligence
```

### 2. Set Up Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app/main.py
```
The application will launch automatically at `http://localhost:8501`.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
