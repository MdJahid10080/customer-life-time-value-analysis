# Customer Lifetime Value Analysis

A Python-based customer segmentation project using **RFM Analysis** and a **Random Forest Classifier**. The project analyzes historical customer behaviour and classifies customers into five segments for customer-retention and marketing insights.

## 🚀 Live Deployment

**Streamlit App:** https://customer-life-time-value-analysis-o5dpapbehwvmafn4exwpzy.streamlit.app/

The interactive application allows users to:
- Enter Recency, Frequency, and Monetary (RFM) values
- Predict a customer segment
- Explore sample customer profiles
- Run what-if RFM scenarios
- View model insights and feature importance
- Understand the five customer segments

## 🤖 Machine Learning

The trained Random Forest model uses these transformed features:

- **Log_Recency** — log-transformed Recency
- **Log_Frequency** — log-transformed Frequency
- **Avg_Order_Value** — Monetary divided by Frequency

The Streamlit app accepts the original RFM values and applies the same feature transformations before prediction.

### Customer Segments

- Champions
- Loyal Customers
- Potential Loyalists
- At Risk
- Lost Customers

## 📊 Model Evaluation

The notebook includes:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Feature Importance

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- RFM Analysis
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 📁 Project Files

- `customer_lifetime_value_analysis.ipynb` — complete analysis and machine-learning workflow
- `app.py` — Streamlit application
- `rf_model.pkl` — trained Random Forest model
- `label_encoder.pkl` — saved target-label encoder
- `feature_columns.pkl` — saved model feature names
- `Online_Retail (1).xlsx` — dataset
- `requirements.txt` — required Python packages

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Developer

**MD JAHID**
