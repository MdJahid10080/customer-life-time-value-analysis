import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Lifetime Value Analysis",
    page_icon="📊",
    layout="wide",
)

# Load trained model and supporting objects
@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, label_encoder, features

try:
    model, label_encoder, features = load_assets()
except Exception as e:
    st.error("Unable to load the trained model files. Please check that rf_model.pkl, label_encoder.pkl and feature_columns.pkl are present in the repository.")
    st.stop()

# Header
st.title("📊 Customer Lifetime Value Analysis")
st.caption("Interactive customer segmentation using RFM analysis and Random Forest")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Customer Predictor", "RFM Analysis", "Business Insights", "About"],
)
st.sidebar.divider()
st.sidebar.info("Developed by MD JAHID")

if page == "Home":
    st.header("Welcome")
    st.write(
        "This application analyzes customer purchasing behavior and predicts a customer value segment using Recency, Frequency and Monetary (RFM) features."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Random Forest")
    c2.metric("Core Features", "RFM")
    c3.metric("Interface", "Streamlit")

    st.subheader("Project Workflow")
    st.markdown(
        "1. Load retail transaction data\n"
        "2. Clean and preprocess the data\n"
        "3. Create customer-level RFM features\n"
        "4. Generate business-rule-based customer segments\n"
        "5. Train and evaluate a Random Forest classifier\n"
        "6. Predict customer segments through this interactive app"
    )

    st.info(
        "Note: this project focuses on customer segmentation based on historical RFM behavior. It does not directly forecast a future monetary CLV amount."
    )

elif page == "Customer Predictor":
    st.header("🔮 Customer Segment Predictor")
    st.write("Enter customer RFM values to predict the corresponding customer segment.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0, max_value=5000, value=30, step=1)
    with col2:
        frequency = st.number_input("Frequency (orders)", min_value=0, max_value=10000, value=10, step=1)
    with col3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=10000000.0, value=1000.0, step=100.0)

    if st.button("Predict Segment", type="primary", use_container_width=True):
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=features)
        prediction = model.predict(input_df)
        segment = label_encoder.inverse_transform(prediction)[0]

        st.success(f"Predicted Segment: {segment}")
        st.subheader("Customer Profile")
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        st.caption("The prediction is based on the trained Random Forest model and the RFM feature values provided above.")

elif page == "RFM Analysis":
    st.header("📈 RFM Analysis")
    st.write("RFM measures customer purchasing behavior using three dimensions:")

    r1, r2, r3 = st.columns(3)
    r1.metric("Recency", "How recently")
    r2.metric("Frequency", "How often")
    r3.metric("Monetary", "How much")

    st.markdown("### Interpretation")
    st.markdown(
        "- **Recency:** Lower values generally indicate more recent purchases.\n"
        "- **Frequency:** Higher values indicate more frequent purchases.\n"
        "- **Monetary:** Higher values indicate greater historical spending."
    )

    st.info("The project's Segment target is created from RFM-based business rules, and the classifier uses the same RFM variables as model inputs.")

elif page == "Business Insights":
    st.header("💡 Business Insights")
    st.write("RFM-based segmentation can support practical customer-management decisions.")

    with st.expander("High-value customers"):
        st.write("Identify customers with strong purchase frequency and monetary contribution and consider retention-focused strategies.")

    with st.expander("Loyal customers"):
        st.write("Customers with repeated purchases can be considered for loyalty programs, personalized offers and engagement campaigns.")

    with st.expander("At-risk customers"):
        st.write("Customers with increasing recency may require re-engagement campaigns, reminders or targeted offers.")

    with st.expander("Low-activity customers"):
        st.write("Lower-engagement customers can be analyzed for suitable activation campaigns and cost-effective communication.")

    st.warning("These are business interpretations of historical customer behavior, not guarantees of future customer actions.")

elif page == "About":
    st.header("ℹ️ About the Project")

    st.markdown("### Customer Lifetime Value Analysis")
    st.write(
        "This academic project uses the Online Retail dataset to transform transaction-level information into customer-level RFM features and customer segments. A Random Forest Classifier is then used to classify the segments."
    )

    st.markdown("### Technologies")
    st.write("Python • Pandas • NumPy • Scikit-learn • Joblib • Streamlit")

    st.markdown("### Developer")
    st.write("MD JAHID")
    st.write("Jagan Nath University, Bahadurgarh (NCR)")

    st.markdown("### Limitation")
    st.write(
        "Because the segment labels are generated from RFM-based rules and the model uses related RFM features, very high classification scores should be interpreted carefully."
    )

    st.markdown("### Future Scope")
    st.markdown(
        "- Numerical future CLV forecasting\n"
        "- Customer churn prediction\n"
        "- Additional behavioral and purchase-interval features\n"
        "- Model comparison and time-based validation\n"
        "- Personalized recommendations and advanced dashboards"
    )
