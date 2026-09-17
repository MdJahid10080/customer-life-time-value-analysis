import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CLV Analytics | MD JAHID",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme and reusable UI helpers
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #eef3f8 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
    }
    [data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #111827 0%, #334155 55%, #0f766e 100%);
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.18);
    }
    .hero h1 { margin: 0; font-size: 2.5rem; }
    .hero p { margin: .55rem 0 0; color: #dbeafe; font-size: 1.05rem; }
    .card {
        padding: 1.1rem 1.25rem;
        border-radius: 18px;
        background: rgba(255,255,255,.92);
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 22px rgba(15,23,42,.07);
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: .35rem .75rem;
        border-radius: 999px;
        background: #ccfbf1;
        color: #115e59;
        font-weight: 700;
        font-size: .85rem;
    }
    .big-result {
        padding: 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #ecfeff, #f0fdfa);
        border: 1px solid #99f6e4;
        text-align: center;
    }
    .big-result .label { color: #475569; font-size: .95rem; }
    .big-result .value { color: #115e59; font-size: 2rem; font-weight: 800; margin-top: .25rem; }
    .footer {
        text-align: center;
        color: #64748b;
        padding: 1.5rem 0 .5rem;
        font-size: .85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Model assets
# -----------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, label_encoder, features

try:
    model, label_encoder, features = load_assets()
except Exception:
    st.error(
        "The trained model files could not be loaded. Please ensure rf_model.pkl, "
        "label_encoder.pkl and feature_columns.pkl are present in the repository."
    )
    st.stop()

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.markdown("## 💎 CLV Analytics")
st.sidebar.caption("Customer value intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Explore",
    [
        "🏠 Dashboard",
        "🔮 Segment Predictor",
        "📊 RFM Explorer",
        "💡 Business Insights",
        "ℹ️ Project Info",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Model**")
st.sidebar.success("Random Forest • RFM")
st.sidebar.markdown("**Developer**")
st.sidebar.info("MD JAHID")

# -----------------------------
# Dashboard
# -----------------------------
if page == "🏠 Dashboard":
    st.markdown(
        """
        <div class="hero">
            <span class="badge">LIVE ANALYTICS DEMO</span>
            <h1>Customer Lifetime Value Analysis</h1>
            <p>Turn customer purchasing behavior into clear, actionable segments using RFM analysis and machine learning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("At a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Random Forest")
    c2.metric("Features", "3 RFM")
    c3.metric("Segments", "5")
    c4.metric("Interface", "Streamlit")

    st.markdown("### ✨ What can you do here?")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="card"><h4>🔮 Predict</h4><p>Enter Recency, Frequency and Monetary values and instantly classify a customer.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><h4>📊 Explore RFM</h4><p>Interact with the RFM sliders and see how customer behavior changes across dimensions.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="card"><h4>💡 Act on insights</h4><p>Understand how different customer groups can support retention and engagement planning.</p></div>', unsafe_allow_html=True)

    st.markdown("### 🔄 Project pipeline")
    steps = [
        ("01", "Transactions", "Raw retail records"),
        ("02", "Cleaning", "Prepare usable data"),
        ("03", "RFM", "Build customer features"),
        ("04", "Segmentation", "Create business labels"),
        ("05", "ML", "Random Forest classification"),
        ("06", "Insights", "Support customer decisions"),
    ]
    cols = st.columns(6)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="card"><strong>{num}</strong><br><b>{title}</b><br><small>{desc}</small></div>', unsafe_allow_html=True)

    st.info(
        "Scope note: this project performs customer-value segmentation from historical RFM behavior. "
        "It does not directly forecast a future monetary CLV amount."
    )

# -----------------------------
# Segment Predictor
# -----------------------------
elif page == "🔮 Segment Predictor":
    st.markdown("## 🔮 Customer Segment Predictor")
    st.caption("Adjust the customer profile and run the trained Random Forest model.")

    presets = {
        "Custom profile": (30, 10, 1000.0),
        "Frequent recent buyer": (7, 35, 4500.0),
        "High-value recent buyer": (12, 22, 8500.0),
        "Occasional buyer": (120, 5, 600.0),
        "Inactive buyer": (365, 2, 250.0),
    }
    preset = st.selectbox("⚡ Try a sample profile", list(presets.keys()))
    default_recency, default_frequency, default_monetary = presets[preset]

    with st.form("predict_form"):
        st.markdown("### Customer RFM profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input(
                "Recency (days)", min_value=0, max_value=5000,
                value=default_recency, step=1,
                help="Number of days since the customer's latest purchase.",
            )
        with col2:
            frequency = st.number_input(
                "Frequency (orders)", min_value=0, max_value=10000,
                value=default_frequency, step=1,
                help="Number of customer orders in the analyzed period.",
            )
        with col3:
            monetary = st.number_input(
                "Monetary (total spend)", min_value=0.0, max_value=10000000.0,
                value=float(default_monetary), step=100.0,
                help="Total historical customer spending.",
            )
        submitted = st.form_submit_button("🚀 Predict Customer Segment", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=features)
        prediction = model.predict(input_df)
        segment = label_encoder.inverse_transform(prediction)[0]

        st.markdown(
            f'<div class="big-result"><div class="label">Predicted Customer Segment</div><div class="value">{segment}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### 📌 Customer profile")
        p1, p2, p3 = st.columns(3)
        p1.metric("Recency", f"{recency} days")
        p2.metric("Frequency", f"{frequency} orders")
        p3.metric("Monetary", f"₹{monetary:,.0f}")

        st.dataframe(input_df, use_container_width=True, hide_index=True)

        segment_guidance = {
            "Champions": "Strong recent, frequent and valuable behavior. Consider retention and loyalty-focused engagement.",
            "Loyal Customers": "Repeated purchasing behavior can support loyalty programs and personalized communication.",
            "Potential Loyalists": "Promising customers may benefit from engagement and conversion campaigns.",
            "At Risk": "Higher recency may indicate reduced recent activity. Consider re-engagement strategies.",
            "Lost Customers": "Long inactivity can be a signal to review cost-effective win-back opportunities.",
        }
        guidance = segment_guidance.get(segment, "Review the customer's RFM profile and historical behavior before taking action.")
        st.info(f"**Business interpretation:** {guidance}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = label_encoder.inverse_transform(model.classes_)
            probability_df = pd.DataFrame({"Segment": classes, "Model probability": probabilities})
            probability_df = probability_df.sort_values("Model probability", ascending=False).set_index("Segment")
            st.markdown("### 📈 Model confidence profile")
            st.bar_chart(probability_df)

# -----------------------------
# RFM Explorer
# -----------------------------
elif page == "📊 RFM Explorer":
    st.markdown("## 📊 Interactive RFM Explorer")
    st.caption("Move the controls to understand the three dimensions used by the model.")

    col1, col2, col3 = st.columns(3)
    with col1:
        r = st.slider("Recency", 0, 500, 30, 1, help="Lower is generally more recent.")
    with col2:
        f = st.slider("Frequency", 1, 100, 10, 1, help="Higher indicates more purchase activity.")
    with col3:
        m = st.slider("Monetary", 100, 10000, 1000, 100, help="Higher indicates greater historical spending.")

    explorer_df = pd.DataFrame({"Metric": ["Recency", "Frequency", "Monetary"], "Value": [r, f, m]})
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("### Current profile")
        st.dataframe(explorer_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Relative RFM values")
        st.bar_chart(explorer_df.set_index("Metric"))

    st.markdown("### 🧠 How to read RFM")
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown('<div class="card"><h4>🕐 Recency</h4><p>Measures how recently a customer purchased. Lower values generally mean more recent activity.</p></div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="card"><h4>🔁 Frequency</h4><p>Measures how often a customer purchased. Higher values indicate repeated purchasing.</p></div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="card"><h4>💰 Monetary</h4><p>Measures historical spending. Higher values indicate greater monetary contribution.</p></div>', unsafe_allow_html=True)

    st.warning("The RFM Explorer is an educational interactive view. It does not replace analysis of the complete transaction history.")

# -----------------------------
# Business Insights
# -----------------------------
elif page == "💡 Business Insights":
    st.markdown("## 💡 Business Insights")
    st.caption("Use the segment definitions as a starting point for customer-management analysis.")

    insights = [
        ("🏆 Champions", "Strong recent, frequent and valuable behavior.", "Retention, loyalty benefits and personalized engagement."),
        ("❤️ Loyal Customers", "Customers with repeated purchase behavior.", "Loyalty programs, cross-sell and relationship campaigns."),
        ("🌱 Potential Loyalists", "Customers showing promising purchasing behavior.", "Nurture engagement and encourage repeat purchases."),
        ("⚠️ At Risk", "Customers whose recency suggests declining recent activity.", "Re-engagement messages, reminders and targeted offers."),
        ("🔄 Lost Customers", "Customers with substantial inactivity.", "Cost-effective win-back testing and reactivation analysis."),
    ]

    for title, meaning, action in insights:
        with st.expander(title):
            st.write(f"**What it means:** {meaning}")
            st.write(f"**Possible business response:** {action}")

    st.markdown("### 🎯 Decision framework")
    framework = pd.DataFrame(
        {
            "RFM signal": ["Recent + frequent + high spend", "Frequent repeat purchases", "Promising but developing", "Longer recency", "Very low recent activity"],
            "Analysis focus": ["Retention", "Loyalty", "Nurturing", "Re-engagement", "Win-back"],
        }
    )
    st.dataframe(framework, use_container_width=True, hide_index=True)
    st.warning("These are interpretations of historical customer behavior, not guarantees of future customer actions.")

# -----------------------------
# Project Info
# -----------------------------
else:
    st.markdown("## ℹ️ Project Information")
    st.markdown(
        '<div class="card"><h3>Customer Lifetime Value Analysis</h3><p>An academic data-science project that transforms retail transaction behavior into customer-level RFM features and value-based segments.</p></div>',
        unsafe_allow_html=True,
    )

    t1, t2 = st.tabs(["🛠️ Technology", "📚 Methodology"])
    with t1:
        st.markdown("**Core technologies**")
        st.write("Python • Pandas • NumPy • Scikit-learn • Joblib • Streamlit")
        st.markdown("**Machine learning**")
        st.write("Random Forest Classifier")
        st.markdown("**Developer**")
        st.write("MD JAHID")
        st.write("Jagan Nath University, Bahadurgarh (NCR)")
    with t2:
        st.markdown("1. Clean transaction data")
        st.markdown("2. Create customer-level Recency, Frequency and Monetary features")
        st.markdown("3. Generate RFM-based business segments")
        st.markdown("4. Train a Random Forest classifier")
        st.markdown("5. Evaluate the classification model")
        st.markdown("6. Provide an interactive Streamlit interface")

    st.info(
        "Important limitation: because the Segment target is generated from RFM-based rules and the classifier uses related RFM features, very high classification scores should be interpreted carefully."
    )

    st.markdown("### 🚀 Future scope")
    st.write("Numerical future CLV forecasting • Churn prediction • Purchase-interval features • Model comparison • Time-based validation • Personalized recommendations • Advanced dashboards")

st.markdown('<div class="footer">Customer Lifetime Value Analysis • Built with Python & Streamlit • MD JAHID</div>', unsafe_allow_html=True)
