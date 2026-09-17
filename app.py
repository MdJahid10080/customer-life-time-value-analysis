import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Lifetime Value Analysis", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(135deg, #f8f5ff, #eef2ff, #ecfeff);}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #1e1b4b, #312e81, #164e63);}
    [data-testid="stSidebar"] * {color: white !important;}
    .hero {padding: 2rem; border-radius: 24px; background: linear-gradient(120deg, #1e1b4b, #4c1d95, #0891b2); color: white; margin-bottom: 1.2rem;}
    .hero h1 {font-size: 2.6rem; margin: .3rem 0;}
    .hero p {color: #dbeafe; font-size: 1.05rem;}
    .card {padding: 1.2rem; border-radius: 18px; background: rgba(255,255,255,.92); border: 1px solid #c4b5fd; box-shadow: 0 8px 22px rgba(49,46,129,.08);}
    .sidebar-brand {display:flex; align-items:center; gap:12px; padding:8px 4px 14px 4px;}
    .brand-icon {width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,.14); font-size:21px;}
    .brand-title {font-size:1.25rem; font-weight:700;}
    .brand-subtitle {font-size:.72rem; opacity:.75; margin-top:2px;}
    .nav-title {font-size:.68rem; letter-spacing:1px; opacity:.7; margin:10px 0 7px 2px;}
    .side-card {padding:14px; border-radius:16px; background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.12);}
    .side-card-title {font-size:.68rem; letter-spacing:1px; opacity:.7; margin-bottom:10px;}
    .side-item {display:flex; align-items:center; gap:10px; padding:9px 6px; margin:4px 0; border-radius:10px; transition:background .2s ease, transform .2s ease;}
    .side-item:hover {background:rgba(255,255,255,.10); transform:translateX(3px);}
    .side-icon {width:30px; height:30px; border-radius:8px; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,.10);}
    .side-item b {display:block; font-size:.82rem;}
    .side-item small {display:block; font-size:.65rem; opacity:.65; margin-top:2px;}
    .developer-card {padding:14px; border-radius:16px; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12); text-align:center;}
    .dev-label {font-size:.62rem; letter-spacing:1.2px; opacity:.65;}
    .dev-name {font-size:1.05rem; font-weight:700; margin-top:4px;}
    .dev-role {font-size:.68rem; opacity:.65; margin-top:2px;}
    .footer {text-align:center; color:#64748b; padding:1.5rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    """Load the trained model and supporting files once."""
    model = joblib.load("rf_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, encoder, features

try:
    model, encoder, features = load_model()
except Exception:
    st.error(
        "The model files could not be loaded. Please make sure rf_model.pkl, "
        "label_encoder.pkl and feature_columns.pkl are present in the project."
    )
    st.stop()

EXPECTED_FEATURES = ["Recency", "Frequency", "Monetary"]
if list(features) != EXPECTED_FEATURES:
    st.error(f"The model expects these inputs: {list(features)}")
    st.stop()

SEGMENT_INFO = {
    "Champions": {"icon": "🏆", "description": "Customers with strong recent and repeat purchasing behaviour.", "action": "Focus on retention, appreciation and relevant offers."},
    "Loyal Customers": {"icon": "❤️", "description": "Customers who show consistent purchasing activity.", "action": "Build loyalty and introduce useful cross-sell opportunities."},
    "Potential Loyalists": {"icon": "🌱", "description": "Customers showing promising behaviour but with room to grow.", "action": "Encourage another purchase and strengthen engagement."},
    "At Risk": {"icon": "⚠️", "description": "Customers whose recent activity suggests reduced engagement.", "action": "Consider a simple re-engagement or retention campaign."},
    "Lost Customers": {"icon": "🔄", "description": "Customers with a strong inactivity signal in their historical data.", "action": "Consider whether a cost-effective win-back approach makes sense."},
}

def predict_segment(recency, frequency, monetary):
    """Predict a customer segment from the three RFM values."""
    customer = pd.DataFrame([[recency, frequency, monetary]], columns=EXPECTED_FEATURES)
    prediction = model.predict(customer)
    segment = encoder.inverse_transform(prediction)[0]
    return segment, customer

# Sidebar
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="brand-icon">📊</div>
        <div>
            <div class="brand-title">CLV Analysis</div>
            <div class="brand-subtitle">Customer Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Understand customer behaviour using RFM analysis.")

st.sidebar.markdown('<div class="nav-title">EXPLORE</div>', unsafe_allow_html=True)

# Box-style navigation using radio buttons styled as a compact menu.
page = st.sidebar.radio(
    "Navigation",
    [
        "🔎  Customer Analysis",
        "🧪  What-if Simulator",
        "🧠  Model Insights",
        "🎯  Segment Guide",
        "📚  About Project",
    ],
    label_visibility="collapsed",
)

page = page.split("  ", 1)[1]

st.sidebar.divider()

# Model card
st.sidebar.markdown(
    """
    <div class="side-card">
        <div class="side-card-title">⚙️ MODEL SETUP</div>
        <div class="side-item">
            <span class="side-icon">🌲</span>
            <div><b>Random Forest</b><small>Classification model</small></div>
        </div>
        <div class="side-item">
            <span class="side-icon">📈</span>
            <div><b>RFM Analysis</b><small>Recency • Frequency • Monetary</small></div>
        </div>
        <div class="side-item">
            <span class="side-icon">🎯</span>
            <div><b>5 Segments</b><small>Customer behaviour groups</small></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.divider()

# Developer card
st.sidebar.markdown(
    """
    <div class="developer-card">
        <div class="dev-label">PROJECT DEVELOPER</div>
        <div class="dev-name">MD JAHID</div>
        <div class="dev-role">Data Science Project</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Customer Analysis":
    st.markdown(
        """
        <div class="hero">
            <h1>📊 Customer Lifetime Value Analysis</h1>
            <p>Enter a customer's RFM details and see how the trained model classifies their historical value profile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    presets = {
        "Try your own values": (30, 10, 1000.0),
        "Frequent recent buyer": (7, 35, 4500.0),
        "High-spending buyer": (12, 22, 8500.0),
        "Occasional buyer": (120, 5, 600.0),
        "Inactive buyer": (365, 2, 250.0),
    }

    profile = st.selectbox("Choose a sample profile", list(presets))
    default_recency, default_frequency, default_monetary = presets[profile]

    col1, col2, col3 = st.columns(3)
    recency = col1.number_input("Recency (days)", 0, 5000, default_recency)
    frequency = col2.number_input("Frequency (orders)", 0, 10000, default_frequency)
    monetary = col3.number_input("Monetary value (₹)", 0.0, 10000000.0, default_monetary, 100.0)

    if st.button("Analyze Customer", type="primary", use_container_width=True):
        segment, customer = predict_segment(recency, frequency, monetary)
        info = SEGMENT_INFO.get(segment, {"icon": "🔎", "description": "The model assigned this historical customer segment.", "action": "Review the customer's RFM values before taking action."})

        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f"""
                <div class="card">
                    <h2>{info['icon']} {segment}</h2>
                    <p>{info['description']}</p>
                    <p><b>Recency:</b> {recency} days</p>
                    <p><b>Frequency:</b> {frequency} orders</p>
                    <p><b>Monetary:</b> ₹{monetary:,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.subheader("What this means")
            st.info(info["description"])
            st.success(f"Possible business focus: {info['action']}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(customer)[0]
                classes = encoder.inverse_transform(model.classes_)
                probability_data = pd.DataFrame({"Segment": classes, "Probability": probabilities}).set_index("Segment")
                st.subheader("Model confidence")
                st.bar_chart(probability_data)

    st.caption("This application classifies historical RFM behaviour. It does not directly forecast a future monetary CLV amount.")

elif page == "What-if Simulator":
    st.markdown(
        """
        <div class="hero">
            <h1>🧪 What-if Simulator</h1>
            <p>Change the RFM values and see how the model responds to the new customer profile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning("This is a hypothetical scenario, not a forecast of future behaviour or future CLV.")
    current = st.columns(3)
    scenario = st.columns(3)
    with current[0]: current_r = st.number_input("Current recency", 0, 5000, 30)
    with current[1]: current_f = st.number_input("Current frequency", 0, 10000, 10)
    with current[2]: current_m = st.number_input("Current monetary (₹)", 0.0, 10000000.0, 1000.0, 100.0)
    st.markdown("### Change the profile")
    with scenario[0]: new_r = st.number_input("New recency", 0, 5000, 120)
    with scenario[1]: new_f = st.number_input("New frequency", 0, 10000, 20)
    with scenario[2]: new_m = st.number_input("New monetary (₹)", 0.0, 10000000.0, 2500.0, 100.0)
    before, _ = predict_segment(current_r, current_f, current_m)
    after, _ = predict_segment(new_r, new_f, new_m)
    result1, result2 = st.columns(2)
    result1.metric("Current segment", before)
    result2.metric("Scenario segment", after)
    st.write(f"**RFM change:** Recency {current_r} → {new_r} days | Frequency {current_f} → {new_f} orders | Monetary ₹{current_m:,.0f} → ₹{new_m:,.0f}")
    if before == after:
        st.info(f"The model keeps the customer in the **{after}** segment for this scenario.")
    else:
        st.success(f"With these changed RFM values, the model classifies the profile as **{after}** instead of **{before}**.")

elif page == "Model Insights":
    st.markdown("""<div class="hero"><h1>🧠 Model Insights</h1><p>A simple view of what goes into the model and how the prediction is produced.</p></div>""", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Algorithm", "Random Forest")
    b.metric("Inputs", "3 RFM features")
    c.metric("Output", "Customer segment")
    st.subheader("Prediction flow")
    st.code("Recency + Frequency + Monetary\n          ↓\n   Random Forest\n          ↓\n     Segment label")
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).set_index("Feature")
        st.subheader("Feature importance")
        st.bar_chart(importance)
    st.info("The segment label is created from RFM-based business rules. Because the model learns from related RFM inputs, very high classification scores should be interpreted carefully.")

elif page == "Segment Guide":
    st.markdown("""<div class="hero"><h1>🎯 Segment Guide</h1><p>Use this page to understand the meaning of each customer segment.</p></div>""", unsafe_allow_html=True)
    segment = st.selectbox("Select a segment", list(SEGMENT_INFO))
    info = SEGMENT_INFO[segment]
    st.markdown(f"""<div class="card"><h2>{info['icon']} {segment}</h2><h4>Customer behaviour</h4><p>{info['description']}</p><h4>Possible business focus</h4><p>{info['action']}</p></div>""", unsafe_allow_html=True)
    st.caption("These suggestions are based on historical RFM segmentation and should be used as analytical guidance.")

else:
    st.markdown("""<div class="hero"><h1>📚 About the Project</h1><p>A practical data-science project for understanding customer purchasing behaviour.</p></div>""", unsafe_allow_html=True)
    st.markdown("### Project overview")
    st.write("Customer Lifetime Value Analysis uses historical retail transactions to create customer-level RFM measures: Recency, Frequency and Monetary value. These measures are then used to classify customers into meaningful segments.")
    st.markdown("### Technology used")
    st.write("Python • Pandas • Scikit-learn • Joblib • Streamlit • Matplotlib")
    st.markdown("### Project scope")
    st.write("The application focuses on customer-value segmentation from historical behaviour. It does not directly predict a future monetary CLV amount.")
    st.markdown("### Future improvements")
    st.write("Possible extensions include numerical CLV forecasting, churn prediction, richer behavioural features, model comparison and time-based validation.")
    st.markdown("### Developer")
    st.write("MD JAHID")

st.markdown('<div class="footer">Customer Lifetime Value Analysis • RFM Customer Segmentation • MD JAHID</div>', unsafe_allow_html=True)
