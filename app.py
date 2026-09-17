import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Lifetime Value Analysis | MD JAHID", page_icon="📊", layout="wide")

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#fff7ed 0%,#fef3c7 45%,#f0fdfa 100%);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#172554 0%,#164e63 52%,#134e4a 100%);}
[data-testid="stSidebar"] * {color:#f8fafc !important;}
.hero {padding:2.6rem;border-radius:30px;background:linear-gradient(120deg,#172554 0%,#7c2d12 48%,#0f766e 100%);color:white;margin-bottom:1.3rem;box-shadow:0 18px 45px rgba(23,37,84,.22);border:1px solid rgba(255,255,255,.18);}
.hero h1 {font-size:3rem;margin:.35rem 0;letter-spacing:-1px;}
.hero p {color:#fef3c7;font-size:1.08rem;}
.badge {display:inline-block;padding:.42rem .95rem;border-radius:999px;background:linear-gradient(90deg,#fbbf24,#5eead4);color:#172554;font-weight:900;}
.card {padding:1.2rem;border-radius:22px;background:rgba(255,255,255,.9);border:1px solid #fed7aa;box-shadow:0 10px 28px rgba(120,53,15,.09);}
.dna {padding:1.5rem;border-radius:25px;background:linear-gradient(135deg,#fff7ed,#f0fdfa);border:1px solid #99f6e4;text-align:center;box-shadow:0 8px 24px rgba(15,118,110,.10);}
.dna-icon {font-size:4rem;}
.result {padding:1.6rem;border-radius:24px;background:linear-gradient(135deg,#fffbeb,#ecfdf5);border:1px solid #fcd34d;text-align:center;box-shadow:0 8px 24px rgba(180,83,9,.10);}
.result-value {font-size:2rem;font-weight:900;color:#9a3412;}
.footer {text-align:center;color:#64748b;padding:2rem 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    return joblib.load("rf_model.pkl"), joblib.load("label_encoder.pkl"), joblib.load("feature_columns.pkl")

try:
    model, encoder, features = load_assets()
except Exception as exc:
    st.error("Model files could not be loaded.")
    st.code(str(exc))
    st.stop()

expected = ["Recency", "Frequency", "Monetary"]
if list(features) != expected:
    st.warning(f"Loaded feature order: {list(features)}")

INFO = {
    "Champions": ("🏆", "Highly engaged historical behaviour.", "Reward, retain and personalize engagement."),
    "Loyal Customers": ("❤️", "Repeated purchasing behaviour.", "Strengthen loyalty and explore cross-sell opportunities."),
    "Potential Loyalists": ("🌱", "Promising but developing behaviour.", "Nurture engagement and encourage repeat purchases."),
    "At Risk": ("⚠️", "Reduced recent activity signal.", "Consider re-engagement and retention experiments."),
    "Lost Customers": ("🔄", "Substantial inactivity signal.", "Evaluate cost-effective win-back opportunities."),
}

def predict(r, f, m):
    x = pd.DataFrame([[r, f, m]], columns=features)
    encoded = model.predict(x)
    return encoder.inverse_transform(encoded)[0], x

def dna_scores(r, f, m):
    return max(0, 100 - min(r, 500) / 5), min(100, f / 50 * 100), min(100, m / 10000 * 100)

def dna_chart(r, f, m):
    rs, fs, ms = dna_scores(r, f, m)
    angles = np.linspace(0, 2 * np.pi, 4)[:-1]
    values = np.array([rs, fs, ms]) / 100
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(np.r_[angles, angles[0]], np.r_[values, values[0]], marker="o")
    ax.fill(np.r_[angles, angles[0]], np.r_[values, values[0]], alpha=.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(["Recency", "Frequency", "Monetary"])
    ax.set_ylim(0, 1)
    return fig

st.sidebar.markdown("# 📊 CLV ANALYSIS")
st.sidebar.caption("Customer value intelligence")
page = st.sidebar.radio("PROJECT MODULES", [
    "📊 CLV Analysis", "🧪 Scenario Simulator", "🌌 Customer Universe",
    "⏳ Time Machine", "🧠 Model X-Ray", "🎯 Action Center", "📚 Project Info"
])
st.sidebar.divider()
st.sidebar.success("🟢 Analytics Engine Online")
st.sidebar.caption("Random Forest • RFM • 5 Segments")
st.sidebar.markdown("**Developer:** MD JAHID")

if page == "📊 CLV Analysis":
    st.markdown("""
    <div class="hero">
      <span class="badge">CUSTOMER VALUE INTELLIGENCE</span>
      <h1>📊 Customer Lifetime Value Analysis</h1>
      <p>Analyze customer historical value using Recency, Frequency and Monetary signals.</p>
    </div>
    """, unsafe_allow_html=True)
    presets = {"Custom Profile": (30, 10, 1000.0), "Power Buyer": (7, 35, 4500.0), "Premium Buyer": (12, 22, 8500.0), "Occasional Buyer": (120, 5, 600.0), "Inactive Buyer": (365, 2, 250.0)}
    preset = st.selectbox("🧩 Choose a demonstration customer profile", list(presets))
    r0, f0, m0 = presets[preset]
    a, b, c = st.columns(3)
    r = a.number_input("Recency — days", 0, 5000, r0)
    f = b.number_input("Frequency — orders", 0, 10000, f0)
    m = c.number_input("Monetary — ₹", 0.0, 10000000.0, m0, 100.0)
    if st.button("📊 Analyze Customer Value", type="primary", use_container_width=True):
        segment, x = predict(r, f, m)
        icon, meaning, action = INFO.get(segment, ("🔎", "Model-derived segment.", "Review the RFM profile."))
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f'<div class="dna"><div class="dna-icon">📊</div><h2>{icon} {segment}</h2><p>R = {r} days &nbsp; | &nbsp; F = {f} orders &nbsp; | &nbsp; M = ₹{m:,.0f}</p></div>', unsafe_allow_html=True)
            st.pyplot(dna_chart(r, f, m), use_container_width=True)
        with right:
            st.markdown(f'<div class="result"><small>CUSTOMER SEGMENT</small><div class="result-value">{icon} {segment}</div><p>{meaning}</p></div>', unsafe_allow_html=True)
            st.info(f"**Business focus:** {action}")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                classes = encoder.inverse_transform(model.classes_)
                chart = pd.DataFrame({"Segment": classes, "Probability": probs}).set_index("Segment")
                st.markdown("### 📡 Model probability signal")
                st.bar_chart(chart)

elif page == "🧪 Scenario Simulator":
    st.markdown("""
    <div class="hero"><span class="badge">WHAT-IF ANALYSIS</span>
    <h1>🧪 Customer Scenario Simulator</h1>
    <p>Change customer behaviour signals and compare the model classification before and after.</p></div>
    """, unsafe_allow_html=True)
    st.warning("This is a scenario simulation, not a forecast of actual future behaviour or future monetary CLV.")
    a, b, c = st.columns(3)
    with a:
        r0 = st.number_input("Current Recency", 0, 5000, 30)
        r1 = st.number_input("Scenario Recency", 0, 5000, 120)
    with b:
        f0 = st.number_input("Current Frequency", 0, 10000, 10)
        f1 = st.number_input("Scenario Frequency", 0, 10000, 20)
    with c:
        m0 = st.number_input("Current Monetary ₹", 0.0, 10000000.0, 1000.0)
        m1 = st.number_input("Scenario Monetary ₹", 0.0, 10000000.0, 2500.0)
    before, _ = predict(r0, f0, m0)
    after, _ = predict(r1, f1, m1)
    x, y = st.columns(2)
    x.metric("BEFORE", before)
    y.metric("AFTER SCENARIO", after)
    st.markdown(f"**RFM change:** R {r0} → {r1}, F {f0} → {f1}, M ₹{m0:,.0f} → ₹{m1:,.0f}")
    if before != after:
        st.success(f"The simulated RFM change moved the classifier from **{before}** to **{after}**.")
    else:
        st.info(f"The classifier remained **{after}** in this scenario.")

elif page == "🌌 Customer Universe":
    st.markdown("""
    <div class="hero"><span class="badge">BEHAVIOUR MAP</span>
    <h1>🌌 Customer Universe</h1>
    <p>Explore a synthetic population of customer value profiles.</p></div>
    """, unsafe_allow_html=True)
    st.caption("These profiles are generated for demonstration and are not the original transaction dataset.")
    n = st.slider("Demo customers", 25, 200, 75, 25)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"Customer": [f"C-{i:03d}" for i in range(1, n + 1)], "Recency": rng.integers(1, 400, n), "Frequency": rng.integers(1, 45, n), "Monetary": rng.uniform(100, 9000, n).round(0)})
    df["Segment"] = [predict(a, b, c)[0] for a, b, c in zip(df.Recency, df.Frequency, df.Monetary)]
    st.scatter_chart(df, x="Recency", y="Monetary", color="Segment", size="Frequency")
    selected = st.selectbox("🔭 Inspect a customer", df.Customer)
    st.dataframe(df[df.Customer == selected], use_container_width=True, hide_index=True)

elif page == "⏳ Time Machine":
    st.markdown("""
    <div class="hero"><span class="badge">PAST → PRESENT → SCENARIO</span>
    <h1>⏳ Customer Time Machine</h1>
    <p>Compare three RFM states and observe the classifier's response.</p></div>
    """, unsafe_allow_html=True)
    st.warning("Scenario is hypothetical; it is not a prediction of actual future CLV.")
    defaults = [(180, 4, 500), (45, 12, 1800), (15, 25, 5000)]
    labels = ["🕰️ Past", "📍 Present", "🔮 Scenario"]
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### {labels[i]}")
            r = st.number_input("Recency", 0, 5000, defaults[i][0], key=f"r{i}")
            f = st.number_input("Frequency", 0, 10000, defaults[i][1], key=f"f{i}")
            m = st.number_input("Monetary ₹", 0.0, 10000000.0, float(defaults[i][2]), 100.0, key=f"m{i}")
            s, _ = predict(r, f, m)
            st.metric("Segment", s)

elif page == "🧠 Model X-Ray":
    st.markdown("""
    <div class="hero"><span class="badge">MODEL TRANSPARENCY</span>
    <h1>🧠 Model X-Ray</h1>
    <p>Understand exactly what the trained model receives and produces.</p></div>
    """, unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Algorithm", "Random Forest")
    b.metric("Inputs", "3 RFM")
    c.metric("Output", "Segment")
    st.markdown("### 🔬 Prediction pipeline")
    st.code("Recency + Frequency + Monetary\n          ↓\n   Random Forest Classifier\n          ↓\n     Segment label")
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).set_index("Feature")
        st.bar_chart(imp)
    st.info("The segment target is derived from RFM-based business rules. Therefore, very high classification performance should be interpreted carefully.")

elif page == "🎯 Action Center":
    st.markdown("""
    <div class="hero"><span class="badge">SEGMENT PLAYBOOK</span>
    <h1>🎯 Action Center</h1>
    <p>Translate analytical segments into possible business responses.</p></div>
    """, unsafe_allow_html=True)
    segment = st.selectbox("Select segment", list(INFO))
    icon, meaning, action = INFO[segment]
    st.markdown(f"## {icon} {segment}")
    st.markdown(f'<div class="card"><h3>Behaviour</h3><p>{meaning}</p><h3>Possible business focus</h3><p>{action}</p></div>', unsafe_allow_html=True)
    st.caption("These are analytical suggestions based on historical RFM segmentation, not guaranteed outcomes.")

else:
    st.markdown("""
    <div class="hero"><span class="badge">ACADEMIC PROJECT</span>
    <h1>📚 Customer Lifetime Value Analysis</h1>
    <p>RFM-based customer segmentation and classification using historical retail behaviour.</p></div>
    """, unsafe_allow_html=True)
    st.markdown("### Technology")
    st.write("Python • Pandas • NumPy • Scikit-learn • Joblib • Streamlit • Matplotlib")
    st.write("Model: Random Forest Classifier")
    st.write("Developer: MD JAHID")
    st.write("University: Jagan Nath University, Bahadurgarh (NCR)")
    st.markdown("### Scope")
    st.info("This application performs customer-value segmentation from historical RFM behaviour. It does not directly forecast a future monetary CLV amount.")
    st.markdown("### Future scope")
    st.write("Numerical CLV forecasting, churn prediction, richer behavioural features, model comparison, time-based validation and personalized recommendations.")

st.markdown('<div class="footer">📊 Customer Lifetime Value Analysis • RFM Customer Segmentation • MD JAHID</div>', unsafe_allow_html=True)
