import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer DNA Lab | MD JAHID",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at top right, #eef2ff 0, #f8fafc 42%, #eef6f5 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1220,#172033); }
    [data-testid="stSidebar"] * { color:#f8fafc !important; }
    .hero { padding:2.2rem 2.4rem; border-radius:28px; background:linear-gradient(135deg,#111827,#312e81,#0f766e); color:white; box-shadow:0 18px 45px rgba(15,23,42,.18); margin-bottom:1.3rem; }
    .hero h1 { font-size:3rem; margin:.35rem 0; }
    .hero p { color:#dbeafe; font-size:1.08rem; margin:0; }
    .badge { display:inline-block; padding:.35rem .8rem; border-radius:999px; background:#ccfbf1; color:#115e59; font-weight:800; font-size:.78rem; letter-spacing:.04em; }
    .card { background:rgba(255,255,255,.9); border:1px solid #e2e8f0; border-radius:20px; padding:1.15rem 1.25rem; margin:.35rem 0 1rem; box-shadow:0 8px 24px rgba(15,23,42,.06); }
    .dna { border-radius:24px; padding:1.6rem; background:linear-gradient(135deg,#ecfeff,#eef2ff); border:1px solid #a5f3fc; text-align:center; }
    .dna-title { font-size:.85rem; color:#475569; letter-spacing:.12em; font-weight:800; }
    .dna-value { font-size:2.15rem; font-weight:900; color:#312e81; margin:.2rem 0; }
    .result { border-radius:22px; padding:1.7rem; text-align:center; background:linear-gradient(135deg,#f0fdfa,#eef2ff); border:1px solid #99f6e4; }
    .result-label { color:#64748b; font-size:.9rem; }
    .result-value { font-size:2.2rem; font-weight:900; color:#0f766e; margin:.2rem 0; }
    .timeline { text-align:center; padding:1rem; border-radius:18px; background:white; border:1px solid #e2e8f0; }
    .footer { text-align:center; color:#64748b; padding:2rem 0 .5rem; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_assets():
    model = joblib.load("rf_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, encoder, features

try:
    model, encoder, features = load_assets()
except Exception as exc:
    st.error("Model files could not be loaded. Make sure rf_model.pkl, label_encoder.pkl and feature_columns.pkl are in the repository.")
    st.caption(str(exc))
    st.stop()

# The trained project uses Recency, Frequency and Monetary as the prediction inputs.
expected = ["Recency", "Frequency", "Monetary"]
if list(features) != expected:
    st.warning(f"Loaded feature order: {list(features)}")

segments = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost Customers"]
segment_info = {
    "Champions": ("🏆", "Highly engaged historical behavior", "Reward, retain and personalize engagement."),
    "Loyal Customers": ("❤️", "Repeated purchasing behavior", "Strengthen loyalty and explore cross-sell opportunities."),
    "Potential Loyalists": ("🌱", "Promising but developing behavior", "Nurture engagement and encourage repeat purchases."),
    "At Risk": ("⚠️", "Reduced recent activity signal", "Consider re-engagement and retention experiments."),
    "Lost Customers": ("🔄", "Substantial inactivity signal", "Evaluate cost-effective win-back opportunities."),
}

st.sidebar.markdown("# 🧬 Customer DNA Lab")
st.sidebar.caption("Decode customer behaviour with RFM + ML")
st.sidebar.divider()
page = st.sidebar.radio("LAB MODULES", [
    "🧬 DNA Lab",
    "🧪 Mutation Simulator",
    "🌌 Customer Universe",
    "⏳ Time Machine",
    "🧠 Model X-Ray",
    "🎯 Action Center",
    "📚 Project Info",
])
st.sidebar.divider()
st.sidebar.success("🟢 Analytics Engine Online")
st.sidebar.caption("Random Forest • RFM • 5 Segments")
st.sidebar.markdown("**Developer**  ")
st.sidebar.markdown("MD JAHID")


def predict(recency, frequency, monetary):
    x = pd.DataFrame([[recency, frequency, monetary]], columns=features)
    encoded = model.predict(x)
    return encoder.inverse_transform(encoded)[0], x


def dna_scores(r, f, m):
    # Visual indices only; the actual ML prediction remains the saved model prediction.
    r_score = 100 * (1 - min(max(r, 0), 500) / 500)
    f_score = 100 * min(max(f, 0), 50) / 50
    m_score = 100 * min(max(m, 0), 10000) / 10000
    return r_score, f_score, m_score


def dna_visual(r, f, m):
    rs, fs, ms = dna_scores(r, f, m)
    points = []
    for i in range(18):
        angle = i * 2 * np.pi / 18
        radius = 0.45 + 0.42 * ((rs * np.sin(angle) ** 2 + fs * np.cos(angle) ** 2 + ms) / 300)
        points.append((radius * np.cos(angle), radius * np.sin(angle)))
    path = " ".join(f"{x*100+50:.1f},{y*100+50:.1f}" for x, y in points)
    return f'''<div class="dna"><div class="dna-title">CUSTOMER DNA</div><svg viewBox="0 0 100 100" width="100%" height="220"><defs><linearGradient id="g" x1="0" x2="1"><stop offset="0"/><stop offset="1"/></linearGradient></defs><polygon points="{path}" fill="none" stroke="currentColor" stroke-width="1.8"/><circle cx="50" cy="50" r="2.5" fill="currentColor"/><text x="50" y="10" text-anchor="middle" font-size="5">R</text><text x="91" y="53" text-anchor="middle" font-size="5">F</text><text x="50" y="95" text-anchor="middle" font-size="5">M</text></svg><div class="dna-value">{rs:.0f} · {fs:.0f} · {ms:.0f}</div><div>Recency Index · Frequency Index · Monetary Index</div></div>'''

# ---------------- DNA LAB ----------------
if page == "🧬 DNA Lab":
    st.markdown('<div class="hero"><span class="badge">CUSTOMER INTELLIGENCE LAB</span><h1>🧬 Customer DNA Lab</h1><p>Decode a customer's historical behaviour from Recency, Frequency and Monetary signals.</p></div>', unsafe_allow_html=True)
    st.subheader("Create a customer profile")
    presets = {
        "Custom DNA": (30, 10, 1000.0),
        "Power Buyer": (7, 35, 4500.0),
        "Premium Buyer": (12, 22, 8500.0),
        "Occasional Buyer": (120, 5, 600.0),
        "Inactive Buyer": (365, 2, 250.0),
    }
    preset = st.selectbox("🧩 Start with a DNA template", list(presets))
    d_r, d_f, d_m = presets[preset]
    with st.form("dna_form"):
        a, b, c = st.columns(3)
        r = a.number_input("Recency (days)", 0, 5000, d_r, 1)
        f = b.number_input("Frequency (orders)", 0, 10000, d_f, 1)
        m = c.number_input("Monetary (total spend ₹)", 0.0, 10000000.0, float(d_m), 100.0)
        go = st.form_submit_button("🧬 Decode Customer DNA", type="primary", use_container_width=True)
    if go:
        segment, x = predict(r, f, m)
        icon, meaning, action = segment_info.get(segment, ("🔎", "Model-derived segment", "Review the RFM profile."))
        left, right = st.columns([1.1, .9])
        with left:
            st.markdown(dna_visual(r, f, m), unsafe_allow_html=True)
        with right:
            st.markdown(f'<div class="result"><div class="result-label">DNA CLASSIFICATION</div><div class="result-value">{icon} {segment}</div><div>{meaning}</div></div>', unsafe_allow_html=True)
            st.metric("Recency", f"{r} days")
            st.metric("Frequency", f"{f} orders")
            st.metric("Monetary", f"₹{m:,.0f}")
        st.markdown("### 🔍 Behaviour reading")
        st.info(f"**Model segment:** {segment}. **Possible business focus:** {action}")
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x)[0]
            classes = encoder.inverse_transform(model.classes_)
            pdf = pd.DataFrame({"Segment": classes, "Probability": probs}).sort_values("Probability", ascending=False)
            st.markdown("### 📡 Model probability signal")
            st.bar_chart(pdf.set_index("Segment"))

# ---------------- MUTATION ----------------
elif page == "🧪 Mutation Simulator":
    st.markdown('<div class="hero"><span class="badge">WHAT-IF ENGINE</span><h1>🧪 Customer Mutation Simulator</h1><p>Change one behaviour signal and observe how the trained classifier responds.</p></div>', unsafe_allow_html=True)
    st.info("This is a scenario simulator. It does not predict the customer's actual future behaviour or future monetary CLV.")
    c1, c2, c3 = st.columns(3)
    with c1:
        r0 = st.number_input("Current Recency", 0, 5000, 30)
        r1 = st.number_input("Mutated Recency", 0, 5000, 80)
    with c2:
        f0 = st.number_input("Current Frequency", 0, 10000, 10)
        f1 = st.number_input("Mutated Frequency", 0, 10000, 20)
    with c3:
        m0 = st.number_input("Current Monetary ₹", 0.0, 10000000.0, 1000.0)
        m1 = st.number_input("Mutated Monetary ₹", 0.0, 10000000.0, 2500.0)
    before, _ = predict(r0, f0, m0)
    after, _ = predict(r1, f1, m1)
    st.markdown("### Mutation result")
    a, b, c = st.columns([1, .3, 1])
    with a:
        st.markdown(f'<div class="result"><div class="result-label">BEFORE</div><div class="result-value">{segment_info.get(before, ("🔎",))[0]} {before}</div><p>R={r0} · F={f0} · M=₹{m0:,.0f}</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown("<div style='text-align:center;font-size:2.2rem;padding-top:2rem;'>→</div>", unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="result"><div class="result-label">AFTER MUTATION</div><div class="result-value">{segment_info.get(after, ("🔎",))[0]} {after}</div><p>R={r1} · F={f1} · M=₹{m1:,.0f}</p></div>', unsafe_allow_html=True)
    if before != after:
        st.success(f"The scenario changed the model classification from **{before}** to **{after}**.")
    else:
        st.info(f"The model classification remains **{after}** for this scenario.")

# ---------------- UNIVERSE ----------------
elif page == "🌌 Customer Universe":
    st.markdown('<div class="hero"><span class="badge">BEHAVIOUR MAP</span><h1>🌌 Customer Universe</h1><p>Explore a simulated customer population and inspect how RFM profiles map to model segments.</p></div>', unsafe_allow_html=True)
    st.caption("The universe is a visual demonstration generated from RFM ranges; it is not a replacement for the original transaction dataset.")
    rng = np.random.default_rng(42)
    n = st.slider("Number of demo customers", 25, 250, 100, 25)
    demo = pd.DataFrame({
        "Recency": rng.integers(1, 400, n),
        "Frequency": rng.integers(1, 45, n),
        "Monetary": rng.uniform(100, 9000, n).round(0),
    })
    demo["Segment"] = [predict(row.Recency, row.Frequency, row.Monetary)[0] for row in demo.itertuples()]
    demo["Customer"] = [f"C-{i:03d}" for i in range(1, n + 1)]
    st.scatter_chart(demo, x="Recency", y="Monetary", color="Segment", size="Frequency")
    selected = st.selectbox("🔭 Inspect a customer", demo["Customer"])
    row = demo.loc[demo.Customer == selected].iloc[0]
    st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)
    st.info("Universe coordinates are visual aids. Segment labels come from the saved Random Forest model.")

# ---------------- TIME MACHINE ----------------
elif page == "⏳ Time Machine":
    st.markdown('<div class="hero"><span class="badge">PAST → PRESENT → SCENARIO</span><h1>⏳ Customer Time Machine</h1><p>Compare three RFM states and see how the classifier labels each scenario.</p></div>', unsafe_allow_html=True)
    st.warning("The third stage is a user-created scenario, not a forecast of actual future customer behaviour.")
    stages = ["Past", "Present", "Scenario"]
    defaults = [(180, 4, 500), (45, 12, 1800), (15, 25, 5000)]
    cols = st.columns(3)
    states = []
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### {stages[i]}")
            rr = st.number_input(f"R · {stages[i]}", 0, 5000, defaults[i][0], key=f"tr{i}")
            ff = st.number_input(f"F · {stages[i]}", 0, 10000, defaults[i][1], key=f"tf{i}")
            mm = st.number_input(f"M · {stages[i]}", 0.0, 10000000.0, float(defaults[i][2]), 100.0, key=f"tm{i}")
            ss, _ = predict(rr, ff, mm)
            states.append((rr, ff, mm, ss))
            st.markdown(f"**{segment_info.get(ss, ('🔎',))[0]} {ss}**")
    st.markdown("### Journey")
    tcols = st.columns(5)
    for i, (label, state) in enumerate(zip(stages, states)):
        with tcols[min(i*2, 4)]:
            st.markdown(f'<div class="timeline"><b>{label}</b><br>{state[3]}</div>', unsafe_allow_html=True)
        if i < 2:
            with tcols[min(i*2+1, 4)]: st.markdown("<div style='text-align:center;padding:1.5rem;font-size:1.5rem;'>→</div>", unsafe_allow_html=True)

# ---------------- X-RAY ----------------
elif page == "🧠 Model X-Ray":
    st.markdown('<div class="hero"><span class="badge">MODEL TRANSPARENCY</span><h1>🧠 Model X-Ray</h1><p>See exactly how the project connects customer RFM inputs to the saved classifier output.</p></div>', unsafe_allow_html=True)
    x1, x2, x3 = st.columns(3)
    x1.metric("Algorithm", "Random Forest")
    x2.metric("Inputs", "3")
    x3.metric("Output classes", "5")
    st.markdown("### Prediction pipeline")
    st.code("Transaction history\n       ↓\nData cleaning & preprocessing\n       ↓\nCustomer-level RFM features\n       ↓\nRFM-based Segment label\n       ↓\nRandom Forest Classifier\n       ↓\nPredicted Customer Segment", language="text")
    st.dataframe(pd.DataFrame({"Input feature": features, "Role": ["Recent activity", "Purchase activity", "Historical spending"]}), use_container_width=True, hide_index=True)
    st.warning("The Segment target is generated from RFM-based business rules, while the classifier uses related RFM features. Therefore, very high classification scores should be interpreted carefully.")

# ---------------- ACTION CENTER ----------------
elif page == "🎯 Action Center":
    st.markdown('<div class="hero"><span class="badge">SEGMENT PLAYBOOK</span><h1>🎯 Action Center</h1><p>Translate segment definitions into practical business-analysis ideas.</p></div>', unsafe_allow_html=True)
    for name in segments:
        icon, meaning, action = segment_info[name]
        with st.expander(f"{icon} {name}"):
            st.write(f"**Behaviour signal:** {meaning}")
            st.write(f"**Possible business response:** {action}")
    st.markdown("### Decision lens")
    st.dataframe(pd.DataFrame({"Segment": segments, "Focus": [segment_info[x][2] for x in segments]}), use_container_width=True, hide_index=True)

# ---------------- INFO ----------------
else:
    st.markdown('<div class="hero"><span class="badge">ACADEMIC PROJECT</span><h1>📚 Project Information</h1><p>Customer Lifetime Value Analysis · Jagan Nath University, Bahadurgarh (NCR)</p></div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Methodology", "Technology", "Scope & Future"])
    with t1:
        st.markdown("1. Clean retail transaction data")
        st.markdown("2. Create customer-level Recency, Frequency and Monetary features")
        st.markdown("3. Generate RFM-based business segments")
        st.markdown("4. Train a Random Forest classifier")
        st.markdown("5. Evaluate classification performance")
        st.markdown("6. Deploy an interactive Streamlit interface")
    with t2:
        st.write("Python • Pandas • NumPy • Scikit-learn • Joblib • Streamlit")
        st.write("Developer: **MD JAHID**")
        st.write("Institution: **Jagan Nath University, Bahadurgarh (NCR)**")
    with t3:
        st.info("Current scope: RFM-based customer-value segmentation. The system does not directly forecast a future monetary CLV amount.")
        st.write("Future scope: numerical CLV forecasting, churn prediction, purchase-interval features, model comparison, time-based validation, personalized recommendations and advanced dashboards.")

st.markdown('<div class="footer">🧬 Customer DNA Lab · Customer Lifetime Value Analysis · Built with Streamlit · MD JAHID</div>', unsafe_allow_html=True)
