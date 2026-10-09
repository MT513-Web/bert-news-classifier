import streamlit as st
import time
import pandas as pd

# 1. Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="News Analytics Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling for Premium SaaS / Dark Tech Aesthetic ---
st.markdown("""
<style>
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.9rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Primary Action Button (Modern Blue Gradient) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55) !important;
    }

    /* Clean Sidebar Badges & Cards */
    .spec-item {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
    }
    .spec-title {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    .spec-val {
        font-size: 0.88rem;
        color: #f1f5f9;
        font-weight: 500;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Model Settings
MODEL_PATH = "./news_classifier_bert_v1"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_ICONS = {
    "World": "🌍",
    "Sports": "⚽",
    "Business": "💼",
    "Sci/Tech": "🔬"
}

# --- Sidebar (Technical Specs Only) ---
with st.sidebar:
    st.markdown('<div class="status-badge">● Production Ready</div>', unsafe_allow_html=True)
    st.write("")
    st.title("System Specs")
    
    st.markdown("""
    <div class="spec-item">
        <div class="spec-title">Architecture</div>
        <div class="spec-val">BERT-Base (Uncased)</div>
    </div>
    <div class="spec-item">
        <div class="spec-title">Benchmark Accuracy</div>
        <div class="spec-val">93.8% Evaluation F1</div>
    </div>
    <div class="spec-item">
        <div class="spec-title">Inference Engine</div>
        <div class="spec-val">PyTorch • CPU Optimized</div>
    </div>
    <div class="spec-item">
        <div class="spec-title">Token Limit</div>
        <div class="spec-val">128 Subword Tokens</div>
    </div>
    <div class="spec-item">
        <div class="spec-title">Embeddings</div>
        <div class="spec-val">768-Dim Contextual Vectors</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("TARGET CATEGORIES")
    for label in LABEL_NAMES:
        st.markdown(f"**{LABEL_ICONS[label]} {label}**")

# --- Main Dashboard ---
st.title("📰 News Headline Classifier")
st.caption("Classify breaking news headlines into relevant domains using Transformer contextual embeddings.")

# Manage input text via session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Quick Sample Test Buttons
st.markdown("##### Quick Test Samples:")
sample_cols = st.columns(4)

samples = {
    "Tech Sample": "NVIDIA reveals next-generation AI chip architecture for enterprise supercomputing clusters.",
    "Sports Sample": "Real Madrid secures dramatic Champions League victory in stoppage time thriller.",
    "Business Sample": "Federal Reserve holds interest rates steady as global markets show cautious optimism.",
    "World Sample": "United Nations climate summit negotiators reach landmark accord on global carbon reductions."
}

def set_sample(text):
    st.session_state.input_text = text

with sample_cols[0]:
    if st.button("🔬 Tech", use_container_width=True):
        set_sample(samples["Tech Sample"])
with sample_cols[1]:
    if st.button("⚽ Sports", use_container_width=True):
        set_sample(samples["Sports Sample"])
with sample_cols[2]:
    if st.button("💼 Business", use_container_width=True):
        set_sample(samples["Business Sample"])
with sample_cols[3]:
    if st.button("🌍 World", use_container_width=True):
        set_sample(samples["World Sample"])

# Input Field
user_input = st.text_area(
    "Headline Text:",
    value=st.session_state.input_text,
    height=100,
    placeholder="Type or paste a full news headline here..."
)

# Action Buttons: Classify & Clear
btn_col1, btn_col2 = st.columns([4, 1])
with btn_col1:
    analyze_clicked = st.button("Classify Headline", type="primary", use_container_width=True)
with btn_col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.input_text = ""
        st.rerun()

# 2. Cached Resource Loader
@st.cache_resource(show_spinner=False)
def load_model_assets():
    from transformers import AutoTokenizer, BertForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

# --- Inference & Results Section ---
if analyze_clicked:
    clean_text = user_input.strip()
    
    if not clean_text:
        st.warning("Please enter a valid headline or pick a sample button above.")
    else:
        with st.spinner("Classifying headline..."):
            import torch
            import torch.nn.functional as F
            
            tokenizer, model = load_model_assets()
            start_time = time.time()
            
            inputs = tokenizer(
                clean_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                pred_index = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][pred_index].item() * 100
            
            latency = (time.time() - start_time) * 1000
            predicted_label = LABEL_NAMES[pred_index]
            icon = LABEL_ICONS.get(predicted_label, "")

        st.divider()

        # Results Display
        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.markdown("##### Prediction Summary")
            with st.container(border=True):
                st.metric(
                    label="Predicted Category",
                    value=f"{icon} {predicted_label}"
                )
                confidence_tag = "🟢 High Confidence" if confidence >= 85 else "🟡 Moderate Confidence"
                st.write(f"**Confidence Level:** `{confidence:.2f}%` • *{confidence_tag}*")
                st.progress(confidence / 100.0)
                st.caption(f"⚡ Latency: **{latency:.1f}ms** (PyTorch Engine)")

        with col_res2:
            st.markdown("##### Probability Breakdown")
            with st.container(border=True):
                prob_dict = {
                    f"{LABEL_ICONS[lbl]} {lbl}": round(p * 100, 2)
                    for lbl, p in zip(LABEL_NAMES, probabilities[0].tolist())
                }
                chart_df = pd.DataFrame(
                    list(prob_dict.items()),
                    columns=["Category", "Probability (%)"]
                ).set_index("Category")
                
                # Sorted so highest probability bar is on top
                chart_df = chart_df.sort_values(by="Probability (%)", ascending=True)
                st.bar_chart(chart_df, horizontal=True)

st.divider()
st.caption("AI/ML Engineering Portfolio • Real-Time Text Classification System")