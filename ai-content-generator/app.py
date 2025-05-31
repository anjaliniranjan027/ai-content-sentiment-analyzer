import streamlit as st
from transformers import pipeline, set_seed
import joblib
import pandas as pd
import base64
import time

# Load sentiment classifier
classifier = joblib.load("text_classifier.pkl")

# Page setup
st.set_page_config(page_title="✨ AI Generator Pro", layout="wide")
st.markdown("<h1 style='text-align:center;'>🧠 AI Content Generator + Sentiment Analyzer</h1>", unsafe_allow_html=True)

# Sidebar Config
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("🤖 Choose Hugging Face Model", ["gpt2", "distilgpt2"])
    max_len = st.slider("📏 Max Generation Length", 50, 300, 100)
    num_return = st.selectbox("📦 Number of Outputs", [1, 2, 3])
    show_probs = st.checkbox("🔍 Show Sentiment Probabilities", value=False)

# Load generator
@st.cache_resource
def load_generator(model_name):
    return pipeline("text-generation", model=model_name)

generator = load_generator(model_choice)
set_seed(42)

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Input
prompt = st.text_input("✍️ Enter your creative prompt:")

# Result animation placeholder
result_placeholder = st.empty()

if st.button("🚀 Generate & Analyze"):
    with st.spinner("Generating amazing content..."):
        time.sleep(1)  # dramatic loading
        
        # UPDATED generation with parameters to reduce repetition
        outputs = generator(
            prompt,
            max_length=max_len,
            num_return_sequences=num_return,
            no_repeat_ngram_size=2,  # prevent repeated 2-word sequences
            temperature=0.8,         # balance creativity & coherence
            top_k=50,
            top_p=0.9
        )

        result_data = []

        # Step-by-step reveal
        for i, output in enumerate(outputs, start=1):
            text = output["generated_text"]
            pred = classifier.predict([text])[0]
            sentiment = "Positive" if pred == 1 else "Negative"

            if show_probs:
                proba = classifier.predict_proba([text])[0]
                pos = round(proba[1] * 100, 2)
                neg = round(proba[0] * 100, 2)
            else:
                pos = neg = None

            result = {
                "Prompt": prompt,
                "Generated Text": text,
                "Sentiment": sentiment,
                "Positive %": pos,
                "Negative %": neg
            }

            st.session_state.history.append(result)
            result_data.append(result)

            with st.expander(f"🔹 Result {i} — {sentiment}"):
                st.markdown(f"**📝 Generated Text:**\n\n{text}")
                if show_probs:
                    st.write(f"✅ Positive: {pos}%")
                    st.write(f"❌ Negative: {neg}%")

            time.sleep(0.7)  # pacing animation

        # Show DataFrame
        df = pd.DataFrame(result_data)
        st.markdown("### 📊 Sentiment Overview")
        st.dataframe(df)

        st.bar_chart(df["Sentiment"].value_counts())

        # Download
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="ai_results.csv">📥 Download Results as CSV</a>',
            unsafe_allow_html=True
        )

# Session History Expander
with st.expander("🕓 Show Full Session History"):
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)
    else:
        st.info("No past prompts yet. Try generating something!")
