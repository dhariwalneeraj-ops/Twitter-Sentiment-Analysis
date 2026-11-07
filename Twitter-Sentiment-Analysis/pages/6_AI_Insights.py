"""
Page 6: Generative AI Sentiment Summaries & Insights
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.genai import generate_sentiment_summary, ask_sentiment_ai
from src.aspect_analysis import AspectSentimentAnalyzer
from src.prediction import SentimentPredictor

st.title("🤖 Generative AI Sentiment Summaries & Intelligence")
st.markdown("""
Transform thousands of unstructured customer tweets into structured, executive-ready intelligence 
using **Generative AI (Google Gemini)** and advanced NLP synthesis.
""")

# Load Dataset
dataset_path = root_dir / "dataset" / "twitter.csv"
if not dataset_path.exists():
    st.error("Twitter dataset file not found.")
    st.stop()

df = pd.read_csv(dataset_path)

# Prepare Aspect Data for richer GenAI Context
predictor = SentimentPredictor(models_dir=root_dir / "models")
aspect_analyzer = AspectSentimentAnalyzer(predictor=predictor)
aspect_summary = aspect_analyzer.analyze_dataset(df)["summary_df"]

tab_exec, tab_ask, tab_custom = st.tabs([
    "📋 Executive Sentiment Briefing",
    "💬 Ask AI (Natural Language Q&A)",
    "🎨 Custom GenAI Prompt Studio"
])

# Retrieve active API key from session state if configured
active_key = st.session_state.get("gemini_api_key", "")

# TAB 1: Executive Briefing
with tab_exec:
    st.subheader("📊 Automated Executive Sentiment Briefing")
    st.write("Synthesizes corpus sentiment metrics, aspect strengths, customer pain points, and strategic recommendations.")

    if active_key:
        st.success("🟢 Connected to Google Gemini Live Cloud Model")
    else:
        st.info("ℹ️ Running via Built-in Executive NLP Synthesizer. (To switch to live Gemini LLM, add your API key in the sidebar!)")

    if st.button("✨ Generate Executive Sentiment Summary", type="primary"):
        with st.spinner("Analyzing tweet corpus & generating executive briefing..."):
            summary_md = generate_sentiment_summary(
                df=df,
                api_key=active_key,
                aspect_df=aspect_summary
            )
        st.markdown("---")
        st.markdown(summary_md)


# TAB 2: Ask AI (Natural Language Q&A)
with tab_ask:
    st.subheader("💬 Ask AI About Customer Sentiment")
    st.write("Ask natural language questions about customer reactions, product features, or feedback trends.")

    query_preset = st.selectbox(
        "Suggested Queries:",
        options=[
            "What are the most frequent complaints regarding battery and charging?",
            "How do customers rate the display and 120Hz refresh rate?",
            "What issues are causing users to return the product or request refunds?",
            "What are customer impressions of delivery times and shipping conditions?",
            "Custom question..."
        ]
    )

    if query_preset == "Custom question...":
        user_query = st.text_input("Enter your question:", placeholder="e.g., Are users happy with customer service?")
    else:
        user_query = query_preset

    if st.button("🤖 Ask AI", key="ask_btn"):
        if user_query.strip():
            with st.spinner("Querying feedback corpus..."):
                response_text = ask_sentiment_ai(
                    query=user_query,
                    df=df,
                    api_key=active_key
                )
            st.markdown("---")
            st.markdown(response_text)
        else:
            st.warning("Please enter a question.")


# TAB 3: Custom GenAI Prompt Studio
with tab_custom:
    st.subheader("🎨 Custom Strategy & Prompt Studio")
    st.write("Customize the analysis persona and strategic angle for the generative summary.")

    role_persona = st.selectbox(
        "Select Analyst Persona:",
        options=[
            "Chief Product Officer (Focus on product roadmap & feature defects)",
            "Head of Customer Support (Focus on escalation resolution & response delays)",
            "VP of Marketing & Brand Strategy (Focus on public perception & brand loyalty)",
            "Quality Assurance Lead (Focus on software glitches, thermal throttling & hardware build)"
        ]
    )

    additional_instructions = st.text_area(
        "Additional Custom Instructions:",
        value=f"Tailor the executive report for the {role_persona}. Highlight 3 high-impact immediate fixes and provide a SWOT matrix of product perception.",
        height=100
    )

    if st.button("🚀 Generate Tailored Strategic Report", type="primary"):
        with st.spinner("Generating tailored strategic briefing..."):
            custom_summary = generate_sentiment_summary(
                df=df,
                api_key=active_key,
                aspect_df=aspect_summary,
                custom_instructions=additional_instructions
            )
        st.markdown("---")
        st.markdown(custom_summary)
