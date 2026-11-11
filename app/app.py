import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_classifier, classify_transcript,
    load_qa_pipeline, generate_answer,
    generate_tsne_data
)

st.set_page_config(page_title="Sports Transcript Analyzer", layout="wide")

st.title("⚽ Sports Transcript Analyzer")

tab1, tab2, tab3 = st.tabs(["📄 Transcript Classification", "❓ Question Answering", "📊 t-SNE Visualization"])

# --------------------------
# Tab 1: Transcript Classification
# --------------------------
with tab1:
    st.header("Classify Interview Transcript Text")

    user_input = st.text_area("Enter transcript text here")

    if st.button("Predict Category"):
        if user_input.strip():
            model, tokenizer = load_classifier()
            category = classify_transcript(model, tokenizer, user_input)
            st.success(f"Predicted Category: **{category}**")
        else:
            st.warning("Please enter some text to classify.")

# --------------------------
# Tab 2: Question Answering
# --------------------------
with tab2:
    st.header("Ask a Question about a Category")
    qa_pipeline = load_qa_pipeline()

    category = st.selectbox("Select Category", list(generate_answer.__annotations__.get('category', {}).keys()) or list([
        "Game Strategy", "Player Performance", "Injury Updates", "Post-Game Analysis", 
        "Team Morale", "Upcoming Matches", "Off-Game Matters", "Controversies"
    ]))
    question = st.text_input("Enter your question")

    if st.button("Get Answer") and question:
        answer = generate_answer(qa_pipeline, category, question)
        st.success(f"Answer: {answer}")

# --------------------------
# Tab 3: Interactive t-SNE Visualization
# --------------------------
with tab3:
    st.header("Explore t-SNE Visualization of Transcripts")

    transcript_texts = st.text_area("Enter multiple transcripts (one per line)", height=200)
    
    if st.button("Generate Visualization"):
        lines = [line.strip() for line in transcript_texts.strip().split('\n') if line.strip()]
        
        if len(lines) < 4:
            st.error("Please enter at least 4 transcript lines for meaningful visualization.")
        else:
            model, tokenizer = load_classifier()
            categories = [classify_transcript(model, tokenizer, line) for line in lines]
            try:
                df_tsne = generate_tsne_data(lines, categories)
                fig = px.scatter(
                    df_tsne, x="x", y="y", color="category",
                    hover_data={"text": True},
                    title="t-SNE Visualization of Transcript Embeddings"
                )
                st.plotly_chart(fig, use_container_width=True)
            except ValueError as ve:
                st.error(str(ve))