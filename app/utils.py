import torch
import numpy as np
import pandas as pd
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForQuestionAnswering, pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# ----------------------
# Category Definitions
# ----------------------
category_map = {
    0: "Game Strategy",
    1: "Player Performance",
    2: "Injury Updates",
    3: "Post-Game Analysis",
    4: "Team Morale",
    5: "Upcoming Matches",
    6: "Off-Game Matters",
    7: "Controversies"
}

category_contexts = {
    "Game Strategy": "Our game strategy involved a high pressing line and rapid counters...",
    "Player Performance": "The striker was key today with two assists and consistent pressure...",
    "Injury Updates": "Our midfielder suffered a slight hamstring pull last week...",
    "Post-Game Analysis": "We controlled possession for 70% of the match and had 15 shots...",
    "Team Morale": "The locker room remains united despite recent challenges...",
    "Upcoming Matches": "Next week we face our biggest rivals, and training has intensified...",
    "Off-Game Matters": "Outside of matches, players have participated in community programs...",
    "Controversies": "There was a dispute regarding the referee’s decisions this match..."
}

# ----------------------
# Load RoBERTa Model
# ----------------------
@torch.no_grad()
def load_classifier():
    model = RobertaForSequenceClassification.from_pretrained("roberta_model")
    tokenizer = RobertaTokenizer.from_pretrained("roberta_model")
    return model.eval(), tokenizer

# ----------------------
# Classify Transcript
# ----------------------
@torch.no_grad()
def classify_transcript(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return category_map[pred]

# ----------------------
# Load QA Pipeline
# ----------------------
def load_qa_pipeline():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

# ----------------------
# Generate QA Answer
# ----------------------
def generate_answer(qa_pipeline, category, question):
    context = category_contexts.get(category, "")
    if not context:
        return "No context found for this category."
    result = qa_pipeline(question=question, context=context)
    return result.get("answer", "Sorry, I couldn't find an answer.")

# ----------------------
# t-SNE + TF-IDF Visualization
# ----------------------
def generate_tsne_data(texts, categories):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()

    n_samples = len(texts)
    if n_samples <= 3:
        raise ValueError("Need at least 4 samples for t-SNE visualization.")

    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(X)

    df = pd.DataFrame(tsne_results, columns=["x", "y"])
    df["category"] = categories
    df["text"] = texts
    return df