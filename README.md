# Sports Interview Transcript Classifier

Fine-tunes RoBERTa-base to classify sports interview transcripts into 8 categories: Game Strategy, Player Performance, Injury Updates, Post-Game Analysis, Team Morale, Upcoming Matches, Off-Game Matters and Controversies.

Results: On 5,000+ NFL/NBA/NHL transcripts, the model reaches 46% accuracy (macro-F1 0.26), against 33% for a logistic regression baseline. Class imbalance is the main limitation.

Streamlit app:

Classify a transcript into one of the 8 categories
Ask category-based questions (DistilBERT extractive Q&A over a short context per category)
Explore transcripts in an interactive t-SNE plot (TF-IDF features)

Training: Hugging Face Transformers and PyTorch, with early stopping and learning-rate scheduling. Topic clusters (20+) were found by analysing sentence embeddings with cosine similarity and t-SNE.

Topics: nlp, roberta, huggingface, transformers, pytorch, text-classification, streamlit, tsne, sports-analytics
