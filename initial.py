import spacy
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Load Spacy model for NER
nlp = spacy.load("en_core_web_sm")

# Sample Resumes and Job Description Data
resumes = [
    "John Doe has 5 years of experience in Data Science and Python. Skilled in machine learning, SQL, and AWS.",
    "Jane Smith, a certified AI engineer, specializes in NLP, transformers, and Python. Worked with Google Cloud Platform.",
    "Alice Johnson, a software developer with expertise in Java, Spring Boot, and React. No ML experience.",
]
job_description = """
We are looking for a skilled AI Engineer proficient in Natural Language Processing, 
transformers, Python, and cloud platforms such as AWS or GCP. 
Experience in machine learning frameworks is a plus.
"""

# Named Entity Recognition (NER) for Key Information Extraction
def extract_key_entities(text):
    doc = nlp(text)
    entities = {
        "Skills": [],
        "Experience": [],
        "Certifications": [],
    }
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON"]:
            continue
        if "experience" in ent.text.lower():
            entities["Experience"].append(ent.text)
        elif ent.label_ in ["GPE", "LANGUAGE", "PRODUCT", "NORP", "FAC"]:
            entities["Skills"].append(ent.text)
        elif "certified" in ent.text.lower():
            entities["Certifications"].append(ent.text)
    return entities

# Extract entities from resumes
print("Extracted Entities:")
for i, resume in enumerate(resumes):
    print(f"Resume {i+1}: {extract_key_entities(resume)}")

# TF-IDF Vectorization and Ranking
vectorizer = TfidfVectorizer()
corpus = resumes + [job_description]
tfidf_matrix = vectorizer.fit_transform(corpus)

# Compute cosine similarity
similarity_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
ranked_indices = np.argsort(-similarity_scores[0])  # Descending order

print("\nRanking Candidates:")
for i, idx in enumerate(ranked_indices):
    print(f"{i+1}. Resume {idx+1}: Similarity Score = {similarity_scores[0][idx]:.2f}")

# Advanced Embedding-based Matching using Transformers
# Load a pretrained transformer model for text classification
similarity_pipeline = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-12-v2")

print("\nRanking with Transformers:")
for i, resume in enumerate(resumes):
    input_text = f"Job Description: {job_description}\nResume: {resume}"
    result = similarity_pipeline(input_text)
    print(f"Resume {i+1}: Match Score = {result[0]['score']:.2f}")
