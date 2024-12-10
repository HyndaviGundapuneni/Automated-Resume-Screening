#Step 1: Download the Dataset
#First, download the dataset from Kaggle:

#Dataset Link: Updated Resume Dataset on Kaggle​
#KAGGLE
.
#Save the dataset locally, then upload it to your working environment (e.g., Colab or Jupyter Notebook).

#Step 2: Project Setup
#Install necessary libraries:


#Step 3: Load and Explore the Dataset

import pandas as pd

# Load the dataset
file_path = "UpdatedResumeDataset.csv"  
df = pd.read_csv(file_path)

# Display basic information
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())
#Step 4: Preprocessing the Data
#Text Cleaning: Remove unnecessary characters, stopwords, and apply tokenization.
#Label Encoding: Encode the resume categories for classification.

import re
from sklearn.preprocessing import LabelEncoder

# Clean text data
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

df['cleaned_resume'] = df['Resume'].apply(clean_text)

# Encode categories
le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

print(le.classes_)  # Check encoded categories
#Step 5: Named Entity Recognition (NER)
#Use the spaCy library for NER to extract key entities like skills, experience, and education.

#bash
#!pip install spacy
#!python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load("en_core_web_sm")

# Extract entities from resumes
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

df['Entities'] = df['cleaned_resume'].apply(extract_entities)
print(df[['cleaned_resume', 'Entities']].head())
#Step 6: Text Classification
#Train a text classification model using TensorFlow and Transformers.


from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_resume'], df['Category_Encoded'], test_size=0.2, random_state=42
)

# Use TF-IDF for feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# Build the model
model = Sequential([
    Dense(512, activation='relu', input_dim=5000),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_tfidf, y_train, validation_data=(X_test_tfidf, y_test), epochs=10, batch_size=32)
#Step 7: Evaluate the Model

loss, accuracy = model.evaluate(X_test_tfidf, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
