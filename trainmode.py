import pandas as pd
import numpy as np
import re
import string
import joblib

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# Download NLTK Data
# -------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv(r'C:\Users\hp\Desktop\Manikanta_ML_Engineer_Resume_5Y.csv')

print(df.head())
print(df.shape)

# -------------------------------
# Rename Columns (if required)
# -------------------------------

df.columns = ["Category", "Resume"]

print(df.head())

# -------------------------------
# Check Missing Values
# -------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# -------------------------------
# Remove Duplicate Rows
# -------------------------------

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates")
print(df.shape)

# -------------------------------
# Initialize NLP Objects
# -------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Text Cleaning Function
# -------------------------------

def clean_resume(text):

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"www\S+", " ", text)

    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"RT|cc", " ", text)

    text = re.sub(r"#\S+", " ", text)

    text = re.sub(r"@\S+", " ", text)

    text = re.sub(r"[0-9]+", " ", text)

    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)

    text = re.sub(r"\n", " ", text)

    text = re.sub(r"\r", " ", text)

    text = re.sub(r"\t", " ", text)

    text = re.sub(r"\s+", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------
# Apply Cleaning
# -------------------------------

print("\nCleaning Resume Text...")

df["Cleaned_Resume"] = df["Resume"].apply(clean_resume)

print(df[["Resume", "Cleaned_Resume"]].head())

# -------------------------------
# Label Encoding
# -------------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(df["Category"])

# -------------------------------
# TF-IDF Vectorization
# -------------------------------

tfidf = TfidfVectorizer(
    max_features=5000
)

X = tfidf.fit_transform(df["Cleaned_Resume"])

print("\nTF-IDF Shape")

print(X.shape)

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Samples :", X_train.shape[0])

print("Testing Samples :", X_test.shape[0])

# -------------------------------
# Train Logistic Regression
# -------------------------------

model = LogisticRegression(

    max_iter=1000,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

# -------------------------------
# Prediction
# -------------------------------

y_pred = model.predict(

    X_test

)

# -------------------------------
# Accuracy
# -------------------------------

accuracy = accuracy_score(

    y_test,

    y_pred

)

print("\nAccuracy")

print(round(accuracy * 100, 2), "%")

# -------------------------------
# Classification Report
# -------------------------------

print("\nClassification Report")

print(

    classification_report(

        y_test,

        y_pred,

        target_names=encoder.classes_

    )

)

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(

    y_test,

    y_pred

)

print("\nConfusion Matrix")

print(cm)

# -------------------------------
# Save Models
# -------------------------------

joblib.dump(

    model,

    "models/model.pkl"

)

joblib.dump(

    tfidf,

    "models/tfidf.pkl"

)

joblib.dump(

    encoder,

    "models/encoder.pkl"

)

print("\nModel Saved Successfully!")

print("model.pkl")

print("tfidf.pkl")

print("encoder.pkl")