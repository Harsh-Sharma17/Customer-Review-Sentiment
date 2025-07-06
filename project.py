import pandas as pd
import nltk
import re
import os
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === Setup NLTK for Streamlit Cloud ===
nltk_data_dir = "/tmp/nltk_data"
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

# === Preprocessing Tools ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# === Load Data ===
df = pd.read_csv("customer_reviews_100.csv")  # Ensure this file exists in your GitHub repo

df['cleaned_review'] = df['review'].apply(preprocess)

# === Feature Extraction ===
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    stop_words='english',
    sublinear_tf=True
)
X = tfidf.fit_transform(df['cleaned_review'])
y = df['sentiment']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# === Prediction Function ===
def predict_sentiment(review):
    review = preprocess(review)
    vec = tfidf.transform([review])
    prediction = model.predict(vec)
    return "Positive 😊" if prediction[0] == 1 else "Negative 😞"

# === Streamlit App ===
st.set_page_config(page_title="Customer Review Sentiment Classifier", page_icon="📝")
st.title("📝 Customer Review Sentiment Classifier")

st.write(f"**Model Accuracy:** {accuracy:.2f}")

user_input = st.text_area("Enter your review:")
if st.button("Predict Sentiment"):
    if user_input.strip():
        result = predict_sentiment(user_input)
        st.success(f"Sentiment: {result}")
    else:
        st.warning("Please enter a review to predict.")

with st.expander("See Model Evaluation Details"):
    st.json(report)
