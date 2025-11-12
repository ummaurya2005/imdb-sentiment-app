# # Step 1: Import Libraries and Load the Model
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model
# import streamlit as st

# # Load IMDB dataset word index
# word_index = imdb.get_word_index()

# # Reverse mapping for decoding
# reverse_word_index = {value: key for key, value in word_index.items()}

# # Load your pre-trained model
# model = load_model('sentiment_model.h5')

# # Step 2: Helper Functions

# # Decode integer review back to text (optional for debugging)
# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# # Preprocess user text
# def preprocess_text(text):
#     # Split and lowercase
#     words = text.lower().split()

#     # Convert words to integers using word_index
#     # Unknown words get mapped to 2 (the <UNK> token)
#     encoded_review = [1]  # start token
#     for word in words:
#         if word in word_index and word_index[word] < 10000:
#             encoded_review.append(word_index[word] + 3)
#         else:
#             encoded_review.append(2)  # unknown word token

#     # Pad sequence to match training input length (500)
#     padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
#     return padded_review

# # Step 3: Streamlit UI
# st.title('🎬 IMDB Movie Review Sentiment Analysis')
# st.write('Enter a movie review below and see whether it’s Positive or Negative.')

# user_input = st.text_area('✍️ Movie Review')

# if st.button('Classify'):
#     preprocessed_input = preprocess_text(user_input)
#     prediction = model.predict(preprocessed_input)
#     sentiment = '😊 Positive' if prediction[0][0] > 0.5 else '😞 Negative'
#     st.write(f'**Sentiment:** {sentiment}')
#     st.write(f'**Confidence Score:** {prediction[0][0]:.4f}')
# else:
#     st.write('Please enter a movie review above.')


# Step 1: Import Libraries and Load the Model
# ============================================================
# 🎬 IMDB Movie Review Sentiment Analysis (Streamlit App)
# Author: Uttam Maurya
# ============================================================

# Step 1: Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re

# ============================================================
# Step 2: Load IMDB Word Index
# ============================================================
@st.cache_resource
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value + 3: key for key, value in word_index.items()}
    reverse_word_index[0] = "<PAD>"
    reverse_word_index[1] = "<START>"
    reverse_word_index[2] = "<UNK>"
    reverse_word_index[3] = "<UNUSED>"
    return word_index, reverse_word_index

word_index, reverse_word_index = load_word_index()

# ============================================================
# Step 3: Load the Trained Model (cached)
# ============================================================
@st.cache_resource
def load_sentiment_model():
    # Use the modern .keras format for TensorFlow >= 2.17
    return load_model("simple_rnn_imdb_fixed.keras")

model = load_sentiment_model()

# ============================================================
# Step 4: Helper Functions
# ============================================================
def decode_review(encoded_review):
    """Convert numeric review back to readable text (for debugging)."""
    return " ".join([reverse_word_index.get(i, "?") for i in encoded_review])

def preprocess_text(text, max_features=10000, max_len=200):
    """Convert raw text to same numeric sequence as training data."""
    # Clean & tokenize
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    words = text.split()

    # Encode words
    encoded_review = [1]  # start token
    for word in words:
        idx = word_index.get(word)
        if idx is not None and idx < max_features:
            encoded_review.append(idx + 3)  # +3 offset
        else:
            encoded_review.append(2)        # unknown token

    # Pad to same length as training input
    padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
    return padded_review

# ============================================================
# Step 5: Streamlit UI
# ============================================================
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below and find out whether it’s **Positive** or **Negative**!")

user_input = st.text_area("✍️ Write your movie review here:")

if st.button("🔍 Classify Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please type a review before clicking Classify.")
    else:
        preprocessed = preprocess_text(user_input)
        prediction = model.predict(preprocessed)
        prob = float(prediction[0][0])
        sentiment = "😊 Positive" if prob >= 0.5 else "😞 Negative"

        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence Score:** `{prob:.4f}`")

        # Optional: Debug decoded tokens
        # st.text(decode_review(preprocessed[0]))

else:
    st.info("💡 Type a review and click *Classify Sentiment* to get started.")

# ============================================================
# End of File
# ============================================================
