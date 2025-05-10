import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset
def load_data():
    df = pd.read_csv("skin_disease.csv")
    df['symptoms'] = df['symptoms'].fillna("")
    df['symptoms_clean'] = df['symptoms'].apply(lambda x: ' '.join(x.lower().replace(';', ' ').split()))
    return df

df = load_data()

st.set_page_config(page_title="Skin Disease Detector", layout="centered")
st.title("AI-Based Skin Disease Classifier")
st.write("Enter your symptoms (e.g., 'pimples; itching; redness') and get possible disease prediction.")

st.sidebar.write("# Skin Disease Classifier")
st.sidebar.write("This app predicts skin diseases based on symptoms you provided.")
st.sidebar.write("### Instructions:")
st.sidebar.write("1. Enter your symptoms in the text area.")
st.sidebar.write("2. Click 'Predict Disease' to get predictions.")
st.sidebar.write("3. AI model can make mistakes, so please consult a doctor for accurate diagnosis.")
st.sidebar.write("### Disclaimer:")
st.sidebar.write("This app is for educational purposes only. Always consult a healthcare professional for medical advice.")
st.sidebar.write("### About:")
st.sidebar.write("This app uses machine learning to predict skin diseases based on symptoms. It is trained on a dataset of skin diseases and their symptoms.")    

# Prepare model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['symptoms_clean'])
y = df['disease']
model = LogisticRegression()
model.fit(X, y)

# Helper to preprocess user input
def preprocess_user_input(text):
    return ' '.join(text.lower().replace(';', ' ').split())

# Predict with optional exclusions
def predict_with_feedback(input_clean, exclude_diseases=[]):
    input_vec = vectorizer.transform([input_clean])
    pred_probs = model.predict_proba(input_vec)[0]
    disease_indices = pred_probs.argsort()[::-1]

    for idx in disease_indices:
        candidate = model.classes_[idx]
        if candidate not in exclude_diseases:
            return candidate
    return "No confident prediction"



user_input = st.text_area("📝 Enter your symptoms", "pimples; red skin")
if st.button("🔍 Predict Disease"):
    clean_input = preprocess_user_input(user_input)
    excluded = []
    max_attempts = len(df['disease'].unique())  # Limiting max attempts to number of unique diseases

    for attempt in range(max_attempts):
        prediction = predict_with_feedback(clean_input, exclude_diseases=excluded)

        if prediction == "No confident prediction":
            st.error("❌ Sorry, no confident prediction found.")
            break

        
        feedback = "Yes"

        if feedback == "Yes":
            st.success(f"✅ Our Prediction **{prediction}**")
            # Optionally show more info
            row = df[df['disease'] == prediction].iloc[0]
            st.markdown("### 🧾 Disease Details")
            st.markdown(f"**Symptoms:** {row['symptoms']}")
            st.markdown(f"**Medicines:** {row['medicines']}")
            st.markdown(f"**Precautions:** {row['precautions']}")
            if 'image_paths' in row and pd.notna(row['image_paths']):
                st.markdown("### 🖼️ Reference Images:")
                image_paths = str(row['image_paths']).split(';')
                columns = st.columns(len(image_paths))  # Create a column for each image

                # Loop over image paths and display images in respective columns
                for i, path in enumerate(image_paths):
                    columns[i].image(path.strip(), width=150)

            
            break
        else:
            excluded.append(prediction)  # Add the rejected prediction to the excluded list
            st.warning("🔄 Retrying with another possible disease...")
