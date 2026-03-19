import streamlit as st
import numpy as np
import pickle

# load the saved model
model = pickle.load(open("spotify_model.pkl", "rb"))

st.title("🎵 Spotify Song Popularity Predictor")
st.write("Enter song features to predict popularity!")

# input sliders
loudness = st.slider("Loudness", -60, 0, -10)
explicit = st.selectbox("Explicit?", [0, 1])
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 50, 250, 120)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)

# predict button
if st.button("Predict Popularity 🎵"):
    features = np.array([[loudness, explicit, danceability, energy, tempo, acousticness]])
    prediction = model.predict(features)
    st.success(f"Predicted Popularity: {prediction[0]:.0f} / 100")