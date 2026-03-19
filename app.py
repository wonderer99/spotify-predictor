import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

@st.cache_resource
def train_model():
    # hardcoded small dataset - no CSV needed!
    data = {
        "loudness": [-6.7,-17.2,-9.7,-18.5,-9.6,-7.8,-11.2,-8.9,-13.4,-6.1],
        "explicit": [0,0,0,0,1,0,1,0,0,1],
        "danceability": [0.676,0.420,0.438,0.266,0.618,0.720,0.532,0.689,0.445,0.756],
        "energy": [0.461,0.166,0.359,0.060,0.443,0.812,0.534,0.623,0.289,0.734],
        "tempo": [87.9,77.4,76.3,181.7,119.9,128.0,95.3,110.2,140.5,100.8],
        "acousticness": [0.032,0.924,0.210,0.905,0.469,0.045,0.234,0.123,0.678,0.056],
        "popularity": [80,45,60,30,75,88,55,70,40,85]
    }
    df = pd.DataFrame(data)
    X = df[["loudness","explicit","danceability","energy","tempo","acousticness"]]
    y = df["popularity"]
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

st.title("🎵 Spotify Song Popularity Predictor")
st.write("Enter song features to predict popularity!")

loudness = st.slider("Loudness", -60, 0, -10)
explicit = st.selectbox("Explicit?", [0, 1])
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo (BPM)", 50, 250, 120)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)

if st.button("Predict Popularity 🎵"):
    features = np.array([[loudness, explicit, danceability, energy, tempo, acousticness]])
    prediction = model.predict(features)
    st.success(f"Predicted Popularity: {prediction[0]:.0f} / 100")
