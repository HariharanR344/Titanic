import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.write("Predict whether a passenger survived using a trained ML model")

# -----------------------------
# Load Pickle Files
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("titanic_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler (1).pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, scaler, le

model, scaler, le = load_artifacts()

st.success("✅ Model loaded successfully")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("🧍 Passenger Details")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 25)
sex = st.selectbox("Sex", ["male", "female"])
fare = st.number_input("Fare", min_value=0.0, value=32.0)
family_size = st.slider("Family Size", 1, 10, 1)

# Encode Sex
sex_encoded = le.transform([sex])[0]

# Create input DataFrame
input_data = pd.DataFrame(
    [[pclass, age, sex_encoded, fare, family_size]],
    columns=['Pclass', 'Age', 'Sex_encoded', 'Fare', 'FamilySize']
)

# Scale input
input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Survival"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("🎉 Passenger is likely to **SURVIVE**")
    else:
        st.error("💀 Passenger is **NOT likely to survive**")
