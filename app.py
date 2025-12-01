import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Doctor & Medical Helper", layout="wide")

QUOTES = [
    "Healing takes time, and asking for help is strength.",
    "Every step towards health matters.",
    "Your body can heal if your mind believes.",
    "You are stronger than this illness.",
    "Health is the real wealth."
]

# ---------------- UTILITIES ----------------
def show_motivation():
    st.info(random.choice(QUOTES))

def show_disclaimer():
    st.warning(
        "⚠️ Educational Use Only — This is NOT a real medical system.\n"
        "Always consult a certified doctor before taking any medicine."
    )

# ---------------- DATA LOADERS ----------------
@st.cache_data
def load_doctor_data():
    try:
        return pd.read_csv("doctor.csv")
    except:
        return pd.DataFrame()

@st.cache_data
def load_medicine_data():
    try:
        return pd.read_csv("medicine.csv")
    except:
        return pd.DataFrame()

# ---------------- SAFE ML FOR DOCTOR ----------------
@st.cache_resource
def train_doctor_model(df):
    if "is_available_today" not in df.columns:
        return None, [], None

    y = df["is_available_today"].astype(int)
    X = df.select_dtypes(include=["int64", "float64"])

    if y.nunique() < 2 or X.empty:
        return None, [], None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        return model, X.columns.tolist(), acc
    except:
        return None, [], None

def predict_doctor(model, features, row):
    try:
        x = row[features].to_frame().T.fillna(0)
        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0][1]
        return pred, prob
    except:
        return None

# ---------------- AI SYMPTOM MODEL ----------------
SYMPTOMS_DATA = pd.DataFrame([
    {
        "symptoms": "fever cold cough sore throat",
        "disease": "Viral Fever",
        "precautions": "Rest, warm fluids, avoid cold drinks",
        "medicine": "Paracetamol"
    },
    {
        "symptoms": "chest pain sweating left hand pain",
        "disease": "Heart Issue",
        "precautions": "Emergency medical help immediately",
        "medicine": "Doctor supervision required"
    },
    {
        "symptoms": "loose motion dehydration vomiting",
        "disease": "Food Poisoning",
        "precautions": "Drink ORS, avoid outside food",
        "medicine": "ORS, Zinc"
    },
    {
        "symptoms": "breathing problem wheezing",
        "disease": "Asthma",
        "precautions": "Avoid dust, smoke",
        "medicine": "Salbutamol Inhaler"
    }
])

@st.cache_resource
def train_symptom_model():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])
    pipe.fit(SYMPTOMS_DATA["symptoms"], SYMPTOMS_DATA["disease"])
    return pipe

def get_disease_info(name):
    row = SYMPTOMS_DATA[SYMPTOMS_DATA["disease"] == name]
    if row.empty:
        return None, None
    return row.iloc[0]["precautions"], row.iloc[0]["medicine"]

# ---------------- MODULE 1: SYMPTOM CHECK ----------------
def symptom_checker(model):
    st.header("🧠 AI Symptom Checker")
    show_motivation()
    show_disclaimer()

    text = st.text_area("Enter Symptoms")

    if st.button("Predict Disease"):
        disease = model.predict([text])[0]
        precautions, medicine = get_disease_info(disease)

        st.success(f"Predicted Disease: {disease}")
        st.write("✅ Precautions:", precautions)
        st.write("💊 Suggested Medicine:", medicine)

# ---------------- MODULE 2: HOSPITAL & DOCTOR ----------------
def hospital_module(df, model, features, acc):
    st.header("🏥 Hospital & Doctor Availability")
    show_motivation()

    locations = sorted(df["location"].unique())
    loc = st.selectbox("Select Hospital", locations)

    loc_df = df[df["location"] == loc]
    st.dataframe(loc_df[["name", "specialization", "rating", "is_available_today"]])

    st.success("🚑 Ambulance Available | Call 108")

    doc_name = st.selectbox("Select Doctor", loc_df["name"].unique())
    row = loc_df[loc_df["name"] == doc_name].iloc[0]

    current = "Available" if row["is_available_today"] == 1 else "Not Available"
    st.info(f"Current Status: {current}")

    if model:
        pred, prob = predict_doctor(model, features, row)
        label = "Available" if pred == 1 else "Not Available"
        st.success(f"ML Prediction: {label} ({prob*100:.1f}% confidence)")

# ---------------- MODULE 3: MEDICINE ----------------
def medicine_module(df):
    st.header("💊 Medicine Availability")
    show_motivation()

    name = st.text_input("Enter medicine name")

    if st.button("Check Medicine"):
        res = df[df["name"].str.lower().str.contains(name.lower())]

        if res.empty:
            st.error("Medicine not found")
        else:
            for _, r in res.iterrows():
                st.success(f"Name: {r['name']}")
                st.write("Use:", r["use0"])
                st.write("Substitute:", r["substitute0"])
                st.write("Side Effect:", r["sideEffect0"])
                st.write("Availability:", r["availability"])

# ---------------- MAIN APP ----------------
def main():
    st.sidebar.title("AI Medical Assistant")

    page = st.sidebar.radio("Menu", [
        "AI Symptom Checker",
        "Hospital & Doctor",
        "Medicine Availability",
        "About"
    ])

    doctor_df = load_doctor_data()
    medicine_df = load_medicine_data()

    doctor_model, features, acc = train_doctor_model(doctor_df)
    symptom_model = train_symptom_model()

    if page == "AI Symptom Checker":
        symptom_checker(symptom_model)

    elif page == "Hospital & Doctor":
        hospital_module(doctor_df, doctor_model, features, acc)

    elif page == "Medicine Availability":
        medicine_module(medicine_df)

    else:
        st.header("About This Project")
        st.write("AI Medical Assistant with ML-based Doctor & Medicine Prediction.")

if __name__ == "__main__":
    main()
