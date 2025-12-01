import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Doctor & Medical Helper", layout="wide")

QUOTES = [
    "Healing takes time, and asking for help is strength.",
    "Your health is your real wealth.",
    "Every small step toward recovery matters.",
    "You are stronger than your illness.",
    "Rest is also a form of treatment."
]

# ---------------- UTILITIES ----------------
def show_motivation():
    st.info(random.choice(QUOTES))

def show_disclaimer():
    st.warning("⚠️ Educational only. This is NOT real medical advice.")

# ---------------- DATA LOAD ----------------
@st.cache_data
def load_doctor_data():
    try:
        return pd.read_csv("doctor.csv")
    except:
        return pd.DataFrame()

@st.cache_data
def load_medicine_data():
    try:
        df = pd.read_csv("medicine.csv")
        if "availability" not in df.columns:
            df["availability"] = "Available"
        return df
    except:
        return pd.DataFrame()

# ---------------- 20+ DISEASE RULE BASE ----------------
DISEASE_DB = [
    ("fever cold cough", "Viral Fever", "Rest, warm fluids", "Paracetamol", "After food, every 6 hours"),
    ("body pain joint pain", "Arthritis", "Warm compress", "Ibuprofen", "After food"),
    ("chest pain sweating", "Heart Issue", "Emergency required", "Aspirin (Doctor only)", "Immediate"),
    ("loose motion diarrhea", "Food Poisoning", "ORS and hydration", "ORS + Zinc", "After each loose stool"),
    ("breath wheezing", "Asthma", "Avoid dust", "Salbutamol", "As needed"),
    ("sugar thirst urination", "Diabetes", "Low sugar diet", "Metformin", "After food"),
    ("acid burning food", "Acidity", "Avoid spicy food", "Omeprazole", "Before breakfast"),
    ("itch sneezing allergy", "Allergy", "Avoid allergen", "Cetirizine", "Night"),
    ("headache vomiting", "Migraine", "Dark room rest", "Paracetamol", "When pain starts"),
    ("ear pain discharge", "Ear Infection", "Keep ear dry", "Antibiotics", "Doctor decides"),
    ("tooth pain swelling", "Tooth Infection", "Dental check", "Pain killer", "After food"),
    ("period missed nausea", "Pregnancy", "Gynecologist consult", "Folic Acid", "Once daily"),
    ("rash fever travel", "Chikungunya", "Mosquito protection", "Paracetamol", "After food"),
    ("weight loss cough night sweat", "Tuberculosis", "Isolation", "Anti-TB drugs", "Strict daily"),
    ("vision blur headache", "Blood Pressure", "Reduce salt", "Amlodipine", "Once daily"),
    ("vomiting child fever", "Pediatric Infection", "Hydration", "Syrup", "As prescribed"),
    ("urine burning", "UTI", "Drink water", "Antibiotic", "After food"),
    ("leg swelling breathless", "Heart Failure", "Emergency visit", "Diuretics", "Doctor decides"),
    ("skin patch itch", "Fungal Infection", "Keep dry", "Antifungal cream", "Twice daily"),
    ("neck stiffness fever", "Meningitis", "Emergency", "Hospital IV drugs", "Immediate")
]

def predict_disease(symptoms):
    symptoms = symptoms.lower()
    for key, disease, precaution, medicine, timing in DISEASE_DB:
        if any(word in symptoms for word in key.split()):
            return disease, precaution, medicine, timing
    return "Unknown Condition", "Consult doctor immediately", "Doctor will decide", "As prescribed"

# ---------------- MODULE 1: SYMPTOM CHECK ----------------
def symptom_module():
    st.header("🧠 AI Symptom Checker")
    show_motivation()
    show_disclaimer()

    name = st.text_input("Patient Name")
    age = st.number_input("Age", 0, 120)
    symptoms = st.text_area("Enter your symptoms")

    if st.button("Predict Disease"):
        disease, precaution, medicine, timing = predict_disease(symptoms)

        st.success(f"Disease: {disease}")
        st.write("✅ Precaution:", precaution)
        st.write("💊 Medicine:", medicine)
        st.write("⏰ Timing:", timing)

# ---------------- MODULE 2: HOSPITAL ----------------
def hospital_module(df):
    st.header("🏥 Hospital & Doctor")
    show_motivation()

    locations = df["location"].unique()
    loc = st.selectbox("Select Hospital", locations)

    loc_df = df[df["location"] == loc]
    st.dataframe(loc_df[["name", "specialization", "fee", "is_available_today"]])

    st.success("🚑 Ambulance Available: Call 108")
    st.info("🛏 ICU Beds: 10 | Oxygen Beds: 20 | General Beds: 50")

# ---------------- MODULE 3: MEDICINE ----------------
def medicine_module(df):
    st.header("💊 Medicine Availability")
    show_motivation()

    name = st.text_input("Search Medicine")

    if st.button("Search"):
        res = df[df["name"].str.lower().str.contains(name.lower())]

        if res.empty:
            st.error("Medicine not found")
        else:
            for _, r in res.iterrows():
                st.success(r["name"])
                st.write("Use:", r["use0"])
                st.write("Substitute:", r["substitute0"])
                st.write("Side Effect:", r["sideEffect0"])
                st.write("Availability:", r["availability"])

# ---------------- MODULE 4: PAYMENT ----------------
def payment_module(df):
    st.header("💳 Payment & Appointment")
    show_motivation()

    doctor = st.selectbox("Select Doctor", df["name"].unique())
    row = df[df["name"] == doctor].iloc[0]

    fee = row["fee"]
    st.write(f"Doctor Fee: ₹{fee}")

    payment = st.radio("Payment Method", ["UPI", "Card", "NetBanking", "Cash"])
    if st.button("Pay Now"):
        st.success(f"Payment Successful via {payment}")

    date = st.date_input("Appointment Date")
    if st.button("Book Appointment"):
        st.success(f"Appointment booked with {doctor} on {date}")

# ---------------- MAIN APP ----------------
def main():
    st.sidebar.title("AI Medical Assistant")
    page = st.sidebar.radio("Menu", [
        "Symptom Checker",
        "Hospital & Doctor",
        "Medicine",
        "Payment & Appointment",
        "About"
    ])

    doctor_df = load_doctor_data()
    medicine_df = load_medicine_data()

    if page == "Symptom Checker":
        symptom_module()
    elif page == "Hospital & Doctor":
        hospital_module(doctor_df)
    elif page == "Medicine":
        medicine_module(medicine_df)
    elif page == "Payment & Appointment":
        payment_module(doctor_df)
    else:
        st.header("About This App")
        st.write("Full AI Medical Assistant for real-world hospital & patient guidance.")

if __name__ == "__main__":
    main()
