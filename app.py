import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import random

# -------------------------------------------------
# 1. BASIC CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Doctor & Hospital Assistant",
    page_icon="🩺",
    layout="wide"
)

# Motivational quotes
MOTIVATIONAL_QUOTES = [
    "Every day may not be good, but there’s something good in every day.",
    "Your body hears everything your mind says. Stay positive.",
    "Healing is not linear. Be kind to yourself today.",
    "Small steps every day lead to big changes.",
    "You’ve survived 100% of your worst days so far. Keep going.",
]

def show_motivation():
    st.markdown("### 💡 Motivational Thought for You")
    st.info(random.choice(MOTIVATIONAL_QUOTES))

# ⚠️ Medical disclaimer
def show_disclaimer():
    st.warning(
        "⚠️ This app is for **educational and informational purposes only**.\n\n"
        "It is **not** a substitute for professional medical advice, diagnosis, "
        "or treatment. Always consult a qualified doctor for health decisions. "
        "In case of emergency, contact your local emergency number immediately."
    )

# -------------------------------------------------
# 2. LOAD DATA
# -------------------------------------------------

@st.cache_data
def load_doctor_data():
    try:
        df = pd.read_csv("doctor.csv")
        return df
    except FileNotFoundError:
        st.error("`doctor.csv` not found. Please upload it to the project directory.")
        return pd.DataFrame()

@st.cache_data
def load_medicine_data():
    try:
        df = pd.read_csv("medicine.csv")
        # If no availability column exists, create a demo one
        if "availability" not in df.columns:
            # Demo only: randomly assign availability
            np.random.seed(42)
            df["availability"] = np.random.choice(
                ["Available", "Out of Stock"], size=len(df), p=[0.7, 0.3]
            )
        return df
    except FileNotFoundError:
        st.error("`medicine.csv` not found. Please upload it to the project directory.")
        return pd.DataFrame()

# -------------------------------------------------
# 3. DOCTOR AVAILABILITY MODEL
# -------------------------------------------------

@st.cache_resource
def train_doctor_model(df: pd.DataFrame):
    """
    Train a simple logistic regression model to predict doctor availability.
    Uses numeric workload-related features if they exist.
    """
    target_col = "is_available_today"
    if target_col not in df.columns:
        return None, [], None

    # Potential feature columns
    candidate_features = [
        "experience_years",
        "wait_time",
        "num_patients",
        "rating",
        "shift_duration_hours",
        "avg_patient_handling_capacity",
        "num_appointments_today",
        "peak_hours_flag",
        "previous_day_availability"
    ]
    feature_cols = [c for c in candidate_features if c in df.columns]

    if not feature_cols:
        return None, [], None

    # Prepare data
    df_model = df.dropna(subset=[target_col]).copy()
    if df_model[target_col].nunique() < 2:
        # Only one class → cannot train
        return None, feature_cols, None

    X = df_model[feature_cols].fillna(0)
    y = df_model[target_col]

    # Encode y if needed (e.g., "Yes"/"No")
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "available": 1, "not available": 0, "true": 1, "false": 0}
        ).fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, feature_cols, acc

def predict_doctor_availability(model, feature_cols, row: pd.Series):
    if model is None or not feature_cols:
        return None
    x = row[feature_cols].fillna(0).to_frame().T
    prob = model.predict_proba(x)[0][1] if hasattr(model, "predict_proba") else None
    pred = model.predict(x)[0]
    return pred, prob

# -------------------------------------------------
# 4. SYMPTOM → DISEASE MODEL (AI DOCTOR)
# -------------------------------------------------

# Small internal dataset (you can extend this with real medical datasets)
SYMPTOM_DATA = pd.DataFrame([
    {
        "symptoms": "fever cough sore throat runny nose sneezing body ache",
        "disease": "Common Cold / Viral Fever",
        "precautions": "Rest, drink plenty of warm fluids, salt-water gargles, avoid cold drinks, monitor temperature.",
        "medicines": "Paracetamol for fever and pain, throat lozenges. Consult a doctor if fever persists >3 days."
    },
    {
        "symptoms": "chest pain breathlessness sweating pain left arm",
        "disease": "Possible Heart-related Issue",
        "precautions": "Seek emergency medical help immediately. Do not exert yourself. Sit or lie comfortably.",
        "medicines": "Do not self-medicate. Aspirin may be used in emergency as advised by a doctor."
    },
    {
        "symptoms": "stomach pain loose motion diarrhea dehydration",
        "disease": "Gastroenteritis / Food Poisoning",
        "precautions": "Drink ORS, avoid outside food, maintain hygiene, eat light food, monitor urine output.",
        "medicines": "ORS, zinc supplements, probiotics. Avoid antibiotics without doctor’s prescription."
    },
    {
        "symptoms": "headache sensitivity to light nausea vomiting",
        "disease": "Migraine (Likely)",
        "precautions": "Rest in dark quiet room, avoid screen time, identify and avoid triggers (stress, certain foods).",
        "medicines": "Paracetamol or NSAIDs as advised. For frequent migraines, consult a neurologist."
    },
    {
        "symptoms": "frequent urination excessive thirst weight loss fatigue",
        "disease": "Possible Diabetes",
        "precautions": "Check blood sugar, avoid sugary foods/drinks, maintain balanced diet, regular exercise.",
        "medicines": "Oral anti-diabetic medicines or insulin may be needed; must be prescribed by doctor."
    },
    {
        "symptoms": "shortness of breath wheezing chest tightness cough",
        "disease": "Asthma (Likely)",
        "precautions": "Avoid dust, smoke, strong smells; use mask; keep inhalers handy as prescribed.",
        "medicines": "Reliever and controller inhalers as per doctor’s advice only."
    }
])

@st.cache_resource
def train_symptom_model(df: pd.DataFrame):
    X = df["symptoms"]
    y = df["disease"]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])
    pipe.fit(X, y)
    return pipe

def get_disease_info(disease_name: str):
    row = SYMPTOM_DATA[SYMPTOM_DATA["disease"] == disease_name]
    if row.empty:
        return None, None
    row = row.iloc[0]
    return row["precautions"], row["medicines"]

# -------------------------------------------------
# 5. HOSPITAL / AMBULANCE / PHARMACY INFO
# -------------------------------------------------

def get_location_info(location_name: str):
    """
    For now, this is a simple rule-based helper.
    You can replace this with your own real hospital/ambulance/pharmacy data.
    """
    # Default ambulance number used in India for emergency
    default_ambulance_number = "108"

    info = {
        "ambulance_available": True,
        "ambulance_number": default_ambulance_number,
        "pharmacies": [
            {"name": f"{location_name} City Pharmacy", "phone": "9999999999"},
            {"name": f"{location_name} Medical Store", "phone": "8888888888"},
        ]
    }
    return info

# -------------------------------------------------
# 6. UI SECTIONS
# -------------------------------------------------

def ui_symptom_checker(symptom_model):
    st.header("🧠 AI Symptom Checker (Home Use)")
    show_motivation()
    show_disclaimer()

    user_symptoms = st.text_area(
        "Describe your symptoms (for example: 'fever, cough, body pain since 2 days')",
        height=120
    )

    if st.button("Analyze Symptoms"):
        if not user_symptoms.strip():
            st.error("Please enter your symptoms.")
            return

        predicted_disease = symptom_model.predict([user_symptoms])[0]
        precautions, medicines = get_disease_info(predicted_disease)

        st.subheader("🩺 Possible Condition")
        st.success(predicted_disease)

        if precautions:
            st.subheader("✅ Suggested Precautions (General)")
            st.write(precautions)

        if medicines:
            st.subheader("💊 Possible Medication (General Information Only)")
            st.write(medicines)

        st.info(
            "This is only a rough prediction based on limited training data. "
            "Always consult a real doctor before taking any decision."
        )

def ui_hospital_doctor_module(doctor_df, doctor_model, feature_cols, doctor_model_acc):
    st.header("🏥 Hospital & Doctor Availability")
    show_motivation()
    show_disclaimer()

    if doctor_df.empty:
        st.error("Doctor data not loaded.")
        return

    st.markdown("#### Select Hospital / Location")
    locations = sorted(doctor_df["location"].dropna().unique())
    if not locations:
        st.error("No `location` values found in `doctor.csv`.")
        return

    selected_location = st.selectbox("Choose a hospital / location:", locations)

    loc_df = doctor_df[doctor_df["location"] == selected_location]
    st.markdown(f"**Total doctors in this location:** {len(loc_df)}")

    # Show ambulance and pharmacy info
    info = get_location_info(selected_location)
    st.subheader("🚑 Ambulance & Emergency")
    if info["ambulance_available"]:
        st.success(f"Ambulance Available ✅ | Contact: {info['ambulance_number']}")
    else:
        st.error("Ambulance Not Available ❌")

    st.subheader("🛒 Nearby Pharmacies")
    for ph in info["pharmacies"]:
        st.write(f"- **{ph['name']}** – 📞 {ph['phone']}")

    st.subheader("👨‍⚕️ Doctors in this Hospital/Location")
    show_cols = [
        "doctor_id", "name", "specialization", "highest_degree", "experience_years",
        "fee", "wait_time", "rating", "is_available_today"
    ]
    show_cols = [c for c in show_cols if c in loc_df.columns]
    st.dataframe(loc_df[show_cols])

    st.markdown("---")
    st.subheader("🔮 Predict Doctor Availability (Model-Based)")

    if doctor_model is None or not feature_cols:
        st.info("Doctor availability model could not be trained (insufficient or unsuitable data). "
                "Using only the availability from the dataset.")
        return

    if doctor_model_acc is not None:
        st.caption(f"Model validation accuracy: **{doctor_model_acc:.2f}**")

    doctor_names = loc_df["name"].dropna().unique()
    selected_doctor_name = st.selectbox("Choose a doctor:", doctor_names)

    selected_doc_row = loc_df[loc_df["name"] == selected_doctor_name].iloc[0]

    # Current status from dataset
    current_status = selected_doc_row.get("is_available_today", "Unknown")
    st.write(f"**Current status from data:** `{current_status}`")

    # Model prediction
    result = predict_doctor_availability(doctor_model, feature_cols, selected_doc_row)
    if result is not None:
        pred, prob = result
        pred_label = "Available" if int(pred) == 1 else "Not Available"
        if prob is not None:
            st.write(f"**Predicted status (model):** `{pred_label}`  "
                     f"(confidence ≈ {prob * 100:.1f}%)")
        else:
            st.write(f"**Predicted status (model):** `{pred_label}`")
    else:
        st.info("Could not compute model-based prediction for this doctor.")

def ui_medicine_module(medicine_df):
    st.header("💊 Medicine Availability & Pharmacy Help")
    show_motivation()
    show_disclaimer()

    if medicine_df.empty:
        st.error("Medicine data not loaded.")
        return

    st.markdown("### Search for a Medicine")
    search_text = st.text_input(
        "Enter disease name or medicine name (e.g., 'fever', 'augmentin 625')"
    )

    if st.button("Check Medicine"):
        if not search_text.strip():
            st.error("Please enter a search term.")
            return

        # Case-insensitive search
        df = medicine_df.copy()
        df["name_lower"] = df["name"].astype(str).str.lower()
        search_lower = search_text.strip().lower()

        matches = df[df["name_lower"].str.contains(search_lower, na=False)]

        if matches.empty:
            st.warning("No exact medicine found with that name. Showing medicines used for similar conditions if any.")
            # Try using 'use0'...'use4'
            use_cols = [c for c in df.columns if c.startswith("use")]
            if use_cols:
                mask = pd.Series(False, index=df.index)
                for c in use_cols:
                    mask = mask | df[c].astype(str).str.lower().str.contains(search_lower, na=False)
                matches = df[mask]

        if matches.empty:
            st.error("No matching medicine found in the dataset.")
            return

        st.success(f"Found {len(matches)} medicine(s). Showing top 5:")
        show_cols = ["id", "name", "availability"]
        show_cols = [c for c in show_cols if c in matches.columns]
        st.dataframe(matches[show_cols].head())

        for _, row in matches.head(5).iterrows():
            st.markdown("---")
            st.subheader(f"Medicine: {row['name']}")
            if "availability" in row:
                st.write(f"**Availability (from data):** `{row['availability']}`")

            # Substitutes
            sub_cols = [c for c in row.index if c.startswith("substitute")]
            substitutes = [row[c] for c in sub_cols if pd.notna(row[c])]
            if substitutes:
                st.markdown("**Possible Substitutes:**")
                for s in substitutes:
                    st.write(f"- {s}")

            # Uses
            use_cols = [c for c in row.index if c.startswith("use")]
            uses = [row[c] for c in use_cols if pd.notna(row[c])]
            if uses:
                st.markdown("**Common Uses / Indications (from dataset):**")
                for u in uses:
                    st.write(f"- {u}")

            # Side effects
            se_cols = [c for c in row.index if c.startswith("sideEffect")]
            side_effects = [row[c] for c in se_cols if pd.notna(row[c])]
            if side_effects:
                st.markdown("**Possible Side Effects (from dataset):**")
                for se in side_effects[:10]:  # limit printing
                    st.write(f"- {se}")

        st.info(
            "Medicine availability above is based on the hospital dataset. "
            "For local pharmacies, please contact them using the phone numbers shown in the Hospital module."
        )

def ui_about():
    st.header("ℹ️ About This App")
    show_motivation()
    st.markdown(
        """
        This app acts as a **mini AI medical assistant** with the following features:

        - ✅ Symptom checker (AI model) – suggests possible diseases, precautions, and generic medicines  
        - ✅ Hospital/Doctor module – shows doctor availability using your `doctor.csv`  
        - ✅ Ambulance & Pharmacy info – simple helper for emergency and local chemist references  
        - ✅ Medicine module – checks medicine availability and substitutes using `medicine.csv`  

        **Important:** This is a **demo / academic project** and must **not** be used as a replacement
        for real doctors, hospitals, or emergency services.
        """
    )

# -------------------------------------------------
# 7. MAIN APP
# -------------------------------------------------

def main():
    # Sidebar navigation
    st.sidebar.title("🩺 AI Doctor & Hospital Assistant")
    page = st.sidebar.radio(
        "Go to:",
        [
            "Symptom Checker (AI Doctor)",
            "Hospital & Doctor Availability",
            "Medicine Availability",
            "About App"
        ]
    )

    # Load data & models once
    doctor_df = load_doctor_data()
    medicine_df = load_medicine_data()
    symptom_model = train_symptom_model(SYMPTOM_DATA)

    doctor_model, feature_cols, doctor_model_acc = (None, [], None)
    if not doctor_df.empty:
        doctor_model, feature_cols, doctor_model_acc = train_doctor_model(doctor_df)

    # Route to pages
    if page == "Symptom Checker (AI Doctor)":
        ui_symptom_checker(symptom_model)
    elif page == "Hospital & Doctor Availability":
        ui_hospital_doctor_module(doctor_df, doctor_model, feature_cols, doctor_model_acc)
    elif page == "Medicine Availability":
        ui_medicine_module(medicine_df)
    else:
        ui_about()

if __name__ == "__main__":
    main()
