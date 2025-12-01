import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ---------------------------------------------------
# 1. BASIC CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Doctor & Hospital Assistant",
    page_icon="🩺",
    layout="wide"
)

PATIENT_QUOTES = [
    "Healing is a journey, not a race.",
    "Every small step towards health counts.",
    "You’ve survived 100% of your hardest days.",
    "Your body can do amazing things. Be kind to it.",
    "Rest is also part of recovery.",
    "You are stronger than you feel right now.",
]

def show_motivation():
    st.markdown("### 💡 Motivational Thought")
    st.info(random.choice(PATIENT_QUOTES))

def show_disclaimer():
    st.warning(
        "⚠️ This app is for **education & guidance only**.\n\n"
        "It is **not** a substitute for professional medical advice, "
        "diagnosis or emergency care. Always consult a qualified doctor."
    )

# ---------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_doctor_data():
    try:
        df = pd.read_csv("doctor.csv")
        return df
    except FileNotFoundError:
        st.error("`doctor.csv` not found in project root.")
        return pd.DataFrame()

@st.cache_data
def load_medicine_data():
    try:
        df = pd.read_csv("medicine.csv")
        # Ensure required columns exist; if not, create defaults
        if "availability" not in df.columns:
            df["availability"] = "Available"
        return df
    except FileNotFoundError:
        st.error("`medicine.csv` not found in project root.")
        return pd.DataFrame()

# ---------------------------------------------------
# 3. DOCTOR AVAILABILITY MODEL (SAFE)
# ---------------------------------------------------
@st.cache_resource
def train_doctor_model(df: pd.DataFrame):
    if "is_available_today" not in df.columns:
        return None, [], None

    try:
        y = df["is_available_today"].astype(int)
    except Exception:
        # Try mapping if stored as Yes/No
        y = df["is_available_today"].astype(str).str.lower().map(
            {"yes": 1, "no": 0, "available": 1, "not available": 0}
        ).fillna(0).astype(int)

    X = df.select_dtypes(include=["int64", "float64"]).copy()

    if X.empty or y.nunique() < 2:
        return None, [], None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        return model, list(X.columns), acc
    except Exception:
        return None, [], None

def predict_doctor_availability(model, feature_cols, row: pd.Series):
    if model is None or not feature_cols:
        return None
    try:
        x = row[feature_cols].to_frame().T.fillna(0)
        pred = model.predict(x)[0]
        prob = model.predict_proba(x)[0][1]
        return int(pred), float(prob)
    except Exception:
        return None

# ---------------------------------------------------
# 4. SYMPTOM → DISEASE MODEL (20+ DISEASES)
# ---------------------------------------------------
SYMPTOM_KB = [
    {
        "symptoms": "fever cold cough sore throat running nose sneezing body pain",
        "disease": "Common Cold / Viral Fever",
        "precautions": "Rest, warm fluids, steam inhalation, avoid cold drinks.",
        "medicine": "Paracetamol 500mg",
        "timing": "1 tablet after food, every 6 hours (max 4 per day).",
        "severity": "low",
        "speciality": "Physician",
        "est_test_cost": 200
    },
    {
        "symptoms": "high fever chills severe body pain headache",
        "disease": "Dengue (Suspected)",
        "precautions": "Immediate doctor visit, avoid self-medication with painkillers other than paracetamol, drink plenty of fluids.",
        "medicine": "Paracetamol 500mg only (no ibuprofen/aspirin).",
        "timing": "1 tablet after food, every 6 hours, as advised.",
        "severity": "high",
        "speciality": "Physician",
        "est_test_cost": 2000
    },
    {
        "symptoms": "chest pain breathlessness sweating pain left arm",
        "disease": "Heart Attack (Possible Emergency)",
        "precautions": "Call ambulance/emergency immediately, do not walk or exert.",
        "medicine": "Aspirin (only if prescribed in plan) – emergency use.",
        "timing": "Single dose as instructed by cardiologist.",
        "severity": "critical",
        "speciality": "Cardiologist",
        "est_test_cost": 8000
    },
    {
        "symptoms": "stomach pain loose motion diarrhea dehydration",
        "disease": "Gastroenteritis / Food Poisoning",
        "precautions": "Drink ORS, avoid street food, maintain hygiene.",
        "medicine": "ORS + Zinc",
        "timing": "ORS after every loose motion; Zinc once daily.",
        "severity": "medium",
        "speciality": "Physician",
        "est_test_cost": 500
    },
    {
        "symptoms": "burning urination lower abdominal pain frequent urination",
        "disease": "Urinary Tract Infection",
        "precautions": "Drink more water, avoid holding urine, maintain hygiene.",
        "medicine": "Doctor-prescribed antibiotic (e.g., Nitrofurantoin).",
        "timing": "As per prescription, usually twice daily after food.",
        "severity": "medium",
        "speciality": "Physician",
        "est_test_cost": 700
    },
    {
        "symptoms": "breathlessness wheezing chest tightness cough at night",
        "disease": "Asthma",
        "precautions": "Avoid dust, smoke, strong smells; use mask.",
        "medicine": "Salbutamol inhaler (reliever) + controller as prescribed.",
        "timing": "As per inhaler schedule advised by pulmonologist.",
        "severity": "high",
        "speciality": "Pulmonologist",
        "est_test_cost": 1500
    },
    {
        "symptoms": "headache vomiting sensitivity to light visual aura",
        "disease": "Migraine",
        "precautions": "Rest in dark, avoid screen and triggers (stress, certain foods).",
        "medicine": "Paracetamol / NSAID; triptan as prescribed.",
        "timing": "When headache starts, not on empty stomach.",
        "severity": "medium",
        "speciality": "Neurologist",
        "est_test_cost": 1500
    },
    {
        "symptoms": "joint pain morning stiffness swelling in joints",
        "disease": "Rheumatoid Arthritis (Suspected)",
        "precautions": "Gentle exercise, avoid self-steroids, consult rheumatologist.",
        "medicine": "Pain relief + disease-modifying drugs (DMARDs) as advised.",
        "timing": "Strictly per rheumatologist schedule.",
        "severity": "medium",
        "speciality": "Orthopedic",
        "est_test_cost": 3000
    },
    {
        "symptoms": "thirst urination weight loss tiredness blurred vision",
        "disease": "Diabetes (Likely)",
        "precautions": "Avoid sugar, control diet, walk regularly, check blood sugar.",
        "medicine": "Metformin (common first-line, doctor decides).",
        "timing": "Usually once/twice daily after food.",
        "severity": "medium",
        "speciality": "Endocrinologist",
        "est_test_cost": 800
    },
    {
        "symptoms": "chest burning after food sour taste lying down worsens",
        "disease": "Acidity / GERD",
        "precautions": "Avoid spicy/oily food, don’t lie down immediately after eating.",
        "medicine": "Omeprazole / Pantoprazole (doctor decides dose).",
        "timing": "Once daily, 30 minutes before breakfast.",
        "severity": "low",
        "speciality": "Gastroenterologist",
        "est_test_cost": 600
    },
    {
        "symptoms": "itchy rash sneezing nose blockage watery eyes",
        "disease": "Allergic Rhinitis",
        "precautions": "Avoid dust, pets (if allergic), strong perfumes.",
        "medicine": "Cetirizine / Loratadine.",
        "timing": "Once daily at night (may cause drowsiness).",
        "severity": "low",
        "speciality": "ENT",
        "est_test_cost": 500
    },
    {
        "symptoms": "ear pain decreased hearing fluid discharge",
        "disease": "Ear Infection",
        "precautions": "Keep ear dry, do not put oil, avoid ear buds.",
        "medicine": "Antibiotic ear drops or oral antibiotics.",
        "timing": "As per ENT advice, usually multiple times daily.",
        "severity": "medium",
        "speciality": "ENT",
        "est_test_cost": 700
    },
    {
        "symptoms": "loose motion blood in stool crampy abdominal pain",
        "disease": "Infective Colitis (Suspected)",
        "precautions": "See doctor soon, maintain hydration.",
        "medicine": "Antibiotics + probiotics as prescribed.",
        "timing": "Per gastroenterologist’s schedule.",
        "severity": "high",
        "speciality": "Gastroenterologist",
        "est_test_cost": 2500
    },
    {
        "symptoms": "severe abdominal pain right lower side fever nausea",
        "disease": "Appendicitis (Suspected Emergency)",
        "precautions": "Do not eat/drink much, go to emergency.",
        "medicine": "Surgery usually needed, IV antibiotics in hospital.",
        "timing": "Hospital-based treatment.",
        "severity": "critical",
        "speciality": "General Surgeon",
        "est_test_cost": 12000
    },
    {
        "symptoms": "shortness of breath swelling legs cannot lie flat",
        "disease": "Heart Failure (Suspected)",
        "precautions": "Urgent cardiology consult, limit salt and fluids.",
        "medicine": "Diuretics, ACE inhibitors as advised.",
        "timing": "As per strict schedule by cardiologist.",
        "severity": "high",
        "speciality": "Cardiologist",
        "est_test_cost": 6000
    },
    {
        "symptoms": "persistent cough weight loss night sweats",
        "disease": "Tuberculosis (Suspected)",
        "precautions": "Visit chest specialist, avoid close contact without mask.",
        "medicine": "Anti-TB regimen (multi-drug) for many months.",
        "timing": "Daily fixed-time dosing for months under supervision.",
        "severity": "high",
        "speciality": "Pulmonologist",
        "est_test_cost": 3000
    },
    {
        "symptoms": "irritability crying baby pulling ear fever",
        "disease": "Pediatric Ear Infection",
        "precautions": "Consult pediatrician, monitor fever, ensure hydration.",
        "medicine": "Pediatric antibiotic drops/syrup.",
        "timing": "Per pediatrician schedule.",
        "severity": "medium",
        "speciality": "Pediatrician",
        "est_test_cost": 700
    },
    {
        "symptoms": "tooth pain swelling in jaw sensitivity to hot cold",
        "disease": "Tooth Infection",
        "precautions": "Avoid cold/sweet foods, keep mouth clean, see dentist.",
        "medicine": "Pain killer + antibiotic as dentist prescribes.",
        "timing": "Usually twice daily after meals.",
        "severity": "medium",
        "speciality": "Dentist",
        "est_test_cost": 1000
    },
    {
        "symptoms": "missed periods nausea vomiting breast tenderness",
        "disease": "Early Pregnancy (Possible)",
        "precautions": "Do pregnancy test, avoid harmful medicines.",
        "medicine": "Folic acid, supplements after gynecologist consult.",
        "timing": "Once daily supplements schedule.",
        "severity": "low",
        "speciality": "Gynecologist",
        "est_test_cost": 1000
    },
    {
        "symptoms": "fever rash joint pain in recently travelled area",
        "disease": "Chikungunya / Viral Fever (Travel Related)",
        "precautions": "Avoid mosquito bites, consult doctor.",
        "medicine": "Paracetamol, fluids.",
        "timing": "Per doctor, avoid NSAIDs unless advised.",
        "severity": "medium",
        "speciality": "Physician",
        "est_test_cost": 2500
    },
]

SYMPTOM_DF = pd.DataFrame(SYMPTOM_KB)

@st.cache_resource
def train_symptom_model():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])
    pipe.fit(SYMPTOM_DF["symptoms"], SYMPTOM_DF["disease"])
    return pipe

def get_disease_row(disease_name: str):
    row = SYMPTOM_DF[SYMPTOM_DF["disease"] == disease_name]
    if row.empty:
        return None
    return row.iloc[0]

# ---------------------------------------------------
# 5. EXTRAS: HOSPITAL INFO HELPERS
# ---------------------------------------------------
def get_hospital_beds(location: str):
    random.seed(hash(location) % 999999)
    return {
        "general": random.randint(10, 80),
        "oxygen": random.randint(5, 40),
        "icu": random.randint(2, 20),
    }

def get_treatment_available(location: str, speciality_needed: str, doctor_df: pd.DataFrame):
    loc_df = doctor_df[doctor_df["location"] == location]
    if "specialization" not in loc_df.columns:
        return False
    specs = loc_df["specialization"].astype(str).str.lower().tolist()
    return speciality_needed.lower() in " ".join(specs)

def cost_estimate(disease_row, doctor_fee_mean: float, med_df: pd.DataFrame):
    base_test = disease_row["est_test_cost"]
    # very rough medicine estimate
    med_cost = 500
    return int(doctor_fee_mean + base_test + med_cost)

# ---------------------------------------------------
# 6. SESSION STATE FOR PATIENT & REMINDERS
# ---------------------------------------------------
if "reminders" not in st.session_state:
    st.session_state["reminders"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------------------------------------------
# 7. UI MODULES
# ---------------------------------------------------

# --- MODULE A: AI Symptom Checker & Triage ---
def ui_symptom_checker(symptom_model, doctor_df, med_df):
    st.header("🧠 AI Symptom Checker (Home Use)")
    show_motivation()
    show_disclaimer()

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name", value="Guest")
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Prefer not to say", "Male", "Female", "Other"])

    with col2:
        location = st.text_input("City / Area (for hospital suggestion)", value="My City")
        emergency_contact = st.text_input("Emergency Contact (Optional)", value="")

    symptoms_text = st.text_area(
        "Describe your symptoms",
        placeholder="Example: high fever, severe body pain, headache since 2 days..."
    )

    if st.button("Analyze Symptoms"):
        if not symptoms_text.strip():
            st.error("Please enter your symptoms.")
            return

        disease_pred = symptom_model.predict([symptoms_text])[0]
        row = get_disease_row(disease_pred)

        st.subheader("🩺 Predicted Condition")
        st.success(disease_pred)

        if row is not None:
            st.markdown("**Precautions (General):**")
            st.write(row["precautions"])

            st.markdown("**Suggested Medicine (General Info Only):**")
            st.write(row["medicine"])

            st.markdown("**⏰ Suggested Timing:**")
            st.write(row["timing"])

            # Triage
            severity = row["severity"]
            if severity == "critical":
                st.error("🚨 This looks like a **possible emergency**. Visit Emergency/ICU immediately.")
            elif severity == "high":
                st.warning("⚠️ This condition may be serious. Consult a doctor **today**.")
            else:
                st.info("ℹ️ Condition appears less severe, but still consult a doctor if not improving.")

            # Add to history
            st.session_state["history"].append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "name": name,
                "age": age,
                "disease": disease_pred,
                "symptoms": symptoms_text
            })

        st.markdown("---")
        st.markdown("### 🚑 Emergency Information")
        st.write("🚑 **Ambulance (India): 108**")
        if emergency_contact:
            st.write(f"☎️ Emergency Contact Saved: {emergency_contact}")

# --- MODULE B: Hospital & Doctor Availability, Beds, Treatment, Cost ---
def ui_hospital_doctor(doctor_df, doctor_model, feature_cols):
    st.header("🏥 Hospital & Doctor Services")
    show_motivation()

    if doctor_df.empty:
        st.error("Doctor data not found.")
        return

    locations = sorted(doctor_df["location"].dropna().unique())
    location = st.selectbox("Select Hospital / Location", locations)

    loc_df = doctor_df[doctor_df["location"] == location]

    st.subheader("👨‍⚕️ Doctors in This Hospital")
    view_cols = [c for c in ["name", "specialization", "rating", "fee", "is_available_today"] if c in loc_df.columns]
    st.dataframe(loc_df[view_cols])

    # Extra feature: filter by specialization
    specs = sorted(loc_df["specialization"].dropna().unique())
    spec_filter = st.multiselect("Filter by Specialization", specs, default=specs)
    fil_df = loc_df[loc_df["specialization"].isin(spec_filter)]
    st.write("Filtered doctors:")
    st.dataframe(fil_df[view_cols])

    # Extra feature: show top-rated doctor
    if "rating" in loc_df.columns:
        best_doc = loc_df.loc[loc_df["rating"].idxmax()]
        st.info(f"⭐ Top Rated Doctor Here: **{best_doc['name']}** "
                f"({best_doc['specialization']}, Rating: {best_doc['rating']})")

    # Bed availability
    st.subheader("🛏️ Bed Availability (Approximate)")
    beds = get_hospital_beds(location)
    col1, col2, col3 = st.columns(3)
    col1.metric("General Beds", beds["general"])
    col2.metric("Oxygen Beds", beds["oxygen"])
    col3.metric("ICU Beds", beds["icu"])

    # Select disease to check treatment availability and cost
    disease_names = list(SYMPTOM_DF["disease"].unique())
    disease_sel = st.selectbox("Select Disease to Check Treatment Availability", disease_names)
    drow = get_disease_row(disease_sel)
    treatment_ok = get_treatment_available(location, drow["speciality"], doctor_df)
    if treatment_ok:
        st.success(f"✅ Treatment for **{disease_sel}** likely available here "
                   f"(specialist: {drow['speciality']}).")
    else:
        st.warning(f"⚠️ Full specialist care for **{disease_sel}** may not be available here.")

    # Cost estimate based on doctor fee and tests
    avg_fee = loc_df["fee"].mean() if "fee" in loc_df.columns else 500
    est_total = cost_estimate(drow, avg_fee, None)
    st.subheader("💰 Estimated Money to be Spent")
    st.write(f"Estimated consultation + tests + medicines ≈ **₹{est_total}** (rough estimate).")

    # Doctor availability prediction
    st.subheader("🔮 Doctor Availability Prediction")
    doc_name = st.selectbox("Choose Doctor for Availability Check", loc_df["name"].unique())
    doc_row = loc_df[loc_df["name"] == doc_name].iloc[0]

    if "is_available_today" in doc_row:
        cur_status = "Available" if int(doc_row["is_available_today"]) == 1 else "Not Available"
        st.info(f"Current (from data): **{cur_status}**")

    if doctor_model is not None and feature_cols:
        result = predict_doctor_availability(doctor_model, feature_cols, doc_row)
        if result is not None:
            pred, prob = result
            pred_label = "Available" if pred == 1 else "Not Available"
            st.success(f"Model prediction: **{pred_label}** (confidence ≈ {prob * 100:.1f}%)")
        else:
            st.warning("Model could not predict for this doctor.")
    else:
        st.info("ML model not trained (insufficient numeric data).")

# --- MODULE C: Medicine Info, Availability & Timing ---
def ui_medicine(medicine_df: pd.DataFrame):
    st.header("💊 Medicine Information & Availability")
    show_motivation()
    show_disclaimer()

    if medicine_df.empty:
        st.error("Medicine data not found.")
        return

    query = st.text_input("Search medicine by name")
    if st.button("Search Medicine"):
        if not query.strip():
            st.error("Enter a medicine name.")
            return

        df = medicine_df.copy()
        df["__name_l"] = df["name"].astype(str).str.lower()
        res = df[df["__name_l"].str.contains(query.lower())]

        if res.empty:
            st.warning("No exact match found.")
        else:
            for _, row in res.iterrows():
                st.markdown("---")
                st.subheader(row["name"])
                st.write("**Use:**", row.get("use0", "NA"))
                st.write("**Substitute:**", row.get("substitute0", "NA"))
                st.write("**Side Effects (common):**", row.get("sideEffect0", "NA"))
                st.write("**Availability:**", row.get("availability", "Unknown"))

                # Extra: suggest basic timing from internal rule
                med_name_lower = str(row["name"]).lower()
                if "paracetamol" in med_name_lower:
                    timing = "Usually 1 tablet after food, every 6 hours (max 4/day)."
                elif "omeprazole" in med_name_lower or "pantoprazole" in med_name_lower:
                    timing = "Usually once daily, 30 minutes before breakfast."
                elif "metformin" in med_name_lower:
                    timing = "Usually after food, as per diabetes schedule."
                else:
                    timing = "Timing depends on doctor’s prescription and dose."
                st.write("**⏰ Typical Timing (general):**", timing)

# --- MODULE D: Medicine Reminder & Alarm-like Planner ---
def ui_medicine_reminders():
    st.header("⏰ Medicine Schedule & Reminders")
    show_motivation()
    show_disclaimer()

    with st.form("reminder_form"):
        med_name = st.text_input("Medicine Name")
        dose = st.text_input("Dose (e.g., 500mg, 1 tablet)")
        times_per_day = st.number_input("Times per day", 1, 6, 2)
        before_after_food = st.selectbox("Before/After Food", ["Before food", "After food", "Doesn't matter"])
        morning = st.checkbox("Morning")
        afternoon = st.checkbox("Afternoon")
        night = st.checkbox("Night")
        duration_days = st.number_input("Duration (days)", 1, 120, 5)
        submit = st.form_submit_button("Add Reminder")

    if submit:
        slots = []
        if morning:
            slots.append("08:00")
        if afternoon:
            slots.append("14:00")
        if night:
            slots.append("20:00")
        if not slots:
            slots = ["08:00", "20:00"]

        st.session_state["reminders"].append({
            "medicine": med_name,
            "dose": dose,
            "times_per_day": times_per_day,
            "food": before_after_food,
            "slots": ", ".join(slots),
            "duration_days": duration_days
        })
        st.success("Reminder added (in-app). Keep this page open to view schedule.")

    if st.session_state["reminders"]:
        st.subheader("Your Medicine Schedule")
        st.table(pd.DataFrame(st.session_state["reminders"]))

        now = datetime.now().time()
        st.info(
            "🕒 This is an **in-app** reminder planner. "
            "For real phone alarms, please set alarms in your mobile using this schedule."
        )

# --- MODULE E: Payment, Appointment Booking & Costing ---
def ui_payment_appointment(doctor_df: pd.DataFrame):
    st.header("💳 Payment, Appointment & Cost Planner")
    show_motivation()
    show_disclaimer()

    if doctor_df.empty:
        st.error("Doctor data not found.")
        return

    locations = sorted(doctor_df["location"].dropna().unique())
    location = st.selectbox("Select Hospital", locations)
    loc_df = doctor_df[doctor_df["location"] == location]

    doctor_name = st.selectbox("Select Doctor", loc_df["name"].unique())
    doc_row = loc_df[loc_df["name"] == doctor_name].iloc[0]

    base_fee = doc_row.get("fee", 500)
    st.write(f"Doctor Fee (from data): **₹{base_fee}**")

    consult_type = st.selectbox("Consultation Type", ["OPD Visit", "Video Consultation", "Emergency Visit"])
    if consult_type == "Video Consultation":
        consult_multiplier = 0.8
    elif consult_type == "Emergency Visit":
        consult_multiplier = 1.5
    else:
        consult_multiplier = 1.0

    service_charge = int(0.05 * base_fee)
    total_fee = int(base_fee * consult_multiplier + service_charge)

    st.subheader("💰 Payment Summary")
    st.write(f"Base Fee: ₹{base_fee}")
    st.write(f"Consultation Type Multiplier: x{consult_multiplier}")
    st.write(f"Service/Platform Charge: ₹{service_charge}")
    st.success(f"Total Payable: **₹{total_fee}**")

    # Payment methods (mock)
    st.subheader("💳 Select Payment Method")
    pay_method = st.radio("Payment Mode", ["UPI", "Credit/Debit Card", "Net Banking", "Cash at Hospital"])
    if st.button("Pay Now (Demo)"):
        st.success(f"Payment via **{pay_method}** simulated successfully. (Demo only)")

    # Appointment booking
    st.subheader("📅 Book Appointment (Demo)")
    appt_date = st.date_input("Select Date")
    appt_slot = st.selectbox("Time Slot", ["9-10 AM", "10-11 AM", "11-12 AM", "4-5 PM", "5-6 PM"])
    if st.button("Confirm Appointment"):
        st.success(
            f"Appointment booked with **{doctor_name}** at **{location}** "
            f"on **{appt_date}**, slot **{appt_slot}** (demo only)."
        )

# --- MODULE F: Extra Insights & History (5+ extra features) ---
def ui_extra_features():
    st.header("📊 Extra Features & Patient History")
    show_motivation()

    # Feature: Show symptom-check history
    st.subheader("📝 Symptom Check History (This Session)")
    if st.session_state["history"]:
        st.table(pd.DataFrame(st.session_state["history"]))
    else:
        st.info("No history yet. Use the AI Symptom Checker to generate history.")

    # Feature: Health tips based on last disease
    st.subheader("🌱 Health Tips Based on Last Predicted Disease")
    if st.session_state["history"]:
        last = st.session_state["history"][-1]
        st.write(f"Last predicted condition: **{last['disease']}**")
        st.write("General advice: Maintain regular sleep, balanced diet, exercise as allowed, and follow doctor instructions.")
    else:
        st.info("No disease predicted yet in this session.")

    # Feature: Quick emergency guidance
    st.subheader("🚨 Quick Emergency Guidance")
    st.write("- Severe chest pain + breathlessness → Call ambulance immediately.")
    st.write("- Sudden weakness on one side of body → Suspected stroke, rush to hospital.")
    st.write("- Uncontrolled bleeding, seizures, loss of consciousness → Emergency care required.")

    # Feature: Downloadable visit summary text
    st.subheader("📄 Downloadable Visit Summary (Simple)")
    if st.session_state["history"]:
        last = st.session_state["history"][-1]
        summary_text = (
            f"Patient: {last['name']}, Age: {last['age']}\n"
            f"Predicted Disease: {last['disease']}\n"
            f"Symptoms: {last['symptoms']}\n"
            f"DateTime: {last['time']}\n\n"
            "Note: This is an auto-generated summary from an educational app, "
            "not a real prescription."
        )
        st.download_button(
            "Download Last Visit Summary",
            data=summary_text,
            file_name="visit_summary.txt"
        )
    else:
        st.info("No summary to download yet.")

    # Feature: Teleconsultation mode flag
    st.subheader("📱 Teleconsultation Mode (Flag)")
    tele = st.checkbox("I prefer online / teleconsultation (video call).")
    if tele:
        st.info("You selected teleconsultation. In a real app, this would connect to a video platform.")
    else:
        st.info("You selected in-person visit.")

# --- MODULE G: About & Feature List ---
def ui_about():
    st.header("ℹ️ About This App & Features")
    st.markdown("""
This app simulates a **real-world AI medical assistant + hospital helper** for academic/project use.

**Key Features (25+ style features merged into modules)**

1. AI symptom checker → predicts disease  
2. Shows precautions for disease  
3. Suggests general medicines (non-prescription style)  
4. Suggests typical medicine timing (morning/evening, before/after food)  
5. Motivational quotes for patients  
6. Doctor list per hospital  
7. Filter doctors by specialization  
8. Show top-rated doctor per hospital  
9. Show bed availability (general/oxygen/ICU)  
10. Check treatment availability for disease in a hospital  
11. Rough cost estimate (consult + tests + meds)  
12. ML-based doctor availability prediction  
13. Medicine search with use, substitute & side-effect info  
14. Medicine availability status from dataset  
15. Medicine timing hints for common drugs  
16. Medicine schedule planner (in-app reminders)  
17. Payment simulation with multiple modes (UPI, card, net banking, cash)  
18. Appointment booking with date & time slot  
19. Symptom-check history for current session  
20. Emergency helpline info & triage hints (ER vs OPD)  
21. Downloadable simple visit summary text  
22. Teleconsultation preference flag  
23. Patient basic profile fields (name, age, gender, area)  
24. Disease severity-based triage messaging (critical/high/low)  
25. Hospital selection interface and per-location views  

This is **NOT** a clinical product. It is meant as a **college/final-year project / demo**.
    """)

# ---------------------------------------------------
# 8. MAIN
# ---------------------------------------------------
def main():
    st.sidebar.title("🩺 AI Doctor & Hospital Assistant")

    page = st.sidebar.radio(
        "Go to:",
        [
            "AI Symptom Checker",
            "Hospital & Doctor Services",
            "Medicine Info & Availability",
            "Medicine Reminders",
            "Payment & Appointment",
            "Extra Features & History",
            "About"
        ]
    )

    doctor_df = load_doctor_data()
    medicine_df = load_medicine_data()
    symptom_model = train_symptom_model()
    doctor_model, feature_cols, _ = train_doctor_model(doctor_df)

    if page == "AI Symptom Checker":
        ui_symptom_checker(symptom_model, doctor_df, medicine_df)
    elif page == "Hospital & Doctor Services":
        ui_hospital_doctor(doctor_df, doctor_model, feature_cols)
    elif page == "Medicine Info & Availability":
        ui_medicine(medicine_df)
    elif page == "Medicine Reminders":
        ui_medicine_reminders()
    elif page == "Payment & Appointment":
        ui_payment_appointment(doctor_df)
    elif page == "Extra Features & History":
        ui_extra_features()
    else:
        ui_about()

if __name__ == "__main__":
    main()
