"""User-facing web app.

    uv run streamlit run ui/app.py
Set API_URL to your deployed API (defaults to localhost)."""

import os

import requests
import streamlit as st

# Where the API lives. Locally: localhost. In the cloud: set the API_URL
# environment variable/secret to your Render URL. No code change needed.
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# The model expects survey CODES (e.g. Age 7 = 50-54). Users see friendly labels;
# a label's position in its list + 1 is its code. Keep these in the same order
# as the codes documented in src/schema.py.

AGE_GROUPS = [
    "18–24",
    "25–29",
    "30–34",
    "35–39",
    "40–44",
    "45–49",
    "50–54",
    "55–59",
    "60–64",
    "65–69",
    "70–74",
    "75–79",
    "80 or older",
]
EDUCATION = [
    "Never attended school",
    "Grades 1–8",
    "Grades 9–11",
    "High school graduate",
    "Some college or technical school",
    "College graduate",
]
INCOME = [
    "Under $10,000",
    "$10,000–15,000",
    "$15,000–20,000",
    "$20,000–25,000",
    "$25,000–35,000",
    "$35,000–50,000",
    "$50,000–75,000",
    "$75,000 or more",
]
GENERAL_HEALTH = ["Excellent", "Very good", "Good", "Fair", "Poor"]
RISK_MESSAGES = {
    "low": ("success", "Your estimated risk is low."),
    "moderate": ("warning", "Your estimated risk is moderate."),
    "high": ("error", "Your estimated risk is high. Consider asking a doctor about a blood sugar test."),
}

st.set_page_config(page_title="Diabetes risk check", page_icon="🩺")
st.title("Diabetes risk check")
st.write(
    "Answer a few questions to get an estimate of your risk of diabetes or prediabetes. "
    "Nothing that identifies you is stored."
)


def yes(label: str, help_text: str | None = None) -> int:
    return int(st.radio(label, ["No", "Yes"], horizontal=True, help=help_text) == "Yes")


with st.form("risk_form"):
    st.subheader("About you")
    c1, c2 = st.columns(2)
    sex = c1.radio("Sex", ["Female", "Male"], horizontal=True)
    age = c2.selectbox("Age", AGE_GROUPS, index=6)
    height_cm = c1.number_input("Height (cm)", 120, 230, 170)
    weight_kg = c2.number_input("Weight (kg)", 30, 300, 75)
    education = st.selectbox("Highest education", EDUCATION, index=4)
    income = st.selectbox("Household income per year", INCOME, index=5)

    st.subheader("Health history")
    high_bp = yes("Have you been told you have high blood pressure?")
    high_chol = yes("Have you been told you have high cholesterol?")
    chol_check = yes("Have you had a cholesterol check in the past 5 years?")
    stroke = yes("Have you ever had a stroke?")
    heart = yes("Have you had heart disease or a heart attack?")
    diff_walk = yes("Do you have serious difficulty walking or climbing stairs?")

    st.subheader("Lifestyle")
    smoker = yes("Have you smoked at least 100 cigarettes in your life?")
    activity = yes("Have you exercised in the past 30 days (outside of work)?")
    fruits = yes("Do you eat fruit at least once a day?")
    veggies = yes("Do you eat vegetables at least once a day?")
    alcohol = yes("Do you drink heavily?", "Men: more than 14 drinks a week. Women: more than 7.")

    st.subheader("Wellbeing and access to care")
    gen_hlth = st.select_slider("How is your health in general?", GENERAL_HEALTH, value="Good")
    ment_hlth = st.slider("In the past 30 days, how many days was your mental health not good?", 0, 30, 0)
    phys_hlth = st.slider("In the past 30 days, how many days was your physical health not good?", 0, 30, 0)
    healthcare = yes("Do you have any health insurance or healthcare coverage?")
    no_doc_cost = yes("In the past year, did cost stop you from seeing a doctor?")

    submitted = st.form_submit_button("Check my risk")

if submitted:
    # Users rarely know their BMI, so we ask for height and weight and compute it
    bmi = round(weight_kg / (height_cm / 100) ** 2, 1)
    payload = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "BMI": min(max(bmi, 12), 98),
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart,
        "PhysActivity": activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": alcohol,
        "AnyHealthcare": healthcare,
        "NoDocbcCost": no_doc_cost,
        "GenHlth": GENERAL_HEALTH.index(gen_hlth) + 1,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
        "DiffWalk": diff_walk,
        "Sex": int(sex == "Male"),
        "Age": AGE_GROUPS.index(age) + 1,
        "Education": EDUCATION.index(education) + 1,
        "Income": INCOME.index(income) + 1,
    }
    try:
        with st.spinner("Calculating (the first request can take up to a minute while the server wakes up)…"):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
            response.raise_for_status()
        result = response.json()
        kind, message = RISK_MESSAGES[result["risk_level"]]
        getattr(st, kind)(message)
        st.metric("Estimated probability", f"{result['probability']:.0%}")
        st.caption(f"Your BMI: {bmi} · Model version {result['model_version']}")
        st.info(result["disclaimer"])
    except requests.RequestException as exc:
        st.error(f"The risk service couldn't be reached at {API_URL}. Check that the API is running. ({exc})")
