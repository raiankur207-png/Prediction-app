
import os

import streamlit as st
import pandas as pd
import joblib

# ============================================================
# LOAD MODEL AND FEATURE COLUMNS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "employee_retention_xgboost_model.pkl")
FEATURE_FILE = os.path.join(BASE_DIR, "feature_columns.pkl")

try:
    model = joblib.load(MODEL_FILE)
    feature_columns = joblib.load(FEATURE_FILE)
except FileNotFoundError:
    st.error(
        "Model files not found. Please keep these files in the same folder as app.py:\n\n"
        "• employee_retention_xgboost_model.pkl\n"
        "• feature_columns.pkl"
    )
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Retention Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("👨‍💼 Employee Retention Prediction")
st.write(
    "Enter employee details below to predict whether the employee "
    "is likely to leave the organization."
)

st.info(
    "This app uses the XGBoost model trained in your Employee Retention "
    "Prediction project."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Employee Information")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.text_input(
        "City",
        value="city_103",
        help="Example: city_103"
    )

    city_development_index = st.number_input(
        "City Development Index",
        min_value=0.0,
        max_value=1.0,
        value=0.920,
        step=0.001
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    relevant_experience = st.selectbox(
        "Relevant Experience",
        [
            "Has relevent experience",
            "No relevent experience"
        ]
    )

with col2:
    enrolled_university = st.selectbox(
        "Enrolled University",
        [
            "no_enrollment",
            "Part time course",
            "Full time course"
        ]
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "Graduate",
            "Masters",
            "High School",
            "Phd",
            "Primary School"
        ]
    )

    major_discipline = st.selectbox(
        "Major Discipline",
        [
            "STEM",
            "Business Degree",
            "Arts",
            "Humanities",
            "No Major",
            "Other"
        ]
    )

    experience = st.selectbox(
        "Experience",
        [
            "<1", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12", "13", "14", "15", "16", "17", "18",
            "19", "20", ">20"
        ]
    )

with col3:
    company_size = st.selectbox(
        "Company Size",
        [
            "<10",
            "10/49",
            "50-99",
            "100-500",
            "500-999",
            "1000-4999",
            "5000-9999",
            "10000+"
        ]
    )

    company_type = st.selectbox(
        "Company Type",
        [
            "Pvt Ltd",
            "Funded Startup",
            "Early Stage Startup",
            "Other",
            "Public Sector",
            "NGO"
        ]
    )

    last_new_job = st.selectbox(
        "Last New Job",
        [
            "never",
            "1",
            "2",
            "3",
            "4",
            ">4"
        ]
    )

    training_hours = st.number_input(
        "Training Hours",
        min_value=0,
        max_value=500,
        value=50,
        step=1
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({
    "city": [city],
    "city_development_index": [city_development_index],
    "gender": [gender],
    "relevent_experience": [relevant_experience],
    "enrolled_university": [enrolled_university],
    "education_level": [education_level],
    "major_discipline": [major_discipline],
    "experience": [experience],
    "company_size": [company_size],
    "company_type": [company_type],
    "last_new_job": [last_new_job],
    "training_hours": [training_hours]
})


# ============================================================
# SAME PREPROCESSING USED DURING TRAINING
# ============================================================

# One-hot encoding, exactly like the notebook
input_encoded = pd.get_dummies(
    input_data,
    drop_first=True
)

# Convert feature names to strings
input_encoded.columns = input_encoded.columns.astype(str)

# Clean special characters exactly like the notebook
input_encoded.columns = (
    input_encoded.columns
    .str.replace("[", "_", regex=False)
    .str.replace("]", "_", regex=False)
    .str.replace("<", "_", regex=False)
    .str.replace(">", "_", regex=False)
)

# Make the input columns exactly match the model's training columns
input_encoded = input_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

if st.button("🔮 Predict Employee Retention", use_container_width=True):

    try:
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🔴 High Risk: Employee is likely to leave")
        else:
            st.success("🟢 Low Risk: Employee is likely to stay")

        col_a, col_b = st.columns(2)

        with col_a:
            st.metric(
                "Probability of Leaving",
                f"{probability * 100:.2f}%"
            )

        with col_b:
            st.metric(
                "Probability of Staying",
                f"{(1 - probability) * 100:.2f}%"
            )

        # Simple risk interpretation
        if probability >= 0.70:
            st.warning("⚠️ Very High Risk of leaving")
        elif probability >= 0.50:
            st.warning("⚠️ Moderate to High Risk of leaving")
        elif probability >= 0.30:
            st.info("ℹ️ Moderate Risk of leaving")
        else:
            st.success("✅ Low Risk of leaving")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write(
            "Make sure app.py, the XGBoost model file, and "
            "feature_columns.pkl are from the same project."
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("About this project"):
    st.write(
        """
        **Project:** Employee Retention Prediction

        **Machine Learning Model:** XGBoost Classifier

        **Target:**
        - 0 = Employee likely to stay
        - 1 = Employee likely to leave

        **Main preprocessing:** One-hot encoding with the same feature
        columns used during model training.
        """
    )
