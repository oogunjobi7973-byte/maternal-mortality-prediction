import requests
import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Early Maternal Mortality Prediction System",
    page_icon="🩺",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000/predict"

# ==========================================================
# Lookup Dictionaries
# ==========================================================

EDUCATION = {0: "No Education", 1: "Primary", 2: "Secondary", 3: "Higher"}

WEALTH = {
    1: "Poorest",
    2: "Poorer",
    3: "Middle",
    4: "Richer",
    5: "Richest",
}

RESIDENCE = {1: "Urban", 2: "Rural"}

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:
    st.title("🩺 Maternal Mortality Prediction")
    st.success("✅ Model Status: Loaded")

    st.markdown(
        """
### Machine Learning Model
Explainable Boosting Machine (EBM)

---

### Explainability
SHAP (SHapley Additive Explanations)

---

### Dataset
Nigeria Demographic and Health Survey (NDHS 2024)

---

### Purpose
Assist healthcare professionals in identifying women who may be at increased maternal mortality risk using explainable machine learning.
"""
    )

    st.info(
        "This application is intended to support clinical decision-making and "
        "should not replace professional medical judgement."
    )

# ==========================================================
# Main Title
# ==========================================================

st.title("🩺 Early Maternal Mortality Prediction System")

st.markdown(
    """
This Clinical Decision Support System predicts maternal mortality risk
using an Explainable Boosting Machine (EBM).

Complete the patient's information below to obtain an individualized
risk assessment together with an explanation of the factors influencing
the prediction.
"""
)

st.divider()

# ==========================================================
# Patient Information Form
# ==========================================================

st.subheader("📋 Patient Information")

left, right = st.columns(2)

with left:
    age = st.number_input(
        "Maternal Age (Years)", min_value=15, max_value=49, value=25
    )
    education = st.selectbox(
        "Education Level",
        options=list(EDUCATION.keys()),
        format_func=lambda x: EDUCATION[x],
    )
    children = st.number_input(
        "Children Ever Born", min_value=0, max_value=20, value=1
    )
    first_birth = st.number_input(
        "Age at First Birth", min_value=10, max_value=49, value=20
    )

with right:
    residence = st.selectbox(
        "Place of Residence",
        options=list(RESIDENCE.keys()),
        format_func=lambda x: RESIDENCE[x],
    )
    wealth = st.selectbox(
        "Household Wealth Index",
        options=list(WEALTH.keys()),
        format_func=lambda x: WEALTH[x],
    )
    living_children = st.number_input(
        "Living Children", min_value=0, max_value=20, value=1
    )

st.divider()

# ==========================================================
# Prediction Action & Display
# ==========================================================

if st.button("🔍 Predict Maternal Mortality Risk", use_container_width=True):
    patient_data = {
        "age": age,
        "residence": residence,
        "education": education,
        "wealth": wealth,
        "children": children,
        "first_birth": first_birth,
        "living_children": living_children,
    }

    try:
        with st.spinner("Analyzing patient information..."):
            response = requests.post(API_URL, json=patient_data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction completed successfully.")

            # Prediction Banner
            is_high_risk = result["prediction"] == "High Risk"
            if is_high_risk:
                st.error("🔴 HIGH MATERNAL MORTALITY RISK")
            else:
                st.success("🟢 LOW MATERNAL MORTALITY RISK")

            # Prediction Metrics
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Prediction", result["prediction"])
            metric2.metric("Risk Probability", f"{result['risk_probability']}%")
            metric3.metric("Confidence", f"{result['confidence']}%")

            st.divider()

            # Clinical Recommendation
            st.subheader("💡 Clinical Recommendation")
            if is_high_risk:
                st.error(result["recommendation"])
            else:
                st.success(result["recommendation"])

            st.divider()

            # SHAP Explainability Section
            st.subheader("🧠 SHAP Explainability")
            st.markdown(
                """
The Explainable Boosting Machine (EBM) uses SHAP (SHapley Additive Explanations)
to show which patient characteristics contributed most to the prediction.

Positive values increase maternal mortality risk, while negative values reduce the predicted risk.
"""
            )

            risk_col, protect_col = st.columns(2)

            # Factors Increasing Risk
            with risk_col:
                st.error("🔺 Factors Increasing Risk")
                if result.get("top_risk_factors"):
                    for factor in result["top_risk_factors"]:
                        st.metric(
                            label=factor["feature"],
                            value=f"+{factor['impact']:.4f}",
                        )
                else:
                    st.success("No major risk-increasing factors detected.")

            # Factors Reducing Risk
            with protect_col:
                st.success("🔻 Factors Reducing Risk")
                if result.get("top_protective_factors"):
                    for factor in result["top_protective_factors"]:
                        st.metric(
                            label=factor["feature"],
                            value=f"{factor['impact']:.4f}",
                        )
                else:
                    st.info("No major protective factors detected.")

            st.divider()

            # Overall Interpretation
            st.subheader("📊 Overall Risk Interpretation")
            if is_high_risk:
                st.error(
                    f"""
The model predicts a **HIGH maternal mortality risk** with an estimated probability of **{result['risk_probability']}%**.

The SHAP explanation above identifies the most influential patient characteristics that contributed to this prediction.

Closer clinical assessment and monitoring are recommended.
"""
                )
            else:
                st.success(
                    f"""
The model predicts a **LOW maternal mortality risk** with an estimated probability of **{result['risk_probability']}%**.

Although the predicted risk is relatively low, routine antenatal care and continued monitoring remain essential throughout pregnancy.
"""
                )

            st.divider()

            # Patient Summary
            st.subheader("👩‍⚕️ Patient Summary")
            patient = result.get("patient", patient_data)
            sum_left, sum_right = st.columns(2)

            with sum_left:
                st.write(f"**Age:** {patient['age']} years")
                st.write(f"**Residence:** {RESIDENCE.get(patient['residence'], 'N/A')}")
                st.write(f"**Education:** {EDUCATION.get(patient['education'], 'N/A')}")
                st.write(f"**Wealth Index:** {WEALTH.get(patient['wealth'], 'N/A')}")

            with sum_right:
                st.write(f"**Children Ever Born:** {patient['children']}")
                st.write(f"**Living Children:** {patient['living_children']}")
                st.write(f"**Age at First Birth:** {patient['first_birth']} years")

            st.divider()
            st.caption(
                "Prediction generated using an Explainable Boosting Machine (EBM) with SHAP Explainability."
            )

        else:
            st.error(f"Prediction failed (HTTP {response.status_code}).")

    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the FastAPI server.")
    except requests.exceptions.Timeout:
        st.error("The request timed out.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ==========================================================
# Footer
# ==========================================================

st.divider()
st.caption(
    "Early Maternal Mortality Prediction System | "
    "Explainable Boosting Machine (EBM) | "
    "SHAP Explainability | "
    "Nigeria Demographic and Health Survey (NDHS 2024)"
)