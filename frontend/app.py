import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Early Maternal Mortality Prediction System",
    page_icon="🩺",
    layout="wide",
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://maternal-mortality-prediction-yn6a.onrender.com",
).rstrip("/")

EDUCATION = {0: "No Education", 1: "Primary", 2: "Secondary", 3: "Higher"}
WEALTH = {1: "Poorest", 2: "Poorer", 3: "Middle", 4: "Richer", 5: "Richest"}
RESIDENCE = {1: "Urban", 2: "Rural"}

# ==========================================================
# Session Helpers
# ==========================================================

if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None


def api_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def authenticate(endpoint, payload):
    response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=30)
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", "Authentication failed.")
        except Exception:
            detail = "Authentication failed."
        raise RuntimeError(detail)
    return response.json()


def logout():
    st.session_state.token = None
    st.session_state.user = None
    st.rerun()

# ==========================================================
# Authentication Screen
# ==========================================================

if not st.session_state.token:
    st.title("🩺 Early Maternal Mortality Prediction System")
    st.caption("Secure access for healthcare workers")

    login_tab, register_tab = st.tabs(["Sign In", "Create Account"])

    with login_tab:
        st.subheader("Healthcare Worker Sign In")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Sign In", use_container_width=True):
            if not email or not password:
                st.warning("Enter your email and password.")
            else:
                try:
                    result = authenticate("/auth/login", {"email": email, "password": password})
                    st.session_state.token = result["access_token"]
                    st.session_state.user = result["user"]
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    with register_tab:
        st.subheader("Create Healthcare Worker Account")
        name = st.text_input("Full Name", key="register_name")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm = st.text_input("Confirm Password", type="password", key="register_confirm")

        if st.button("Create Account", use_container_width=True):
            if not name or not email or not password:
                st.warning("Complete all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif len(password) < 8:
                st.error("Password must contain at least 8 characters.")
            else:
                try:
                    result = authenticate(
                        "/auth/register",
                        {"name": name, "email": email, "password": password},
                    )
                    st.session_state.token = result["access_token"]
                    st.session_state.user = result["user"]
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.info("Create an account or sign in to access the maternal risk prediction dashboard.")
    st.stop()

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:
    st.title("🩺 Maternal Mortality Prediction")
    st.success("✅ Model Status: Loaded")

    user = st.session_state.user or {}
    st.write(f"**Signed in as:** {user.get('name', 'Healthcare Worker')}")
    st.caption(user.get("email", ""))

    if st.button("Sign Out", use_container_width=True):
        logout()

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

### Data Storage
PostgreSQL Database

---

### Purpose
Assist healthcare professionals in identifying women who may be at increased maternal mortality risk using explainable machine learning.
"""
    )

    st.info(
        "This application is intended to support clinical decision-making and should not replace professional medical judgement."
    )

# ==========================================================
# Main Dashboard
# ==========================================================

st.title("🩺 Early Maternal Mortality Prediction System")
st.markdown(
    """
This Clinical Decision Support System predicts maternal mortality risk using an Explainable Boosting Machine (EBM).

Complete the patient's information below to obtain an individualized risk assessment together with an explanation of the factors influencing the prediction.
"""
)

st.divider()
st.subheader("📋 Patient Information")

left, right = st.columns(2)

with left:
    age = st.number_input("Maternal Age (Years)", min_value=15, max_value=49, value=25)
    education = st.selectbox("Education Level", options=list(EDUCATION.keys()), format_func=lambda x: EDUCATION[x])
    children = st.number_input("Children Ever Born", min_value=0, max_value=20, value=1)
    first_birth = st.number_input("Age at First Birth", min_value=10, max_value=49, value=20)

with right:
    residence = st.selectbox("Place of Residence", options=list(RESIDENCE.keys()), format_func=lambda x: RESIDENCE[x])
    wealth = st.selectbox("Household Wealth Index", options=list(WEALTH.keys()), format_func=lambda x: WEALTH[x])
    living_children = st.number_input("Living Children", min_value=0, max_value=20, value=1)

st.divider()

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
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=patient_data,
                headers=api_headers(),
                timeout=60,
            )

        if response.status_code == 401:
            st.session_state.token = None
            st.session_state.user = None
            st.error("Your session has expired. Please sign in again.")
            st.stop()

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction completed and securely stored.")

            is_high_risk = result["prediction"] == "High Risk"
            if is_high_risk:
                st.error("🔴 HIGH MATERNAL MORTALITY RISK")
            else:
                st.success("🟢 LOW MATERNAL MORTALITY RISK")

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric("Prediction", result["prediction"])
            metric2.metric("Risk Probability", f"{result['risk_probability']}%")
            metric3.metric("Confidence", f"{result['confidence']}%")

            st.caption(f"Prediction ID: {result.get('prediction_id', 'N/A')} | Patient Code: {result['patient'].get('patient_code', 'N/A')}")
            st.divider()

            st.subheader("💡 Clinical Recommendation")
            if is_high_risk:
                st.error(result["recommendation"])
            else:
                st.success(result["recommendation"])

            st.divider()
            st.subheader("🧠 SHAP Explainability")
            st.markdown(
                "Positive values increase predicted risk, while negative values reduce predicted risk."
            )

            risk_col, protect_col = st.columns(2)
            with risk_col:
                st.error("🔺 Factors Increasing Risk")
                if result.get("top_risk_factors"):
                    for factor in result["top_risk_factors"]:
                        st.metric(label=factor["feature"], value=f"+{factor['impact']:.4f}")
                else:
                    st.success("No major risk-increasing factors detected.")

            with protect_col:
                st.success("🔻 Factors Reducing Risk")
                if result.get("top_protective_factors"):
                    for factor in result["top_protective_factors"]:
                        st.metric(label=factor["feature"], value=f"{factor['impact']:.4f}")
                else:
                    st.info("No major protective factors detected.")

            st.divider()
            st.subheader("📊 Overall Risk Interpretation")
            if is_high_risk:
                st.error(
                    f"The model predicts a **HIGH maternal mortality risk** with an estimated probability of **{result['risk_probability']}%**. Closer clinical assessment and monitoring are recommended."
                )
            else:
                st.success(
                    f"The model predicts a **LOW maternal mortality risk** with an estimated probability of **{result['risk_probability']}%**. Routine antenatal care and continued monitoring remain essential."
                )

            st.divider()
            st.subheader("👩‍⚕️ Patient Summary")
            patient = result["patient"]
            sum_left, sum_right = st.columns(2)
            with sum_left:
                st.write(f"**Patient Code:** {patient['patient_code']}")
                st.write(f"**Age:** {patient['age']} years")
                st.write(f"**Residence:** {RESIDENCE.get(patient['residence'], 'N/A')}")
                st.write(f"**Education:** {EDUCATION.get(patient['education'], 'N/A')}")
                st.write(f"**Wealth Index:** {WEALTH.get(patient['wealth'], 'N/A')}")
            with sum_right:
                st.write(f"**Children Ever Born:** {patient['children']}")
                st.write(f"**Living Children:** {patient['living_children']}")
                st.write(f"**Age at First Birth:** {patient['first_birth']} years")

        else:
            try:
                detail = response.json().get("detail", "Prediction failed.")
            except Exception:
                detail = f"Prediction failed (HTTP {response.status_code})."
            st.error(detail)

    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the FastAPI server.")
    except requests.exceptions.Timeout:
        st.error("The request timed out.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

# ==========================================================
# Prediction History
# ==========================================================

st.divider()
st.subheader("🗂️ Recent Prediction History")

if st.button("Refresh History"):
    try:
        response = requests.get(
            f"{API_BASE_URL}/predictions/history",
            headers=api_headers(),
            timeout=30,
        )
        if response.status_code == 200:
            history = response.json()
            if history:
                st.dataframe(history, use_container_width=True)
            else:
                st.info("No prediction records are available for this account yet.")
        else:
            st.error("Unable to retrieve prediction history.")
    except Exception as exc:
        st.error(f"Unable to retrieve prediction history: {exc}")

st.divider()
st.caption(
    "Early Maternal Mortality Prediction System | Explainable Boosting Machine (EBM) | SHAP Explainability | PostgreSQL"
)
