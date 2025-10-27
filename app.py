# ============================================
# 🧠 HealthAI Frontend (Streamlit)
# ============================================

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 🌐 Backend Configuration
# ------------------------------
BACKEND_URL = "http://127.0.0.1:8000"  # Change if your backend runs elsewhere

# ------------------------------
# ⚙️ Helper Function
# ------------------------------
def call_backend(endpoint: str, payload: dict):
    """Send a POST request to the backend API endpoint."""
    try:
        response = requests.post(f"{BACKEND_URL}/{endpoint}", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Unable to connect to backend. Make sure backend.py is running.")
        return None


# ------------------------------
# 🎨 Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="AI Health Assistant", page_icon="🧠", layout="wide")

st.title("🧠 HealthAI Assistant")
st.caption("Your personal AI-powered health companion (Educational use only)")

# ------------------------------
# 🧍 User Details (Top-left Sidebar)
# ------------------------------
st.sidebar.header("👤 User Details")

# Initialize session state
if "user_details" not in st.session_state:
    st.session_state.user_details = {
        "name": "",
        "age": "",
        "gender": "",
        "medical_history": "",
        "current_medication": ""
    }

# Input fields in proper sequence
st.session_state.user_details["name"] = st.sidebar.text_input("Full Name:", value=st.session_state.user_details["name"])
st.session_state.user_details["age"] = st.sidebar.number_input(
    "Age:", 
    min_value=1, 
    max_value=120, 
    step=1, 
    value=int(st.session_state.user_details["age"]) if str(st.session_state.user_details["age"]).isdigit() else 25
)
st.session_state.user_details["gender"] = st.sidebar.selectbox(
    "Gender:", 
    options=["Select", "Male", "Female", "Other"], 
    index=["Select", "Male", "Female", "Other"].index(st.session_state.user_details["gender"]) if st.session_state.user_details["gender"] in ["Select", "Male", "Female", "Other"] else 0
)
st.session_state.user_details["medical_history"] = st.sidebar.text_area(
    "Medical History:", 
    value=st.session_state.user_details["medical_history"], 
    height=100, 
    placeholder="E.g., diabetes, asthma, hypertension"
)
st.session_state.user_details["current_medication"] = st.sidebar.text_area(
    "Current Medication:", 
    value=st.session_state.user_details["current_medication"], 
    height=80, 
    placeholder="E.g., Metformin, Paracetamol"
)

st.sidebar.divider()

# ------------------------------
# 🗂️ Navigation
# ------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["💬 AI Chat", "🧩 Disease Prediction", "💊 Treatment Plan", "📊 Analytics Dashboard"]
)

# ------------------------------
# 💬 AI Chat Page
# ------------------------------
if page == "💬 AI Chat":
    st.header("💬 Chat with HealthAI Assistant")
    st.write("Ask any health-related question. The AI will provide educational insights (not medical advice).")

    user_input = st.text_area("Type your question:", height=100, placeholder="E.g., What are the symptoms of dehydration?")
    if st.button("Send", use_container_width=True):
        if user_input.strip():
            payload = {
                "message": user_input,
                **st.session_state.user_details
            }
            with st.spinner("Thinking..."):
                response = call_backend("chat", payload)
                if response:
                    st.success("AI Response:")
                    st.write(response.get("reply", "No response received."))
        else:
            st.warning("Please enter a question before sending.")

# ------------------------------
# 🧩 Disease Prediction Page
# ------------------------------
elif page == "🧩 Disease Prediction":
    st.header("🧩 Disease Prediction")
    st.write("Enter your symptoms to get AI-based possible conditions.")

    symptoms = st.text_area("Describe your symptoms:", height=100, placeholder="E.g., fever, fatigue, sore throat")

    if st.button("Predict Disease", use_container_width=True):
        user = st.session_state.user_details
        if symptoms.strip() and user["name"].strip():
            payload = {
                **user,
                "symptoms": symptoms
            }
            with st.spinner("Analyzing symptoms..."):
                response = call_backend("predict/disease", payload)
                if response:
                    st.success(f"Possible Conditions for {user['name']}:")
                    st.write(response.get("prediction", "No prediction returned."))
        else:
            st.warning("Please enter your name and symptoms.")

# ------------------------------
# 💊 Treatment Plan Page
# ------------------------------
elif page == "💊 Treatment Plan":
    st.header("💊 AI-Generated Treatment Plan")
    st.write("Get AI-generated care recommendations (for informational purposes only).")

    condition = st.text_input("Enter the diagnosed condition:", placeholder="E.g., Type 2 Diabetes")

    if st.button("Generate Plan", use_container_width=True):
        user = st.session_state.user_details
        if condition.strip() and user["name"].strip():
            payload = {
                **user,
                "condition": condition
            }
            with st.spinner("Generating treatment plan..."):
                response = call_backend("treatment/plan", payload)
                if response:
                    st.success(f"Suggested Plan for {user['name']}:")
                    st.write(response.get("plan", "No plan available."))
        else:
            st.warning("Please enter your name and condition.")

# ------------------------------
# 📊 Analytics Dashboard Page
# ------------------------------
elif page == "📊 Analytics Dashboard":
    st.header("📊 Health Analytics Dashboard")
    st.write("Upload your health data (CSV) for insights and visualization.")

    uploaded_file = st.file_uploader("Upload your health data (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        st.write("### Basic Data Overview")
        st.write(df.describe())

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if numeric_cols:
            feature = st.selectbox("Select a feature to visualize:", numeric_cols)
            fig, ax = plt.subplots()
            df[feature].hist(ax=ax, bins=20)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for visualization.")

# ------------------------------
# 🧩 End of App
# ------------------------------

