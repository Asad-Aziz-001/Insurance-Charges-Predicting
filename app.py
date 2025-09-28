import streamlit as st
import joblib
import pandas as pd

# -----------------------
# Load trained pipeline
# -----------------------
model = joblib.load("insurance_model.joblib")

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="💰 Insurance Charges Predictor",
    page_icon="💡",
    layout="centered"
)

# -----------------------
# Custom CSS for modern UI
# -----------------------
st.markdown("""
    <style>
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        color: #333333;
    }
    /* Card style */
    .result-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .result-title {
        font-size: 22px;
        font-weight: bold;
        color: #0078D7;
    }
    .result-value {
        font-size: 28px;
        font-weight: bold;
        color: #28a745;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Navigation menu
# -----------------------
menu = st.sidebar.radio("📍 Navigation", ["Home", "About"])

if menu == "Home":
    st.title("💰 Insurance Charges Prediction App")
    st.markdown(
        """
        Welcome! This app predicts **medical insurance charges** based on your personal and health details.  
        Adjust the inputs on the sidebar and click **Predict** to see your estimated charges.  
        """
    )

    # Sidebar inputs
    st.sidebar.header("📌 Input Parameters")
    age = st.sidebar.slider("Age", 18, 100, 30)
    bmi = st.sidebar.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
    children = st.sidebar.slider("Number of Children", 0, 5, 0)
    sex = st.sidebar.radio("Sex", ("male", "female"))
    smoker = st.sidebar.radio("Smoker", ("yes", "no"))
    region = st.sidebar.selectbox("Region", ("southeast", "southwest", "northeast", "northwest"))

    # Prepare input
    input_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region],
        }
    )

    # Fix missing 'index' column if required
    if hasattr(model, "feature_names_in_") and "index" in model.feature_names_in_:
        input_data["index"] = 0

    # Prediction button
    if st.sidebar.button("🔮 Predict Charges"):
        prediction = model.predict(input_data)[0]
        lower, upper = prediction * 0.8, prediction * 1.2

        # Display in a styled card
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">Estimated Insurance Charges</div>
                <div class="result-value">${prediction:,.2f}</div>
                <p>Predicted Range: ${lower:,.0f} – ${upper:,.0f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Input summary
        with st.expander("📊 View Input Summary"):
            st.table(input_data.T.rename(columns={0: "Value"}))

elif menu == "About":
    # About Page
    st.title("ℹ️ About This App")
    st.markdown("""
    This **Insurance Charges Prediction App** was built with:
    - 🧠 Machine Learning (Regression Models: Linear, Random Forest, XGBoost)
    - 📊 Scikit-learn & XGBoost
    - 🎨 Streamlit for modern UI

    **How it works:**
    1. Input personal + health details  
    2. ML model predicts estimated medical charges  
    3. Result is displayed with an approximate range  

    🔮 Future Upgrades:
    - Model interpretability (SHAP values)  
    - Visual analytics dashboard  
    - Deployment with cloud APIs  
    """)

    st.markdown("---")

    # Profile Card
    st.markdown("""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:15px; text-align:center; 
                box-shadow: 0px 4px 8px rgba(0,0,0,0.1);">
        <h2 style="color:#333;">👨‍💻 Developer</h2>
        <h3 style="margin-top:10px; color:#0078D7;">ASAD AZIZ</h3>
        <p style="color:gray;">Student of BS-Artificial Intelligence | AI Enthusiast | Developer</p>
        <a href="https://github.com/Asad-Aziz-001" target="_blank" style="margin:10px; text-decoration:none;">🐙 GitHub</a> |
        <a href="https://www.linkedin.com/in/asad-aziz-140p" target="_blank" style="margin:10px; text-decoration:none;">💼 LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
