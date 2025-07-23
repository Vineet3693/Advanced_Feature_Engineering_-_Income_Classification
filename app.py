
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Income Classification Model",
    page_icon="💰",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('income_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        with open('imputer.pkl', 'rb') as f:
            imputer = pickle.load(f)
        
        return model, scaler, imputer
    except FileNotFoundError:
        st.error("Model files not found! Please run the model training script first.")
        return None, None, None

# Main app
def main():
    st.title("💰 Advanced Income Classification Model")
    st.markdown("---")
    
    # Load models
    model, scaler, imputer = load_models()
    
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("Enter Your Details")
    
    # Input fields
    age = st.sidebar.slider("Age", 18, 70, 30)
    education = st.sidebar.selectbox(
        "Education Level",
        ["High School", "Bachelor", "Master", "PhD"]
    )
    experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
    job_category = st.sidebar.selectbox(
        "Job Category",
        ["Tech", "Healthcare", "Finance", "Education", "Other"]
    )
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    
    # Feature engineering
    education_multiplier = {'High School': 0.8, 'Bachelor': 1.0, 'Master': 1.3, 'PhD': 1.6}
    education_mult = education_multiplier[education]
    
    # Prepare features
    features = np.array([[age, experience, credit_score, education_mult]])
    
    # Prediction button
    if st.sidebar.button("Predict Income Category", type="primary"):
        # Preprocess features
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Result")
            
            # Create prediction display
            if prediction == "Low":
                st.error(f"Predicted Income Category: {prediction}")
                income_range = "< $40,000"
            elif prediction == "Medium":
                st.warning(f"Predicted Income Category: {prediction}")
                income_range = "$40,000 - $70,000"
            elif prediction == "High":
                st.info(f"Predicted Income Category: {prediction}")
                income_range = "$70,000 - $100,000"
            else:
                st.success(f"Predicted Income Category: {prediction}")
                income_range = "> $100,000"
            
            st.write(f"**Estimated Income Range:** {income_range}")
        
        with col2:
            st.subheader("Prediction Confidence")
            
            # Create probability chart
            categories = model.classes_
            prob_df = pd.DataFrame({
                'Category': categories,
                'Probability': probabilities
            })
            
            fig = px.bar(
                prob_df,
                x='Category',
                y='Probability',
                title="Prediction Probabilities",
                color='Probability',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        st.subheader("Your Profile Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Age", age)
            st.metric("Experience", f"{experience} years")
        
        with col2:
            st.metric("Education Multiplier", f"{education_mult}x")
            st.metric("Credit Score", credit_score)
        
        with col3:
            st.metric("Job Category", job_category)
            tech_bonus = "Yes" if job_category == "Tech" else "No"
            st.metric("Tech Bonus", tech_bonus)
    
    # Model information
    st.markdown("---")
    st.subheader("About This Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **Features Used:**
        - Age
        - Years of Experience
        - Education Level (with multipliers)
        - Credit Score
        - Job Category (with tech bonus)
        """)
    
    with col2:
        st.write("""
        **Model Details:**
        - Algorithm: Random Forest Classifier
        - Income Categories: Low, Medium, High, Very High
        - Feature Engineering: Education multipliers, Tech bonuses
        - Preprocessing: Imputation, Standardization
        """)

if __name__ == "__main__":
    main()
