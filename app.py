"""
Streamlit Web Interface for Text Classification
"""

import streamlit as st
import joblib
import os
import pandas as pd
from preprocessing import TextPreprocessor, FeatureExtractor

# Page configuration
st.set_page_config(
    page_title="Text Classification System",
    layout="wide"
)

# Title
st.title("Text Classification System")
st.markdown("---")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["Naive Bayes", "Logistic Regression", "SVM"]
)

# Load models
@st.cache_resource
def load_models():
    """Load all models and preprocessors"""
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        st.error("Models not found! Please train the models first using train_models.py")
        return None, None, None
    
    try:
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        feature_extractor = joblib.load(os.path.join(models_dir, 'feature_extractor.joblib'))
        
        models = {}
        model_files = {
            'Naive Bayes': 'naive_bayes.joblib',
            'Logistic Regression': 'logistic_regression.joblib',
            'SVM': 'svm.joblib'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
        
        return models, preprocessor, feature_extractor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Load models
models, preprocessor, feature_extractor = load_models()

if models is not None:
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text to Classify")
        user_input = st.text_area(
            "Text Input:",
            height=200,
            placeholder="Enter your text here..."
        )
        
        if st.button("Classify Text", type="primary"):
            if user_input.strip():
                # Preprocess text
                preprocessed = preprocessor.preprocess_text(user_input)
                
                # Extract features
                features = feature_extractor.transform([preprocessed])
                
                # Get predictions from all models
                predictions = {}
                probabilities = {}
                
                for model_name, model in models.items():
                    pred = model.predict(features)[0]
                    proba = model.predict_proba(features)[0]
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = dict(zip(model.classes_, proba))
                
                # Store in session state
                st.session_state['predictions'] = predictions
                st.session_state['probabilities'] = probabilities
                st.session_state['input_text'] = user_input
            else:
                st.warning("Please enter some text to classify.")
    
    with col2:
        st.subheader("Model Information")
        st.info(f"**Selected Model:** {model_choice}")
        st.info(f"**Available Models:** {len(models)}")
        
        if st.session_state.get('predictions'):
            st.markdown("---")
            st.subheader("Prediction")
            
            selected_pred = st.session_state['predictions'][model_choice]
            selected_proba = st.session_state['probabilities'][model_choice]
            
            st.success(f"**Predicted Class:** {selected_pred}")
            st.metric("Confidence", f"{max(selected_proba.values()):.2%}")
            
            # Show probability distribution
            st.markdown("### Probability Distribution")
            proba_df = pd.DataFrame({
                'Class': list(selected_proba.keys()),
                'Probability': list(selected_proba.values())
            }).sort_values('Probability', ascending=False)
            
            st.bar_chart(proba_df.set_index('Class'))
    
    # Show all model predictions
    if st.session_state.get('predictions'):
        st.markdown("---")
        st.subheader("All Model Predictions")
        
        comparison_data = []
        for model_name, pred in st.session_state['predictions'].items():
            proba = st.session_state['probabilities'][model_name]
            max_prob = max(proba.values())
            comparison_data.append({
                'Model': model_name,
                'Prediction': pred,
                'Confidence': f"{max_prob:.2%}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
    
    # Model comparison section
    st.markdown("---")
    st.subheader("Model Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Naive Bayes", "✓ Available" if "Naive Bayes" in models else "✗ Not Available")
    with col2:
        st.metric("Logistic Regression", "✓ Available" if "Logistic Regression" in models else "✗ Not Available")
    with col3:
        st.metric("SVM", "✓ Available" if "SVM" in models else "✗ Not Available")
    
    # Instructions
    with st.expander("How to Use"):
        st.markdown("""
        1. Enter text in the text area
        2. Click "Classify Text" button
        3. View predictions from all models
        4. Select different models from the sidebar to see their predictions
        5. Check the probability distribution for confidence scores
        """)
    
    # Model explanations
    with st.expander("Model Explanations"):
        st.markdown("""
        **Naive Bayes:**
        - Probabilistic classifier based on Bayes' theorem
        - Assumes feature independence (naive assumption)
        - Fast training and prediction
        - Works well with high-dimensional text data
        
        **Logistic Regression:**
        - Linear classifier using logistic function
        - Provides probability estimates
        - Interpretable coefficients
        - Good baseline for text classification
        
        **SVM (Support Vector Machine):**
        - Finds optimal decision boundary
        - Effective with high-dimensional data
        - Can handle non-linear relationships with kernels
        - Generally good performance on text data
        """)

else:
    st.error("""
    ## Models Not Found
    
    Please train the models first by running:
    ```bash
    python train_models.py
    ```
    
    Make sure you have:
    1. Created a `data` directory
    2. Added your dataset as `data/sentiment_data.csv`
    3. The CSV should have 'text' and 'label' columns
    """)
