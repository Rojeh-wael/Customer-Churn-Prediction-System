import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .churn-high {
        background-color: #ffebee;
        border-left: 5px solid #d32f2f;
    }
    .churn-low {
        background-color: #e8f5e9;
        border-left: 5px solid #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('churn_model.h5')

model = load_model()

# Load the scaler and encoders
@st.cache_resource
def load_preprocessors():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    with open('onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)
    
    return scaler, label_encoder, onehot_encoder

scaler, label_encoder, onehot_encoder = load_preprocessors()

def predict_churn(input_data):
    """Make churn prediction"""
    geo_encoded = onehot_encoder.transform(input_data[['Geography']])
    geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(['Geography']))
    
    input_processed = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)
    input_processed['Gender'] = label_encoder.transform(input_processed['Gender'])
    
    input_scaled = scaler.transform(input_processed)
    prediction = model.predict(input_scaled, verbose=0)
    
    return prediction[0][0]

def get_risk_level(probability):
    """Categorize risk level"""
    if probability < 0.3:
        return "Low", "🟢"
    elif probability < 0.6:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

# Main app
st.title("🎯 Customer Churn Prediction System")
st.markdown("---")

# Sidebar for navigation
with st.sidebar:
    st.header("📋 Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        geography = st.selectbox("Geography", onehot_encoder.categories_[0], key="geo")
        gender = st.selectbox("Gender", label_encoder.classes_, key="gen")
        age = st.slider("Age", min_value=18, max_value=100, value=40, key="age")
    
    with col2:
        st.subheader("Financial")
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650, key="cs")
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=75000.0, key="sal")
        balance = st.number_input("Account Balance ($)", min_value=0.0, value=50000.0, key="bal")
    
    st.divider()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (years)", min_value=0, max_value=10, value=5, key="ten")
        num_of_products = st.slider("# of Products", min_value=1, max_value=4, value=2, key="prod")
    
    with col4:
        st.subheader("Service Status")
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"], key="card")
        is_active_member = st.selectbox("Active Member", ["Yes", "No"], key="active")
    
    st.divider()
    
    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    predict_button = col_btn1.button("🔍 Predict", use_container_width=True, type="primary")
    reset_button = col_btn2.button("↻ Reset", use_container_width=True)

# Main content area
if reset_button:
    st.rerun()

if predict_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [1 if has_cr_card == "Yes" else 0],
        'IsActiveMember': [1 if is_active_member == "Yes" else 0],
        'EstimatedSalary': [estimated_salary]
    })
    
    # Make prediction
    churn_probability = predict_churn(input_data)
    risk_level, emoji = get_risk_level(churn_probability)
    
    # Display results
    st.header(f"{emoji} Prediction Results")
    
    # Main metric display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Probability", f"{churn_probability*100:.1f}%", 
                 delta=f"{(churn_probability-0.5)*100:.1f}%" if churn_probability > 0.5 else None)
    
    with col2:
        st.metric("Risk Level", risk_level, delta="Higher Risk" if churn_probability > 0.5 else "Lower Risk")
    
    with col3:
        stay_probability = (1 - churn_probability) * 100
        st.metric("Customer Retention", f"{stay_probability:.1f}%")
    
    st.divider()
    
    # Prediction explanation
    if churn_probability > 0.5:
        st.warning(f"⚠️ **HIGH CHURN RISK** - This customer has a {churn_probability*100:.1f}% probability of churning. Immediate retention action recommended.", 
                  icon="⚠️")
    else:
        st.success(f"✅ **LOW CHURN RISK** - This customer has a {churn_probability*100:.1f}% probability of churning. The customer appears satisfied.", 
                  icon="✅")
    
    # Visualization
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("Churn Probability Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_probability*100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Risk (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#e8f5e9"},
                    {'range': [30, 60], 'color': "#fff3e0"},
                    {'range': [60, 100], 'color': "#ffebee"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_viz2:
        st.subheader("Risk Distribution")
        risk_data = {
            'Status': ['Will Churn', 'Will Stay'],
            'Percentage': [churn_probability*100, (1-churn_probability)*100],
            'Color': ['#d32f2f', '#388e3c']
        }
        fig = go.Figure(data=[go.Pie(
            labels=risk_data['Status'],
            values=risk_data['Percentage'],
            marker=dict(colors=risk_data['Color']),
            hole=0.3,
            textposition='inside',
            textinfo='label+percent'
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Profile Summary
    st.divider()
    st.subheader("📊 Customer Profile Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.info(f"**Location**\n{geography}")
    
    with summary_col2:
        st.info(f"**Age Group**\n{age} years old")
    
    with summary_col3:
        st.info(f"**Customer Since**\n{tenure} years")
    
    with summary_col4:
        st.info(f"**Account Status**\n{'Active' if is_active_member == 'Yes' else 'Inactive'}")
    
    # Detailed metrics
    st.subheader("💰 Financial Overview")
    
    fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
    
    with fin_col1:
        st.metric("Credit Score", f"{credit_score}", "Good" if credit_score > 700 else "Fair")
    
    with fin_col2:
        st.metric("Monthly Salary", f"${estimated_salary:,.0f}")
    
    with fin_col3:
        st.metric("Account Balance", f"${balance:,.0f}")
    
    with fin_col4:
        st.metric("Active Products", f"{num_of_products}")

else:
    st.info("👈 **Please fill in the customer details on the left sidebar and click 'Predict' to get started!")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.subheader("📌 How It Works")
        st.markdown("""
        1. **Enter Customer Data**: Fill in the customer information in the sidebar
        2. **Click Predict**: Submit the form to analyze churn risk
        3. **Review Results**: Get detailed insights on customer retention probability
        4. **Take Action**: Use insights to develop retention strategies
        """)
    
    with col_info2:
        st.subheader("🎯 Risk Factors")
        st.markdown("""
        - **Credit Score**: Financial reliability indicator
        - **Age & Tenure**: Customer loyalty patterns
        - **Account Balance**: Engagement level
        - **Product Count**: Service depth
        - **Activity Status**: Recent engagement
        """)

