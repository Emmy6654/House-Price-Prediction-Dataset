
import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# LOAD MODEL

model = joblib.load('house_price_model.pkl') # Model saved with this name previously

# CUSTOM CSS

st.markdown('''
<style>

.st-emotion-cache-zt5ig8 { # Main app container
    padding-top: 35px;
}

.st-emotion-cache-z5rd5k { # Sidebar container
    padding-top: 30px;
}

.st-emotion-cache-vk337f { # block container
    padding: 0px;
}

</style>
''', unsafe_allow_html=True)

#SIDEBAR

with st.sidebar:
     st.image(
        "https://cdn-icons-png.flaticon.com/512/2940/2940798.png", # House icon
        width=150
    )    

    st.title("🧮 Project Overview")

    st.markdown('''
    ### Models Used

    - Linear Regression
    - Random Forest
    - XGBoost

    ### Objective

    Predict house prices based on various property features.

    ### Features

    - Real-time prediction
    - ML-powered decision making
    ''')

    st.divider()

    st.info(
        "Built with Streamlit, Scikit-Learn and XGBoost."
    )

# HEADER

st.markdown(
    "🏠 House Price Prediction System",
    unsafe_allow_html=True

)

st.divider()

# INPUT SECTION

col1, col2, col3 = st.columns(3)

# Define the features that were used for training the model
# These should match the X.columns from the notebook after preprocessing
feature_names = [
    'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt',
    'Location_Suburban', 'Location_Urban', 'Condition_Fair',
    'Condition_Good', 'Condition_Poor', 'Garage_Yes'
]

input_values = {}

with col1:
    st.subheader("Property Details")
    input_values['Area'] = st.number_input("Area (sq ft)", value=2500, min_value=500, max_value=5000)
    input_values['Bedrooms'] = st.number_input("Bedrooms", value=3, min_value=1, max_value=5)
    input_values['Bathrooms'] = st.number_input("Bathrooms", value=2, min_value=1, max_value=4)
    input_values['Floors'] = st.number_input("Floors", value=2, min_value=1, max_value=3)
    input_values['YearBuilt'] = st.number_input("Year Built", value=1990, min_value=1900, max_value=2023)

with col2:
    st.subheader("Location and Condition")
    location = st.selectbox("Location", ['Downtown', 'Suburban', 'Rural', 'Urban'])
    condition = st.selectbox("Condition", ['Excellent', 'Good', 'Fair', 'Poor'])
    garage = st.selectbox("Garage", ['Yes', 'No'])

    # Initialize one-hot encoded columns to False
    input_values['Location_Suburban'] = False
    input_values['Location_Urban'] = False
    input_values['Condition_Fair'] = False
    input_values['Condition_Good'] = False
    input_values['Condition_Poor'] = False
    input_values['Garage_Yes'] = False

    # Set True based on user selection
    if location == 'Suburban':
        input_values['Location_Suburban'] = True
    elif location == 'Urban':
        input_values['Location_Urban'] = True

    if condition == 'Fair':
        input_values['Condition_Fair'] = True
    elif condition == 'Good':
        input_values['Condition_Good'] = True
    elif condition == 'Poor':
        input_values['Condition_Poor'] = True

    if garage == 'Yes':
        input_values['Garage_Yes'] = True

# PREDICTION

if st.button("🔍 Predict House Price"):
    # Create a DataFrame from the input dictionary
    input_df = pd.DataFrame([input_values])

    # Match model features dynamically and prevent feature name validation errors
    if hasattr(model, "feature_names_in_"):
        final_input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0).astype(float)
    else:
        final_input_df = input_df[feature_names].astype(float)

    prediction = model.predict(final_input_df.values)[0]

    st.divider()

    st.subheader("Prediction Results")

    st.success(
        f"The predicted house price is: **${prediction:,.2f}**"
    )


# FOOTER

st.divider()

st.caption(
    "House Price Prediction Dashboard | Machine Learning Project"
)
