import streamlit as st
import pandas as pd
import joblib
#loading model,scler and encoders
model = joblib.load("LogisticRegression.pkl")
le=joblib.load("Encoder.pkl")
ohe=joblib.load('OneHot.pkl') 
scaler=joblib.load('StandardScaler.pkl')

#user input

creditscore = st.number_input('Enter your Credit Score')

Gender=st.selectbox('Gender',["Male","Female"])
Age=st.number_input('Enter your Age')
Tenure=st.number_input('Enter your Tenure')
Balance=st.number_input('Enter your Balance')
NumberOfProducts=st.number_input('Enter your Number of Products')
HasCrCard=st.selectbox('Do you have a Credit Card?',["Yes","No"])
IsActiveMember=st.selectbox('Are you an Active Member?',["Yes","No"])
EstimatedSalary=st.number_input('Enter your Estimated Salary')
Geography=st.selectbox('Geography',["France","Germany","Spain"])

# Preprocess inputs
# Gender = 1 if Gender == "Male" else 0
HasCrCard = 1 if HasCrCard == "Yes" else 0
IsActiveMember = 1 if IsActiveMember == "Yes" else 0


# Encode Geography
Geography = ohe.transform([[Geography]]).toarray()
Geography=pd.DataFrame(Geography,columns=ohe.get_feature_names_out())



# Encode Gender
gender=le.transform([[Gender]])
# st.write(gender)

# # Create a DataFrame for the inputs
input_data = pd.DataFrame({
    'CreditScore': [creditscore],
    'Gender': [gender],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumberOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary],
})

data=pd.concat([input_data,Geography],axis=1)
# st.write(data)

# # Scale the input data
data_scaled = scaler.transform(data)

# # Make prediction
prediction = model.predict(data_scaled)
if st.button("Predict"):
    # # Display the result
    if prediction[0] == 1:
        st.success("The customer is likely to churn.")
    else:
        st.error("The customer is not likely to churn.")
