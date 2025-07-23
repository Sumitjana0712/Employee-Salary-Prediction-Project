import streamlit as st
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer

model_loaded = joblib.load("best_model.pkl")
model = model_loaded[0] if isinstance(model_loaded, tuple) else model_loaded

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

global_mappings = {
    "education": {"Bachelors":0, "Masters":1, "PhD":2, "HS-grad":3, "Assoc":4, "Some-college":5},
    "occupation": {"Tech-support":0, "Craft-repair":1, "Other-service":2, "Sales":3, "Exec-managerial":4, "Prof-specialty":5, "Handlers-cleaners":6, "Machine-op-inspct":7, "Adm-clerical":8, "Farming-fishing":9, "Transport-moving":10, "Priv-house-serv":11, "Protective-serv":12, "Armed-Forces":13},
    "workclass": {"Private":0, "Self-emp-not-inc":1, "Self-emp-inc":2, "Federal-gov":3, "Local-gov":4, "State-gov":5, "Without-pay":6, "Never-worked":7},
    "marital-status": {"Never-married":0, "Married-civ-spouse":1, "Divorced":2, "Separated":3, "Widowed":4, "Married-spouse-absent":5, "Married-AF-spouse":6},
    "relationship": {"Husband":0, "Not-in-family":1, "Own-child":2, "Unmarried":3, "Wife":4, "Other-relative":5},
    "race": {"White":0, "Black":1, "Asian-Pac-Islander":2, "Amer-Indian-Eskimo":3, "Other":4},
    "gender": {"Male":0, "Female":1},
    "native-country": {"United-States":0, "Mexico":1, "Philippines":2, "Germany":3, "Canada":4, "India":5, "England":6, "China":7, "Other":8}
}

def preprocess_dataframe(df):
    for column, mapping in global_mappings.items():
        if column in df.columns:
            df[column] = df[column].map(mapping)
    if 'income' in df.columns:
        df.drop(columns=['income'], inplace=True)
    if 'education' in df.columns:
        df.drop(columns=['education'], inplace=True)
    
    # Impute any remaining NaN values with median to prevent GradientBoosting errors
    imputer = SimpleImputer(strategy='median')
    df[df.columns] = imputer.fit_transform(df)
    
    return df

st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", list(global_mappings["education"].keys()))
occupation = st.sidebar.selectbox("Job Role", list(global_mappings["occupation"].keys()))
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
workclass = st.sidebar.selectbox("Workclass", list(global_mappings["workclass"].keys()))
marital_status = st.sidebar.selectbox("Marital Status", list(global_mappings["marital-status"].keys()))
relationship = st.sidebar.selectbox("Relationship", list(global_mappings["relationship"].keys()))
race = st.sidebar.selectbox("Race", list(global_mappings["race"].keys()))
gender = st.sidebar.selectbox("Gender", list(global_mappings["gender"].keys()))
native_country = st.sidebar.selectbox("Native Country", list(global_mappings["native-country"].keys()))

input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience],
    'workclass': [workclass],
    'marital-status': [marital_status],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

if st.button("Predict Salary Class"):
    try:
        processed_input = preprocess_dataframe(input_df.copy())
        prediction = model.predict(processed_input)
        st.success(f"✅ Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:", batch_data.head())
        processed_batch = preprocess_dataframe(batch_data.copy())
        st.write("Preprocessed data preview:", processed_batch.head())
        batch_preds = model.predict(processed_batch)
        processed_batch['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.write(processed_batch.head())
        csv = processed_batch.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
