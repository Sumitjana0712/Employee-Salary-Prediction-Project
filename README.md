**🚀 EMPLOYEE SALARY PREDICTION**

🎯 A machine learning web app to predict whether an employee earns >50K or ≤50K based on personal and professional attributes. Built using Streamlit, trained on the UCI Adult Income Dataset.

**📦 PROJECT OVERVIEW**

🔍 This project aims to:

Classify employee salary range (>50K or ≤50K)

Leverage preprocessing techniques (Encoding, Scaling, PCA)

Utilize a trained machine learning model

Provide an interactive web interface using Streamlit

Visualize feature importance and model performance

**🗂️ PROJECT STRUCTURE**

Employee Salary Prediction Project/	

├── 📄 app.py                         → Main Streamlit application	
 
├── 📄 knn_adult_csv_updated.ipynb    → Jupyter notebook for EDA & model building	 

├── 📦 best_model.pkl                → Trained machine learning model	 

├── 📄 adult 3.csv                   → Dataset used for training	 

└── 📄 Employee Salary Classification.pdf → Project report/presentation/Screenshot	 




**⚙️ HOW TO RUN LOCALLY**

✅ PREREQUISITES

Python 3.8 or higher

Required libraries: streamlit, pandas, numpy, joblib, plotly, scikit-learn

**📦 Install all dependencies:**

pip install -r requirements.txt



(Note: A requirements.txt file is not provided. To generate this file, run pip freeze > requirements.txt after installing all dependencies.)

▶️ TO START THE STREAMLIT APP

streamlit run app.py



Then open your browser and go to the URL displayed (usually http://localhost:8501).

**📊 MODEL INFORMATION**

Dataset: UCI Adult Income

Problem Type: Binary Classification

Target Variable: Salary (<=50K or >50K)

Preprocessing:

Label Encoding for categorical columns

StandardScaler for numerical features

PCA for dimensionality reduction (explored during development in knn_adult_csv_updated.ipynb, but not directly applied in app.py's live prediction pipeline).

Model: Gradient Boosting Classifier, saved as best_model.pkl after evaluation.

Evaluation Metrics: Accuracy, F1-score, Precision, Recall

LINK TO THE APP :- https://employee-salary-prediction-project-sumitjana.streamlit.app/
