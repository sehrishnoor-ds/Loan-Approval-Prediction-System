# Loan Approval Prediction System
An end-to-end Machine Learning project that predicts the likelihood of loan approval based on borrower profiles. This project features a comparative analysis of multiple ML models and a real-time interactive dashboard for predictions.
# 🚀 Project Overview
In the financial sector, accurately predicting loan defaults is critical for risk management. This project explores various classification algorithms to find the most robust model for credit scoring.

## Key Features:
* **Data Preprocessing:** Implemented a robust pipeline using StandardScaler for numerical scaling and OneHotEncoder for categorical variables.
* **Model Comparison:** Evaluated 4 different algorithms to find the optimal balance between accuracy and interpretability.
* **Interactive UI:** Built a Streamlit dashboard that allows users to input borrower details (Income, Credit Score, etc.) and get an instant "Approved" or "Rejected" result with a confidence score.

# Model Performance & Analysis
I compared four different machine learning models to determine which performed best on the loan dataset:
![Analysis of 4 ML models](loan model table.png)

## Why XGBoost?
While **Logistic Regression** is standard in banking for its "explainability," **XGBoost** was chosen for the final deployment. Its gradient boosting architecture and built-in regularization allowed it to capture subtle patterns (like the interaction between loan-to-income ratio and credit history) that simpler model (LR) missed.

# 🛠️ Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Scikit-Learn, XGBoost
- Deployment: Streamlit
- Serialization: Joblib

# 💻 How to Run the App
**1. Clone the repository:**
Run the following commands,
- git clone https://github.com/sehrishnoor-ds/loan-approval-predictor.git
- cd loan_approval_predictor
**2. Install dependencies:**
- pip install -r requirements.txt
**3. Run the Streamlit app:**
Open your termial of VSCode RUN(if you are working in it),
- streamlit run app.py

# 📝 Conclusion
By utilizing XGBoost, this system achieved a 93% accuracy rate, significantly outperforming the baseline linear model. This highlights the power of ensemble learning in financial risk assessment, where even a 1-2% increase in accuracy can prevent significant financial loss.









