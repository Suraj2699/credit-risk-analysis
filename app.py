import streamlit as st
import numpy as np
from scipy.special import boxcox1p
import concurrent.futures
import datetime
from src.model_utils import load_models, load_lambda_values, run_prediction
from src.db_utils import dbinsertion
import warnings
warnings.filterwarnings('ignore')


models = load_models()
lambda_values = load_lambda_values()

st.title("Credit Risk Analyzer")
st.write("Enter Your Specific Credit Information Below:")

## USER INPUT STARTS HERE:

name = st.text_input("Enter Your Name: ", '')
email = st.text_input("Enter Your Email: ", '')

def categorical_mapping_logic(label, dictionary):
    choice = st.selectbox(label, list(dictionary.keys()))
    return dictionary[choice]

## Comparing Deutsche Marks (DEM) to INR (₹)

existing_checking_account_status = categorical_mapping_logic("Existing Checking Account Status", {
    '< ₹0': 0, '< ₹10000': 1, '>= ₹10000': 2, 'No Checking Account': 3
})

duration = st.number_input("Duration in Months", min_value=0, step=1)

credit_history = categorical_mapping_logic('Credit History', {
    'No Credits Taken/All Credits Paid Back Duly': 0,
    'All Credits at this Bank Paid Back Duly': 1,
    'Existing Credits Paid Back Duly Till Now': 2,
    'Delay in Paying off in the Past': 3,
    'Critical Account/Other Credits Existing (Not at this Bank)': 4,
})

purpose = categorical_mapping_logic("Purpose", {
    'Car (New)': 0, 'Car (Used)': 1, 'Furniture/Equipment': 2, 'Radio/Television': 3, 'Domestic Appliances': 4,
    'Repairs': 5, 'Education': 6, 'Retraining': 7, 'Business': 8, 'Others': 9
})

credit_amount = st.number_input("Credit Amount", min_value=0, step=1)
credit_amount = credit_amount//50

savings_account_bonds = categorical_mapping_logic("Saving Account/Bonds", {
    '< ₹5000': 0, '₹5000 <= X <₹25000': 1, '₹25000 <= X < ₹50000': 2,
    '>= ₹50000': 3, 'Unknown/No Savings Account': 4
})

present_employment_status = categorical_mapping_logic("Present Employment Since", {
    'unemployed': 0, '< 1 Year': 1, '1 <= X < 4 Years': 2,
    '4 <= X < 7 Years': 3, '>= 7 Years': 4
})

installment_rates = st.selectbox("Installment Rates (in %) of Disposable Income", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'More than 10'))
if installment_rates == 'More than 10':
    installment_rates= st.number_input("Enter the Installment Rate", min_value=11, step=1)

## There are no single female in the dataset, but as this option is at the end
## of the list, we can include it. It would not break the label encoding chain.

personal_status_sex = categorical_mapping_logic("Personal Status and Sex", {
    'Male: Divorced/Separated': 0, 'Female: Divorced/Separated/Married': 1,
    'Male: Single': 2, 'Male: Married/Widowed': 3, 'Female: Single': 4
})

other_debtors = categorical_mapping_logic("Are There Other Debtors/Guarantors", {
    'None': 0,
    'Co-Applicant': 1,
    'Guarantor': 2
})

present_residence_since = st.number_input("Present Residence Since (in Years)", min_value=0, step=1)
if present_residence_since > 4:
    present_residence_since = 4

property_status = categorical_mapping_logic("Property", {
    'Real Estate': 0,'If Not Real Estate: Building Society Savings Agreement/ Life Insurance': 1,
    'If Not the Above Options: Car or Other': 2, 'Unknown/No Property': 3
})

age = st.number_input("Enter Your Age:", min_value=0, max_value=100, step=1)

other_installment_plans = categorical_mapping_logic("Do You Have Other Installment Plans", {
    'Bank': 0, 'Stores': 1, 'None': 2
})

housing = categorical_mapping_logic("Current Living Status", {
    'Rent': 0, 'Own': 1, 'For Free': 2
})

bank_existing_credits = st.selectbox("Number of Existing Credits at This Bank", (1, 2, 3, 4, '> 4'))
if bank_existing_credits == '> 4':
    bank_existing_credits = 4

job = categorical_mapping_logic("Select Your Job Status", {
    'Unemployed/Unskilled-Non-Resident':0, 'Unskilled-Resident':1,
    'Skilled Employee/Official': 2, 'Management/Self-Employed/Highly Qualified Employee/ Officer': 3
})

total_reliable_people = st.number_input("Enter the Total Number of People Being Liable to Provide Maintenance for", min_value=0, step=1)
if total_reliable_people > 2:
    total_reliable_people = 2

telephone = categorical_mapping_logic("Telephone", {
    'None': 0, 'Yes, Registered Under the Customers Name': 1
})

foreign_worker = categorical_mapping_logic("Are You a Foreign Worker?", {
    'Yes': 0, 'No': 1
})


if present_employment_status == 0:
    employment_adjusted = present_employment_status + 1
else:
    employment_adjusted = present_employment_status

def handled_boxcox1p(x, lambda_value):
    if x <= 0:
        x = 1e-5
    return boxcox1p(x, lambda_value)

transformed_duration = handled_boxcox1p(duration - 1, lambda_values['lambda2'])
transformed_credit_amount = handled_boxcox1p(credit_amount - 1, lambda_values['lambda1'])
transformed_age = handled_boxcox1p(age - 1, lambda_values['lambda3'])

debt_to_income_ratio = credit_amount/(employment_adjusted*1000)
transformed_debt_to_income_ratio = handled_boxcox1p(debt_to_income_ratio - 1, lambda_values['lambda4'])

credit_utilization = bank_existing_credits/(savings_account_bonds + 1)
transformed_credit_utilization = handled_boxcox1p(credit_utilization - 1, lambda_values['lambda5'])

linearfeatures = np.array([[existing_checking_account_status, transformed_duration, credit_history, savings_account_bonds, present_employment_status, installment_rates, personal_status_sex, other_debtors, property_status, transformed_age, other_installment_plans, housing, bank_existing_credits, total_reliable_people, telephone, foreign_worker, transformed_debt_to_income_ratio, transformed_credit_utilization]])
treefeatures = np.array([[existing_checking_account_status, transformed_duration, credit_history, purpose, transformed_credit_amount, savings_account_bonds, present_employment_status, installment_rates, personal_status_sex, other_debtors, present_residence_since, property_status, transformed_age, other_installment_plans, housing, job, telephone, transformed_debt_to_income_ratio, transformed_credit_utilization]])
originalfeatures = np.array([[existing_checking_account_status, transformed_duration, credit_history, purpose, transformed_credit_amount, savings_account_bonds, present_employment_status, installment_rates, personal_status_sex, other_debtors, present_residence_since, property_status, transformed_age, other_installment_plans, housing, bank_existing_credits, job, total_reliable_people, telephone, foreign_worker]])

model_mapping = {
    'Logistic Regression': (models['logisticRegressor'], linearfeatures),
    'Gaussian Naive Bayes': (models['gaussianNB'], originalfeatures),
    'Decision Tree': (models['decisionTree'], treefeatures),
    'Random Forest': (models['randomForest'], treefeatures),
    'XGBoost Classifier': (models['xgboost'], originalfeatures),
    'LightGBM Classifier': (models['lightgbm'], originalfeatures),
    'Catboost Classifier': (models['catboost'], originalfeatures)
}

model = st.sidebar.selectbox(
    "Select the Model You Wish to Use",
    ('Logistic Regression', 'Gaussian Naive Bayes', 'Decision Tree', 'Random Forest', 'XGBoost Classifier', 'LightGBM Classifier', 'CatBoost Classifier')
)

selected_model, selected_features = model_mapping[model]

## Output Logic

def show_result(prediction):
    if prediction == 0:
        st.success("Thank You for Submitting This Application! Our Team Will Reach Out to You for Further Proceedings.")
    else:
        st.error("Sorry, You Are Not Eligible for the Loan. Try Again After a Few Months!")

created_at = datetime.date.today()

if st.button("Check Loan Eligibility"):
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_prediction, selected_model, selected_features)
            prediction = future.result()
            show_result(prediction)
        
        #user_info = (created_at, name, email, existing_checking_account_status, duration, credit_history, purpose, credit_amount, savings_account_bonds, present_employment_status, installment_rates, personal_status_sex, other_debtors, present_residence_since, property_status, age, other_installment_plans, housing, bank_existing_credits, job, total_reliable_people, telephone, foreign_worker, prediction)

        dbinsertion(created_at, name, email, existing_checking_account_status, duration, credit_history, purpose, credit_amount, savings_account_bonds, present_employment_status, installment_rates, personal_status_sex, other_debtors, present_residence_since, property_status, age, other_installment_plans, housing, bank_existing_credits, job, total_reliable_people, telephone, foreign_worker, prediction)

    except Exception as e:
        st.error(f"Prediction Error: {e}")