import streamlit as st
from sqlalchemy import text

def dbinsertion(created_at, name, email, existing_checking_account_status, duration, credit_history, purpose, credit_amount, savings_account_bonds, present_employment_status, installment_rates, personal_status_sex, other_debtors, present_residence_since, property_status, age, other_installment_plans, housing, bank_existing_credits, job, total_reliable_people, telephone, foreign_worker, prediction):
    
    cxdb = st.connection('mysql', type='sql')    

    sql_query = text("""INSERT INTO new_user_table VALUES 
                        (:created_at, :name, :email, :existing_checking_account_status, :duration, :credit_history,
                        :purpose, :credit_amount, :savings_account_bonds, :present_employment_status, :installment_rates,
                        :personal_status_sex, :other_debtors, :present_residence_since, :property_status, :age, :other_installment_plans,
                        :housing, :bank_existing_credits, :job, :total_reliable_people, :telephone, :foreign_worker, :prediction)
                        """)

    with cxdb.session as s:
        s.execute(sql_query,
                    {"created_at": created_at.strftime("%Y-%m-%d"), "name": name,
                    "email": email, "existing_checking_account_status": existing_checking_account_status,
                    "duration": duration, "credit_history": credit_history, "purpose": purpose,
                    "credit_amount": credit_amount, "savings_account_bonds": savings_account_bonds,
                    "present_employment_status": present_employment_status, "installment_rates": installment_rates,
                    "personal_status_sex": personal_status_sex, "other_debtors": other_debtors, "present_residence_since": present_residence_since,
                    "property_status": property_status, "age": age, "other_installment_plans": other_installment_plans,
                    "housing": housing, "bank_existing_credits": bank_existing_credits, "job": job, "total_reliable_people": total_reliable_people,
                    "telephone": telephone, "foreign_worker": foreign_worker, "prediction": int(prediction[0])})
        s.commit()