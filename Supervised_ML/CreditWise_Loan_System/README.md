# Project: CreditWise Loan Approval System

## 🏦 Business Context & Problem Statement
**SecureTrust Bank**, a mid-sized financial institution in India, processes hundreds of personal and home loan applications daily across urban and rural sectors. The legacy manual underwriting process conducted by loan officers is inherently time-consuming, prone to human bias, and inconsistent. This operational bottleneck creates two major financial risks:
1. **Type I Error (False Negatives):** Creditworthy applicants are rejected, resulting in a direct loss of business and customer dissatisfaction.
2. **Type II Error (False Positives):** High-risk profiles are approved, leading to loan defaults and capital losses.

## 🎯 Project Objective
The goal of this project is to engineer an intelligent automated loan risk-assessment engine using a **Decision Tree Classifier**. The model will analyze complex historical applicant profiles to predict loan eligibility ($1 = \text{Approved}$, $0 = \text{Rejected}$) instantly, mitigating financial risk while preserving model interpretability for institutional compliance.

## 📊 Feature Architecture
The historical dataset contains comprehensive multi-dimensional attributes mapping applicant risk:

* **Demographics:** `Age`, `Gender`, `Marital_Status`, `Dependents`, `Education_Level`
* **Financial Profile:** `Applicant_Income`, `Coapplicant_Income`, `Savings`, `Employer_Category`, `Employment_Status`
* **Credit Metrics:** `Credit_Score`, `Existing_Loans`, `DTI_Ratio` (Debt-to-Income)
* **Loan Specifics:** `Loan_Amount`, `Loan_Term`, `Loan_Purpose`, `Collateral_Value`, `Property_Area`
* **Target Variable:** `Loan_Approved` (Binary classification)
