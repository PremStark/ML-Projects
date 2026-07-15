# Stroke Prediction using Random Forest

This repository contains a Jupyter Notebook designed to predict the likelihood of a patient having a stroke based on several clinical and demographic features. The project cleans the dataset, handles missing data, encodes categorical variables, and trains a Random Forest Classifier.

---

## 📌 Project Overview

Stroke is one of the leading causes of death and disability globally. Early identification of high-risk individuals can lead to timely medical interventions. This project utilizes machine learning to predict stroke occurrence using medical data.

* **Dataset:** Stroke Prediction Dataset (Kaggle)
* **Classifier:** Random Forest Classifier
* **Performance:** **~94.19% Test Accuracy**

---

## 📂 Dataset Details

### Features Used:

* **Demographics:** `gender`, `age`, `ever_married`, `work_type`, `Residence_type`
* **Clinical Factors:** `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `smoking_status`
* **Target:** `stroke` (1 if the patient had a stroke, 0 otherwise)

---

## ⚙️ Data Preprocessing Pipeline

The notebook performs the following preprocessing steps:

1. **Missing Value Imputation:** Missing values in the `bmi` column (201 records) are imputed using the mean strategy via scikit-learn's `SimpleImputer`.
2. **Categorical Encoding:** One-hot encoding is applied to categorical columns (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) with `drop_first=True` to avoid the dummy variable trap.
3. **Data Splitting:** The processed data is split into **70% training** and **30% testing** sets.

---

## 🤖 Model & Training

A **Random Forest Classifier** was trained using the following hyperparameters:

* `n_estimators = 1000`
* `max_depth = 4`
* `random_state = 42`

---

## 📊 Evaluation Results

| Dataset Split | Accuracy Score |
| --- | --- |
| **Train Set** | 95.53% |
| **Test Set** | 94.19% |

---
