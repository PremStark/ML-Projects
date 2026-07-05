# NovaGen Research Labs — Predictive Health Modeling

An end-to-end machine learning pipeline exploring advanced ensemble techniques to predict health outcomes based on clinical, lifestyle, and demographic metrics. This repository demonstrates the practical implementation and comparative analysis of **Heterogeneous** (Voting, Stacking) and **Homogeneous** (Random Forest, Boosting) ensemble architectures.

## 📊 Dataset Overview
The dataset (`novagen_dataset.csv`) consists of **9,549 records** and **22 features** detailing patient clinical metrics and lifestyle profiles:
* **Continuous Features:** Age, BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate, Sleep Hours, Exercise Hours, Water Intake, Stress Level.
* **Categorical & Encoded Features:** Smoking, Alcohol, Diet, Mental Health, Physical Activity, Medical History, Allergies.
* **One-Hot Encoded Structural Features:** Diet Types (Vegan, Vegetarian) and Blood Groups (AB, B, O).

## ⚙️ Pipeline Architecture

1.  **Data Preprocessing & Encoding:** Explicit boolean-to-integer mapping for categorical indicators to prevent model parse exceptions.
2.  **Strategic Data Splitting:** The dataset is partitioned into a 70/30 Train/Test split utilizing a fixed random state to ensure strict reproducibility.
3.  **Data Scaling:** A `StandardScaler` is applied to eliminate feature magnitude bias. To prevent **data leakage**, the scaler is strictly fitted on the training set (`fit_transform`) and subsequently applied to the test set (`transform`).
4.  **Ensemble Modeling:** Deployment of diverse ensemble paradigms to evaluate prediction boundary strengths.

---

## 🏆 Model Evaluation & Performance Summary

The models were evaluated using classification accuracy across both training and testing splits to monitor variance (overfitting) and generalization:

| Ensemble Type | Model / Method | Architecture / Configurations | Train Accuracy | Test Accuracy | Generalization Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Heterogeneous** | **Voting Classifier** | Hard voting ensemble combining Logistic Regression, SVC, Decision Tree ($d=4$), and KNN. | 92.01% | 89.42% | Stable baseline prediction. |
| **Heterogeneous** | **Stacking Classifier** | Base: LR, SVC, DTC, KNN.<br>Meta-Learner: KNN ($k=5$) with 5-Fold Cross-Validation. | 95.23% | 92.77% | Strong meta-feature blend. |
| **Homogeneous** | **Random Forest** | Bagging architecture utilizing 200 estimators with a regularized `max_depth=6`. | 90.28% | 89.84% | **Extremely Robust**; lowest variance gap. |
| **Homogeneous** | **Gradient Boosting** | Sequential Boosting using 250 trees, `max_depth=4`, and an 80% feature/row subsample. | **97.85%** | **94.66%** | 🥇 **Best Performer** overall. |
| **Homogeneous** | **AdaBoost** | Sequential Boosting using 500 estimators backed by a regularized `DecisionTreeClassifier` ($max\_depth=2$). | 93.58% | 93.40% | High structural generalization. |

---

## 🔍 Key Insights & Technical Takeaways

* **Gradient Boosting Dominance:** The `GradientBoostingClassifier` achieved the highest test accuracy (**94.66%**). Building trees sequentially to minimize the loss gradient allowed the model to map intricate, non-linear interactions among clinical attributes (e.g., Blood Pressure, Glucose, and Stress Levels) that linear variations missed.
* **AdaBoost Regularization:** Transitioning from a shallow decision stump ($max\_depth=1$) to a depth of $2$ for the base estimators dramatically improved AdaBoost's capacity to process the 22-feature space, capturing baseline feature interactions cleanly and yielding an exceptional **93.40%** test accuracy with zero overfitting.
* **Meta-Learner Trade-offs:** The Stacking Classifier performs excellently with a KNN meta-estimator. However, moving forward, exploring a smoother linear meta-learner (like `LogisticRegression`) could provide cleaner regularization across the base models' boundary intersections.

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.12+
* **Libraries:** `pandas`, `numpy`, `scikit-learn`
