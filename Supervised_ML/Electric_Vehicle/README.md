# EV Transition Analysis: Predicting Consumer Adoption Likelihood

This repository contains a machine learning pipeline that explores the factors influencing electric vehicle (EV) adoption, charging behavior, and consumer psychographics. Using a synthetic dataset of 50,000 consumer records, the project frames EV adoption prediction as a multi-class classification problem. 

The primary objective is to evaluate how well Support Vector Machine (SVM) Classifiers handle mixed numerical and categorical features to predict whether a consumer has a **High**, **Medium**, or **Low** likelihood of adopting an EV.

## 📊 Dataset Overview
The dataset contains 50,000 synthetic records with 23 initial features mapping consumer demographics, driving habits, infrastructure availability, and psychological thresholds:
* **Demographics:** Age, Annual Income, Education Level, City Type.
* **Driving Profile:** Daily Commute (km), Weekly Travel Distance (km), Current Vehicle Type, Vehicle Age, Monthly Fuel Expense.
* **Infrastructure & Costs:** Charging Station Accessibility, Nearest Station Distance (km), Home Charging Availability, Electricity Cost per kWh.
* **Psychographics:** Environmental Awareness, Government Incentive Awareness, Technology Affinity, Range Anxiety, Battery Replacement Concern, EV Knowledge.
* **Target Variable:** `ev_adoption_likelihood` (Categorical: High / Medium / Low).

---

## 🛠️ Data Preprocessing & Feature Engineering

To guarantee mathematical integrity and optimal performance for distance-based classification, the pipeline implements a rigorous cleanup and encoding flow:

1. **Missing Data Imputation:**
   * Missing numerical metrics (`charging_station_accessibility`, `ev_knowledge_score`) are filled using **Median Imputation** to prevent outlier distortion.
   * Missing categorical strings (`education_level`) are filled using **Most Frequent (Mode) Imputation**.

2. **Mitigating Target Leakage:**
   * Post-adoption behavioral features—specifically `monthly_energy_consumption_kwh` and `monthly_charging_cost`—are dropped. These columns introduce target leakage because their values only exist *after* a consumer has chosen to adopt an EV.
   * `education_level` was dropped during feature matrix separation to streamline structural complexity.

3. **Categorical Feature Encoding:**
   * **Target Variable:** Encoded using `LabelEncoder` to convert categorical text boundaries into discrete integers (`0, 1, 2`).
   * **Nominal Feature Columns:** Non-ordinal categorical values (`city_type`, `current_vehicle_type`) are transformed using **One-Hot Encoding** (`pd.get_dummies`). The parameter `drop_first=True` is utilized to avoid multi-collinearity (the dummy variable trap).

4. **Feature Scaling:**
   * Because SVMs depend heavily on geometric distances between data vectors, all features are uniformly scaled using `StandardScaler` to achieve a mean of 0 and a variance of 1.

---

## 🏎️ Model Architecture & Evaluation
The project systematically evaluates a **Support Vector Classifier (SVC)** across three distinct mathematical kernel architectures using a 60/40 Train/Test split:

### Performance Matrix

| SVM Kernel Method | Training Accuracy | Testing Accuracy | Insights |
| :--- | :---: | :---: | :--- |
| **RBF (Default)** | 89.61% | 86.73% | Strong initial boundary mapping; minor variance gap. |
| **Linear** | 87.57% | **87.16%** | **Top Performer.** Highest generalizability and structural stability. |
| **Polynomial** | 89.15% | 86.37% | Higher complexity boundary mapping with slightly lower test fit. |

### Key Takeaway
The **Linear Kernel** stands out as the optimal model choice. While non-linear alternatives like RBF score slightly higher during training iterations, the Linear kernel generalizes best onto unseen validation slices, yielding a robust test accuracy of **87.16%**. This implies the hyperplanes separating adoption tiers scale cleanly along continuous lifestyle score metrics.
