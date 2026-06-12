# E-Commerce Customer Purchase Intention

## 🛍️ Project Overview
This project focuses on analyzing online shopper session behavior to predict user purchase intent. By understanding how web visitors interact with product pages, administrative features, and engagement metrics, the goal is to build a predictive classification engine that identifies high-converting traffic. This enables e-commerce platforms to optimize automated marketing strategies and maximize digital revenue.

---

## 📊 Feature Engineering & Architecture

The workflow utilizes multi-dimensional attributes mapping user behavior across three primary layers:

* **Session Activity:** Tracks the count and duration of visits to administrative, informational, and product-related pages.
* **Engagement Metrics:** Analyzes visitor interest using exit rates, bounce rates, and overall page values.
* **Temporal & Technical Context:** Accounts for external factors like weekends, operating systems, geographic regions, traffic sources, and visitor categories (e.g., returning vs. new visitors).

---

## ⚙️ Machine Learning Workflow

The pipeline is engineered systematically to go from raw data to an optimized, deployment-ready classifier:

### 1. Data Preprocessing & Encoding
* Handled missing value checks and cleaned structural anomalies.
* Transformed categorical variables into numerical indicators suitable for tree-based algorithms.
* Engineered custom behavioral metrics including page-interaction densities and total active site duration.

### 2. Model Training & Evaluation Strategy
* Splitted data into training and validation subsets to ensure model generalization.
* Addressed inherent dataset class imbalances to prioritize minority class classification capacity.
* Evaluated performance strictly via the **F1-Score** metric to balance precision and recall.

### 3. Structural Pruning (Optimization)
* **Pre-Pruning:** Controlled initial tree growth by tuning constraints such as maximum depth parameters.
* **Post-Pruning:** Implemented Cost-Complexity Pruning ($ccp\_alpha$) to systematically strip away weak leaf branches, minimizing overfitting and maximizing validation performance.

---
