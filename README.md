# Airline Passenger Satisfaction Analysis

This project analyzes airline passenger satisfaction data using Exploratory Data Analysis (EDA), Data Preparataion, and four supervised machine learning models with visualizations of key metrics. The goal is to identify the key drivers of satisfaction and accurately classify passengers as satisfied or dissatisfied using a balanced and well-prepared dataset.

---

## 📊 Tools & Libraries Used

- Python (Jupyter Notebook)
- Pandas & NumPy – Data manipulation
- Seaborn & Matplotlib – Visualizations
- Scikit-learn – ML modeling & evaluation
- imbalanced-learn – SMOTE (Synthetic Minority Over-sampling)

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights uncovered during EDA:

- **Dataset Shape**: 100,000 rows and 25 columns
- **Distribution of Satisfaction**: Most passengers were either neutral or dissatisfied, with a class imbalance present with satifised passengers.
- **Missing & Duplicate Values**: Some features had null values and duplicate rows, which were addressed in preprocessing
- **Features Dropped**: `trip_id` and `passenger_id` were removed as they are identifiers and irrelevant for prediction

---

## 🧹 Data Preparation

- **Dropped duplicate records** to ensure data quality
- **Imputed missing values** using `SimpleImputer`:
  - Used mean or mode depending on feature type
- Converted the target variable `satisfaction` into binary using **Label Encoding**:
  - `Satisfied` → 1  
  - `Neutral or Dissatisfied` → 0
- Applied **Ordinal Encoding** to ordered service rating features:
- Applied **Dummy Encoding** (One-Hot Encoding) to nominal features:
- Scaled numerical features using **StandardScaler**:
- Balanced the training data using **SMOTE** to address class imbalance
- Split the dataset into **training (75%)** and **testing (25%)** using stratified sampling

---

## 🧠 Machine Learning Models Used

Four classification models were trained:

### 1. Logistic Regression
- Interpretable linear model
- Used as a baseline

### 2. K-Nearest Neighbors (KNN)
- Instance-based classifier
- Sensitive to feature scaling and the value of `k`

### 3. Random Forest Classifier
- Tree-based ensemble method
- Handles nonlinear relationships and provides feature importance

### 4. Linear Support Vector Classifier (LinearSVC)
- Finds an optimal separating hyperplane
- Effective for high-dimensional data

All models were trained on the **SMOTE-balanced training set (75%)** and evaluated on the test set (25%).

---

## 📈 Evaluation Metrics

All models were evaluated using:

- ✅ **Accuracy**
- ✅ **Precision, Recall, and F1-Score**
- ✅ **Confusion Matrix**
- ✅ **ROC Curve & AUC Score**

The **ROC curve** helped visualize the trade-off between the **true positive rate** (sensitivity) and **false positive rate** across various threshold settings.  
It enabled direct comparison of classifier performance and highlighted the models' ability to distinguish between satisfied and dissatisfied passengers.  
A higher **AUC score** indicated stronger predictive performance, with values closer to 1 showing better class separation.

---

## 📌 Key Features

- Clean and insightful **Exploratory Data Analysis (EDA)** to uncover trends in satisfaction by demographics, travel type, and service ratings
- Encoding strategies tailored to feature type:
  - **Ordinal Encoding** for ordered service ratings
  - **Dummy Encoding** (One-Hot) for nominal features 
- Identifier features (`passenger_id` and `trip_id`) were dropped as they do not contribute to prediction
- Handled missing values using **SimpleImputer**, rather than dropping them outright
- Balanced the training data using **SMOTE**, improving classification performance on the minority class
- Built and evaluated **four classification models**: Logistic Regression, KNN, Random Forest, and LinearSVC
- Performance measured using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC
- ROC curves were used to visualize model performance across thresholds, highlighting the trade-off between sensitivity and specificity