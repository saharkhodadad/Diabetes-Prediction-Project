# Diabetes Prediction Project

This project develops a complete machine learning pipeline for predicting diabetes based on medical attributes from the **Pima Indians Diabetes Dataset**. The workflow includes data loading, exploration, visualization, preprocessing, feature selection, model training, evaluation, and saving the best-performing model.

---

## 📁 Project Structure

```
├── data/
│   └── diabetes.csv               # Dataset
├── model/
│   └── diabetes_lr_model.pkl      # Saved best model (Logistic Regression)
├── Diabetes_Prediction.ipynb      # Full Jupyter implementation
└── README.md                      # Project documentation
```

---

## 🚀 1. Importing Libraries

The project uses:

* **Pandas, NumPy** for data handling
* **Matplotlib, Seaborn** for visualization
* **Scikit-learn** for preprocessing, model training & evaluation
* **XGBoost** for advanced boosting-based classification
* **Joblib** for saving/loading trained models

Decimal formatting is applied to improve the readability of statistical outputs.

---

## 📊 2. Basic Data Exploration

### 2.1 Loading the Dataset

The dataset is loaded from the `data/` directory:

```python
df = pd.read_csv('./data/diabetes.csv')
```

(Alternatively downloaded via **kagglehub**.)

### 2.2 Descriptive Analysis

The following exploratory steps are performed:

* Display a random sample of the dataset
* Count rows and columns
* Inspect data types and structure using `df.info()`
* Check missing values, duplicates
* Generate summary statistics with `df.describe()`

### 2.3 Data Visualization

Visual analysis includes:

* Outcome distribution plot
* Individual numerical feature distributions
* Boxplots showing relationships between predictors and outcome
* Feature correlation heatmap
* Standard deviation analysis

These help detect trends, correlations, and data quality issues.

---

## 🛠️ 3. Data Preprocessing

### 3.1 Features and Label Separation

```python
X = df.drop('Outcome', axis=1)
y = df['Outcome']
```

### 3.2 Feature Selection

Mutual Information is used to assess feature importance. Two features are dropped:

* **SkinThickness**
* **BloodPressure**

### 3.3 Train–Test Split

Performed using stratification to maintain class balance:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_new, y, stratify=y, random_state=10
)
```

### 3.4 Feature Scaling

StandardScaler ensures all features share a similar scale.

---

## 🤖 4. Model Building & Evaluation

A generic evaluation function computes:

* Accuracy
* Precision
* Recall
* F1-score
* Cross-validation accuracy
* Confusion matrix
* Classification report

### Trained Models

The following classifiers are trained and evaluated:

* Random Forest
* Gradient Boosting
* AdaBoost
* Decision Tree
* Support Vector Machine
* K-Nearest Neighbors
* Logistic Regression
* Gaussian Naive Bayes
* XGBoost

### Best Model

**Logistic Regression** achieved the highest evaluation metrics and is selected for deployment.

---

## 💾 5. Saving the Best Model

A pipeline including scaling + classifier is saved with Joblib:

```python
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
joblib.dump(pipe_lr, './model/diabetes_lr_model.pkl')
```

---

## 🔍 6. Loading and Predicting

The saved model is loaded and used to predict diabetes for a new instance:

```python
loaded_pipe_lr = joblib.load('./model/diabetes_lr_model.pkl')
example = [[2, 101, 90, 21.80, 0.15, 22]]
prediction = loaded_pipe_lr.predict(example)
```

Prediction for the given example:

* Pregnancies: 2
* Glucose: 101
* Insulin: 90
* BMI: 21.80
* DiabetesPedigreeFunction: 0.15
* Age: 22

➡️ **Outcome = 0 (No Diabetes)**

---

## 📌 Summary

This project demonstrates a full end-to-end machine learning workflow including:

* Data acquisition
* Exploratory data analysis
* Visualization
* Preprocessing
* Feature selection
* Training of multiple models
* Robust evaluation
* Model saving & inference

It can serve as a strong baseline for medical diagnostic prediction tasks.

---

## 📝 Requirements

Python libraries needed:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
joblib
kagglehub
```
