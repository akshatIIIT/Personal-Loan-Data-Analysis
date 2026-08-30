# Personal Loan Campaign Classification

## 📌 Project Overview

This project uses **Machine Learning and Data Analysis** to help a bank identify customers who are more likely to purchase a personal loan.

The objective is to build a classification model that predicts whether a liability customer will accept a personal loan offer based on demographic, financial, and banking-related attributes.

The project uses **Decision Tree Classification** and compares different approaches to control overfitting, including:

* Default Decision Tree
* Pre-pruned Decision Tree
* Post-pruned Decision Tree

The models are evaluated using **Accuracy, Precision, Recall, and F1 Score**.

---

## 🎯 Business Problem

AllLife Bank wants to increase its personal-loan customer base while retaining existing deposit customers.

A previous marketing campaign achieved a conversion rate of more than 9%. The bank wants to improve its targeting strategy by identifying customers with a higher probability of accepting a personal loan.

This project aims to answer:

> **Which customers are most likely to accept a personal loan, and which customer attributes are important in making this prediction?**

---

## 🎯 Objectives

* Predict whether a customer will accept a personal loan.
* Analyze customer characteristics associated with loan acceptance.
* Identify important features influencing loan decisions.
* Reduce unnecessary manual customer screening.
* Provide actionable recommendations for future marketing campaigns.
* Compare different Decision Tree configurations and select an appropriate model.

---

## 📊 Dataset

The dataset contains **5,000 customer records and 14 columns**.

### Features

| Feature              | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| `ID`                 | Unique customer ID                                                          |
| `Age`                | Customer's age in years                                                     |
| `Experience`         | Years of professional experience                                            |
| `Income`             | Annual income in thousand dollars                                           |
| `ZIPCode`            | Customer's home ZIP code                                                    |
| `Family`             | Family size                                                                 |
| `CCAvg`              | Average monthly credit-card spending in thousand dollars                    |
| `Education`          | Education level: 1 = Undergraduate, 2 = Graduate, 3 = Advanced/Professional |
| `Mortgage`           | Mortgage value in thousand dollars                                          |
| `Personal_Loan`      | Target variable: 0 = No, 1 = Yes                                            |
| `Securities_Account` | Whether the customer has a securities account                               |
| `CD_Account`         | Whether the customer has a certificate of deposit account                   |
| `Online`             | Whether the customer uses internet banking                                  |
| `CreditCard`         | Whether the customer uses a credit card issued by another bank              |

The target variable is:

```text
Personal_Loan
```

where `0` represents rejection/non-acceptance and `1` represents acceptance of the personal loan.

---

## 🔍 Exploratory Data Analysis

The project performs both **univariate and bivariate analysis** to understand customer behavior and relationships between features.

Key areas investigated include:

* Distribution of mortgage values
* Credit-card ownership
* Correlation between features and personal-loan acceptance
* Relationship between age and loan acceptance
* Relationship between education level and loan acceptance
* Correlation between age and professional experience

### Key EDA Insights

* `Income` has a strong relationship with personal-loan acceptance.
* Age and experience are highly correlated.
* Mortgage values contain several outliers and are positively skewed.
* Customers with higher education levels show a greater tendency to accept personal loans.
* The dataset contains 3,530 customers without a credit card and 1,470 customers with a credit card.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### 1. Outlier Treatment

Negative values in the `Experience` feature were identified and converted to `0`.

Mortgage outliers were retained because Decision Trees are relatively insensitive to such feature distributions.

### 2. Feature Engineering

Feature preparation was performed before training the model, including encoding the city-related information derived from the ZIP code.

### 3. Train-Test Split

The dataset was divided into:

```text
80% → Training Data
20% → Testing Data
```

using:

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🤖 Machine Learning Approach

### Decision Tree Classification

A Decision Tree classifier was initially trained using the default configuration:

```python
DecisionTreeClassifier(random_state=42)
```

The initial model achieved high accuracy but showed signs of **overfitting**, with very small leaf-node sample sizes.

---

## 🌳 Model Optimization

To address overfitting, two pruning approaches were explored.

### 1. Pre-Pruning

Different combinations of:

* `max_depth`
* `max_leaf_nodes`
* `min_samples_split`

were tested.

The best model was selected by minimizing the difference between training and testing F1 scores.

### 2. Post-Pruning

Cost-complexity pruning was applied using:

```python
cost_complexity_pruning_path()
```

Different `ccp_alpha` values were evaluated, and the model with the highest testing F1 score was selected.

Three models were ultimately compared:

```text
Decision Tree (Default)
Decision Tree (Pre-pruning)
Decision Tree (Post-pruning)
```

---

## 📈 Model Evaluation

The models were evaluated using:

* **Accuracy** – Overall percentage of correct predictions.
* **Precision** – How many predicted loan customers actually accepted the loan.
* **Recall** – How many actual loan customers were correctly identified.
* **F1 Score** – Harmonic mean of precision and recall.
* **Confusion Matrix** – Breakdown of correct and incorrect predictions.

The notebook compares the training and testing performance of all three Decision Tree models.

### Model Selection

Based on the notebook's evaluation, the **post-pruned Decision Tree** was selected as the final model because it provided the best overall performance among the evaluated models while reducing the overfitting seen in the default tree.

---

## 🔑 Feature Importance

Feature importance was analyzed using the trained Decision Tree.

The analysis indicates that **Education** is an important factor contributing to personal-loan acceptance.

The EDA also showed that customers with higher education levels were more likely to accept personal loans.

Other customer characteristics such as income, age/family characteristics, and banking behavior can also help the bank identify promising customer segments.

---

## 🔮 Single Customer Prediction

The trained model can also be used to predict the loan acceptance of an individual customer.

Example:

```python
applicant_details = X_test.iloc[:1, :]

prediction = model2.predict(applicant_details)

probability = model2.predict_proba(applicant_details)

print(prediction)
print(probability[0, 1])
```

The model provides both:

* Predicted class
* Probability of accepting the loan

This probability can be used to prioritize customers for marketing campaigns.

---

## 💡 Business Insights & Recommendations

Based on the analysis, the bank can:

1. **Prioritize higher-income customers** who demonstrate a greater likelihood of accepting personal loans.

2. **Target customers with higher education levels**, as education was identified as an important contributing factor.

3. **Use customer demographics and financial characteristics** to create targeted marketing segments rather than contacting the entire customer base.

4. **Prioritize customers in selected geographic locations** where loan acceptance is higher.

5. Customers seeking loans for **education or health-related needs** can be considered higher-priority segments, while also evaluating their income and ability to repay.

6. Using the prediction probability can help the bank **reduce manual screening and improve campaign efficiency**.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical computation
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Machine Learning
* **Jupyter Notebook / Google Colab**

---

## 📂 Project Structure

```text
Personal-Loan-Campaign-Classification/
│
├── Personal_Loan_Campaign_Classfication.ipynb
├── Loan_Modelling.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Personal-Loan-Campaign-Classification
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

The original notebook also uses `uszipcode` and `sqlalchemy-mate` during its setup.

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Personal_Loan_Campaign_Classfication.ipynb
```

### 4. Run the notebook

Run the cells sequentially to perform:

```text
Data Loading
      ↓
Data Exploration
      ↓
EDA
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Decision Tree
      ↓
Pre-Pruning
      ↓
Post-Pruning
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Customer Prediction
      ↓
Business Recommendations
```

---

## 📌 Key Takeaway

This project demonstrates how a **Decision Tree-based classification approach** can be used to identify customers with a higher likelihood of accepting personal loans.

Rather than marketing to the entire customer base, the bank can use the model's predictions and customer characteristics to **focus marketing efforts on high-potential segments**, potentially improving campaign efficiency and reducing unnecessary screening.
