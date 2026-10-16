# Bank Fraud Detection

## Overview
This project focuses on detecting fraudulent banking transactions using machine learning.  
The dataset contains historical transaction records, and the goal is to build models that can identify fraudulent activity with high recall and precision.

## Dataset
The dataset used is **BankSim1** (`bs140513_032310.csv`), which contains simulated bank transactions.
[BankSim1 Dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/banksim1)


### Features
| Column | Description |
|--------|-------------|
| `step` | Simulation step (time unit) |
| `customer` | Customer ID |
| `age` | Age group of the customer |
| `gender` | Gender of the customer |
| `zipcodeOri` | Customer ZIP code |
| `merchant` | Merchant ID |
| `zipMerchant` | Merchant ZIP code |
| `category` | Transaction category |
| `amount` | Transaction amount |
| `fraud` | Target variable (1 = fraud, 0 = non-fraud) |

## Project Workflow
1. **Import Libraries**  
   Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, XGBoost, Imbalanced-learn.

2. **Load Data**  
   Read the CSV file into a Pandas DataFrame.

3. **Exploratory Data Analysis (EDA)**  
   - Transaction growth over time (`step`)
   - Top customers by transaction count
   - Transaction distribution by age group
   - Transaction distribution by gender
   - Top merchants by transaction volume
   - Transaction distribution by category
   - Transaction amount distribution (box plot)

4. **Data Preprocessing**  
   - Remove irrelevant columns (`zipcodeOri`, `zipMerchant`)
   - Map age and gender to readable labels
   - Handle categorical variables using One-Hot Encoding
   - Scale numerical features using StandardScaler

5. **Handling Imbalanced Data**  
   Use **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the fraud and non-fraud classes.

6. **Modeling**  
   The following models are trained and evaluated:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost

7. **Evaluation Metrics**  
   - Classification Report (Precision, Recall, F1-Score)
   - ROC-AUC Score
   - Precision-Recall Curve
   - Confusion Matrix

## Key EDA Insights
- **Transaction Growth:** Transaction volume increases over simulation steps, stabilizing near 3,700 transactions per step.
- **Top Customers:** A small group of customers initiates hundreds of transactions (up to 265).
- **Age Distribution:** The 26–35 age group dominates with over 185,000 transactions.
- **Gender Distribution:** Female customers account for the highest transaction volume (324,565), followed by males (268,385).
- **Top Merchants:** Two merchants alone handle over 500,000 transactions, showing heavy concentration.
- **Category Distribution:** `es_transportation` is overwhelmingly the most frequent category (505,119 transactions).
- **Amount Distribution:** Most transactions are small, with a median of 26.90 and a long tail up to 8,329.96.

## Requirements
```txt
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
imbalanced-learn
