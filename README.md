# Heart_attack_model

# Heart Attack Prediction using Machine Learning

This project applies supervised machine learning algorithms to predict the likelihood of a heart attack based on patient health indicators such as age, cholesterol, blood pressure, and other clinical features.

---

## Dataset

- **Source:** [Heart Attack Dataset by Fatemeh Mohammadinia (Kaggle)](https://www.kaggle.com/datasets/fatemehmohammadinia/heart-attack-dataset-tarik-a-rashid)  
- **Description:**  
  The dataset contains medical attributes like age, cholesterol level, resting ECG results, and exercise-induced angina.  
  The target variable indicates whether a person is likely to have a heart attack.

---

## Project Workflow

1. **Data Loading and Exploration**  
   - Checked data structure, null values, and feature types.  
   - Visualized correlations using heatmaps and pair plots.

2. **Data Preprocessing**  
   - Handled missing values.  
   - Encoded categorical features.  
   - Scaled numerical variables.

3. **Model Building**  
   The following algorithms from scikit-learn and XGBoost were used:  
   - DecisionTreeClassifier  
   - RandomForestClassifier  
   - XGBClassifier  

4. **Model Evaluation**  
   - Used 5-fold cross-validation (cv=5) for robust evaluation.  
   - Compared mean accuracy scores across all models.

---

## Results

Cross-Validation Score (cv=5) 

Decision Tree 93.58% 

Random Forest 96.23% 

XGBoost 96.23% 

Best performing models: Random Forest and XGBoost (tied at 96.23%).



## Key Insights

- Ensemble methods (Random Forest, XGBoost) performed better than a single decision tree.  
- Cross-validation provided a reliable measure of model generalization.  
- Feature importance can help identify key health indicators related to heart attack risk.



## Tech Stack

- **Language:** Python  
- **Libraries:**  
  - pandas, numpy, matplotlib, seaborn  
  - scikit-learn  
  - xgboost

