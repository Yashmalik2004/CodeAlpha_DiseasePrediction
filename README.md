# CodeAlpha - Task 4: Disease( heart ) Prediction

This project is built for Task 4 of the CodeAlpha Internship, where i aim to predict whether a patient is likely to have heart disease using machine learning techniques.

## Overview

Initially, we worked with the Cleveland Heart Disease dataset. However, the model's accuracy was limited due to the small sample size (only 300+ rows). Despite applying techniques like SMOTE, hyperparameter tuning, and switching models, the performance plateaued below our target.

To resolve this, we **merged another public heart disease dataset** with similar structure and features. This larger combined dataset helped the model learn better patterns and significantly improved accuracy—reaching over **95%**.

---

## Dataset

- **Sources**:
  - [Cleveland Heart Disease Dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
  - [Additional Heart Dataset from Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) *(used for merging)*

- **Final Merged File**:  
  `merged_heart_disease.csv`

- **Target Column**:  
  `condition` → 0 = No heart disease, 1 = Heart disease

---

## Features

- Data Cleaning & Standardization
- Dataset Merging
- Feature Scaling with `StandardScaler`
- Class Imbalance handling using `SMOTE`
- Model training using **XGBoost**
- Evaluation using:
  - Precision, Recall, F1-Score
  - ROC-AUC Score
- Visual outputs:
  - Confusion Matrix
  - ROC Curve
  - Feature Importance Plot

---

## Folder Structure

```
CodeAlpha_Task4_HeartDiseasePrediction/
├── data/
│ └── meart.csv
├── images/
│ ├── confusion_matrix.png
│ ├── roc_auc_curve.png
│ └── feature_importance.png
├── notebooks/
│ └── HeartDiseaseModel.py
├── requirements.txt
└── README.md
```
## Setup Instructions


1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate

```

2. Install Dependencies
```bash

pip install -r requirements.txt
python notebooks/HeartDiseaseModel.py
```
3. Run the model
```bash
python notebooks/HeartDiseaseModel.py
```

## Output Files

After successful execution, the following visualizations will be generated inside the images/ folder:

confusion_matrix.png – Model prediction performance

roc_auc_curve.png – ROC-AUC Curve

feature_importance.png – Top feature contributions

## Model Details
Target column: target

0 = No heart disease

1 = Presence of heart disease

**Preprocessing:**

Features are scaled using StandardScaler

Dataset is balanced using SMOTE

**Model:**

Trained using XGBoostClassifier

Evaluated on precision, recall, F1-score, and ROC-AUC

Accuracy Achieved: ~95%
(after merging two heart disease datasets for improved training)

---
