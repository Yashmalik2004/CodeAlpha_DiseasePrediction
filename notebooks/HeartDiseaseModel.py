import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/heart.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

clf1 = LogisticRegression()
clf2 = RandomForestClassifier(random_state=42)
clf3 = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

model = VotingClassifier(estimators=[
    ('lr', clf1), ('rf', clf2), ('xgb', clf3)
], voting='soft')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_probs))

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.show()

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC-AUC Curve")
plt.savefig("images/roc_auc_curve.png")
plt.show()
