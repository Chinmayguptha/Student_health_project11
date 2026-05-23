import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("===== GENERATING MODEL VISUALIZATIONS =====\n")

# Create plots directory
os.makedirs("results/plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/student_health_cleaned.csv")

# Encode categorical columns
categorical_columns = ["Gender", "Physical_Activity", "Sleep_Quality", "Mood"]

encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df["Health_Risk_Level"] = target_encoder.fit_transform(df["Health_Risk_Level"])

X = df.drop("Health_Risk_Level", axis=1)
y = df["Health_Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load tuned model
model = RandomForestClassifier(
    class_weight='balanced',
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=10,
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===============================
# 1️⃣ CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.title("Confusion Matrix - Tuned Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/plots/confusion_matrix.jpg", dpi=300)
plt.close()

# ===============================
# 2️⃣ FEATURE IMPORTANCE
# ===============================
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/plots/feature_importance.jpg", dpi=300)
plt.close()

# ===============================
# 3️⃣ CROSS VALIDATION ACCURACY
# ===============================
cv_scores = cross_val_score(model, X, y, cv=5)

plt.figure(figsize=(6,5))
plt.plot(range(1,6), cv_scores, marker='o')
plt.title("5-Fold Cross Validation Accuracy")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.ylim(0.8,1.0)
plt.tight_layout()
plt.savefig("results/plots/cross_validation_accuracy.jpg", dpi=300)
plt.close()

# ===============================
# 4️⃣ VALIDATION CURVE
# ===============================
param_range = [50, 100, 200, 300, 400]
train_scores, test_scores = validation_curve(
    RandomForestClassifier(random_state=42),
    X, y,
    param_name="n_estimators",
    param_range=param_range,
    cv=5
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(6,5))
plt.plot(param_range, train_mean, label="Training Score")
plt.plot(param_range, test_mean, label="Validation Score")
plt.title("Validation Curve - n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("results/plots/validation_curve.jpg", dpi=300)
plt.close()

# ===============================
# 5️⃣ EVALUATION METRICS TABLE IMAGE
# ===============================
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

plt.figure(figsize=(10,6))
plt.axis('off')
plt.table(cellText=np.round(report_df.values,3),
          colLabels=report_df.columns,
          rowLabels=report_df.index,
          loc='center')
plt.title("Evaluation Metrics Table")
plt.tight_layout()
plt.savefig("results/plots/evaluation_metrics_table.jpg", dpi=300)
plt.close()

# ===============================
# 6️⃣ MODEL COMPARISON BAR CHART
# ===============================
models = ["Logistic (Scaled)", "Random Forest", "Tuned RF"]
accuracies = [0.825, 0.965, 0.98]

plt.figure(figsize=(7,5))
bars = plt.bar(models, accuracies)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.7,1.0)

for i in range(len(accuracies)):
    plt.text(i, accuracies[i]+0.005, round(accuracies[i],3),
             ha='center')

plt.tight_layout()
plt.savefig("results/plots/model_comparison.jpg", dpi=300)
plt.close()

print("All visualizations saved successfully in results/plots/")