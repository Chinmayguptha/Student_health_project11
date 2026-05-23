import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Setup paths
os.makedirs("results/plots", exist_ok=True)

# 2. Load and Prepare Data
df = pd.read_csv("data/student_health_cleaned.csv")
target_col = "Health_Risk_Level"

# Encode features and target
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop(target_col, axis=1)
y = df[target_col]
class_names = le.classes_ if hasattr(le, 'classes_') else np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Generate FIXED Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
# Create the heatmap WITHOUT annotations first to control them manually
ax = sns.heatmap(cm, annot=False, cmap="Blues", 
                 xticklabels=class_names, yticklabels=class_names)

# --- THE FIX: Smart Text Color ---
thresh = cm.max() / 2.  # Find the midpoint of the color scale
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        # If value is above threshold (dark background), use white text
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(j + 0.5, i + 0.5, cm[i, j], 
                ha="center", va="center", color=color, fontsize=12, fontweight='bold')

plt.title("Confusion Matrix - Optimized Visibility")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# 5. Save the result
save_path = "results/plots/fixed_confusion_matrix.jpg"
plt.savefig(save_path, dpi=300)
print(f"✅ Success! Fixed confusion matrix saved to: {save_path}")
plt.show()