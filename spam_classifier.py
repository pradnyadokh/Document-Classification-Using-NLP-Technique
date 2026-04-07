import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

logging.basicConfig(level=logging.INFO)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("emails.csv")

# Drop non-feature column
df = df.drop(columns=["Email No."])

# Features and labels
X = df.drop(columns=["Prediction"])
y = df["Prediction"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Training
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
logging.info("\n" + classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix Chart
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# numbers inside matrix
for i in range(len(cm)):
    for j in range(len(cm[0])):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.show()

# -----------------------------
# Metrics Chart (Precision, Recall, F1)
# -----------------------------
report = classification_report(y_test, y_pred, output_dict=True)

classes = ['0', '1']
precision = [report[c]['precision'] for c in classes]
recall = [report[c]['recall'] for c in classes]
f1 = [report[c]['f1-score'] for c in classes]

x = np.arange(len(classes))
width = 0.25

plt.figure()
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1 Score')

plt.xticks(x, ['Class 0', 'Class 1'])
plt.xlabel("Class")
plt.ylabel("Score")
plt.title("Model Performance")
plt.legend()

plt.show()