# 📧 Email Spam Classifier (Machine Learning)

A machine learning project that classifies emails as **spam or not spam** using a **Naive Bayes classifier** on a preprocessed dataset.

---

## 🚀 Overview

This project uses a dataset of emails represented as **word frequency features** (Bag-of-Words format) to train a classification model.

The model predicts whether an email is:

* `0` → Not Spam
* `1` → Spam

---

## 🧠 Model Used

* **Multinomial Naive Bayes**
* Suitable for text classification and frequency-based features

---

## 📂 Project Structure

```
NLP/
│
├── spam_classifier.py   # Main script
├── emails.csv           # Dataset
├── venv/                # Virtual environment (optional)
```

---

## 📊 Features

* Data preprocessing (drop unnecessary columns)
* Train-test split
* Model training using Naive Bayes
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* Visualization:

  * Confusion Matrix
  * Performance comparison chart

---

## ⚙️ Installation

### 1. Clone or download the project

```bash
git clone <https://github.com/pradnyadokh/Document-Classification-Using-NLP-Technique.git>
cd NLP
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

**Windows:**

```bash
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install pandas scikit-learn matplotlib
```

---

## ▶️ How to Run

```bash
python spam_classifier.py
```

---

## 📈 Output

### Terminal Output:

* Accuracy score
* Classification report (precision, recall, F1-score)

### Visual Output:

* Confusion Matrix (model performance)
* Bar chart comparing:

  * Precision
  * Recall
  * F1 Score

---

## 📊 Example Results

```
Accuracy: ~95%

Class 0 (Not Spam):
Precision: 0.98
Recall:    0.95

Class 1 (Spam):
Precision: 0.89
Recall:    0.96
```

---

## ⚠️ Important Notes

* The dataset is **already preprocessed** (Bag-of-Words format)
* No text cleaning or TF-IDF is required
* Do NOT apply NLP preprocessing again (it will break the model logic)

---

## 🧠 Key Learnings

* Difference between raw text vs processed features
* Importance of dataset structure
* Model evaluation beyond accuracy
* Visualization of ML performance

---

## 🚀 Future Improvements

* Add Logistic Regression / SVM for comparison
* Hyperparameter tuning
* Save/load trained model
* Build API using FastAPI
* Deploy as a web app

---

## 📌 Author

Pradnya Dokh

---

## 🏁 Final Thought

This project demonstrates the **complete ML pipeline**:
data → training → evaluation → visualization
