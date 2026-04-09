<<<<<<< HEAD
# Email Spam Classifier using TF-IDF

A simple NLP project that classifies emails/messages as:
- `0` -> Ham (not spam)
- `1` -> Spam

The model uses:
- `TfidfVectorizer(stop_words="english", max_df=0.7)`
- `LogisticRegression`
=======
# 📧 Email Spam Classifier (Machine Learning)
>>>>>>> 07e5def98411644505f8b437559d02adb321edc1

## Project Structure

```text
NLP/
|-- TF-IDF.py
|-- spam.csv
|-- emails.csv
|-- readme.md
|-- .gitignore
```

## Dataset Format

The script supports either of these column formats:
- `v1` (label), `v2` (message)
- `Category` (label), `Message` (message)

Labels are mapped as:
- `ham` -> `0`
- `spam` -> `1`

## Features Implemented

- Dataset loading with error handling
- Column validation and renaming to `label` and `message`
- Label conversion to numeric values
- Train/test split (`80/20`)
- TF-IDF vectorization with English stopwords removed
- Logistic Regression training
- Evaluation:
  - Accuracy score
  - Classification report
  - Confusion matrix
- Confusion matrix plot (Seaborn if available, otherwise Matplotlib fallback)
- Custom message prediction function

## Installation

```bash
pip install pandas scikit-learn matplotlib seaborn
```

`seaborn` is optional because the script has a Matplotlib fallback.

## Run

```bash
python TF-IDF.py
```

## Example Output

- Accuracy score in terminal
- Classification report in terminal
- Confusion matrix chart
- Sample custom message prediction:
  - `Spam` or `Ham`

## Notes

<<<<<<< HEAD
- Keep `spam.csv` in the same folder as `TF-IDF.py`.
- If the dataset file is missing or columns are incorrect, the script prints a clear error message.
=======
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
Kaustubh Naikwadi
Aryan Singh
Piyush Dwivedi


---

## 🏁 Final Thought

This project demonstrates the **complete ML pipeline**:
data → training → evaluation → visualization
>>>>>>> 07e5def98411644505f8b437559d02adb321edc1
