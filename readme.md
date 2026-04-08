# Email Spam Classifier using TF-IDF

A simple NLP project that classifies emails/messages as:
- `0` -> Ham (not spam)
- `1` -> Spam

The model uses:
- `TfidfVectorizer(stop_words="english", max_df=0.7)`
- `LogisticRegression`

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

- Keep `spam.csv` in the same folder as `TF-IDF.py`.
- If the dataset file is missing or columns are incorrect, the script prints a clear error message.
