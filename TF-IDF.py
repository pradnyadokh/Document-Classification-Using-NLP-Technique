import os
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load spam dataset and prepare label/message columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, encoding="latin-1")

    # Support common column name variants and standardize to label/message
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
    elif {"Category", "Message"}.issubset(df.columns):
        df = df[["Category", "Message"]].copy()
    else:
        raise ValueError(
            "Dataset must contain either columns (v1, v2) or (Category, Message)"
        )

    df.columns = ["label", "message"]

    # Convert labels: ham -> 0, spam -> 1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Remove rows with invalid labels/messages if any
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(int)

    return df


def train_model(df: pd.DataFrame):
    """Split data, vectorize text with TF-IDF, and train Logistic Regression."""
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Stopword removal is handled by TfidfVectorizer with stop_words='english'
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer, X_test_tfidf, y_test


def evaluate_model(model, X_test_tfidf, y_test):
    """Print evaluation metrics and return confusion matrix."""
    y_pred = model.predict(X_test_tfidf)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    return cm


def plot_confusion_matrix(cm):
    """Plot confusion matrix with clear axis labels."""
    plt.figure(figsize=(6, 4))
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ham (0)", "Spam (1)"],
            yticklabels=["Ham (0)", "Spam (1)"],
        )
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], ["Ham (0)", "Spam (1)"])
        plt.yticks([0, 1], ["Ham (0)", "Spam (1)"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def predict_custom_message(message: str, model, vectorizer) -> int:
    """Predict a custom message. Returns 1 for spam, 0 for ham."""
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    return int(prediction)


def main():
    file_path = "spam.csv"

    try:
        df = load_and_prepare_data(file_path)
        model, vectorizer, X_test_tfidf, y_test = train_model(df)
        cm = evaluate_model(model, X_test_tfidf, y_test)
        plot_confusion_matrix(cm)

        # Bonus: example custom prediction
        sample_message = "Congratulations! You won a free vacation. Call now!"
        pred = predict_custom_message(sample_message, model, vectorizer)
        print(f"\nCustom message prediction: {'Spam' if pred == 1 else 'Ham'}")

    except FileNotFoundError as err:
        print(f"Error: {err}")
    except ValueError as err:
        print(f"Data Error: {err}")
    except Exception as err:
        print(f"Unexpected error: {err}")


if __name__ == "__main__":
    main()
