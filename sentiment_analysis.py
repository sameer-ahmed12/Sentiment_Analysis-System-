# ============================================================
#   Sentiment Analysis Tool
#   Made by Sameer Ahmed
# ============================================================

import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────
# 1. LABELED DATASET  (tweets / reviews)
# ─────────────────────────────────────────────
DATA = [
    # Positive (label = 1)
    ("I love this product! It's absolutely amazing and works perfectly.", 1),
    ("Great quality, fast shipping, very satisfied with the purchase!", 1),
    ("This is the best thing I've ever bought. Highly recommend!", 1),
    ("Excellent service and the product exceeded my expectations.", 1),
    ("Wonderful experience, the staff were incredibly helpful.", 1),
    ("I enjoy using this every day, it makes my life so much better.", 1),
    ("Fantastic product, five stars without any doubt.", 1),
    ("Really happy with this purchase, it arrived quickly and works well.", 1),
    ("Outstanding quality and incredible value for the price.", 1),
    ("The best purchase I made this year. Absolutely love it!", 1),
    ("Amazing features and very easy to use. Totally worth it.", 1),
    ("So happy I found this. It's perfect for what I needed!", 1),
    ("Superb build quality. I'm thoroughly impressed.", 1),
    ("Brilliant! Does exactly what it promises. Very pleased.", 1),
    ("Couldn't be happier. This exceeded every expectation I had.", 1),

    # Negative (label = -1)
    ("This is horrible, completely broken after two days of use.", -1),
    ("Worst product ever. Total waste of money, deeply disappointed.", -1),
    ("Terrible quality, fell apart immediately. Do not buy this.", -1),
    ("Very unhappy with this purchase. It doesn't work at all.", -1),
    ("Awful experience, customer service was rude and unhelpful.", -1),
    ("This product is garbage. Broke on first use.", -1),
    ("Disappointed and frustrated. Not as described at all.", -1),
    ("Poor quality and slow delivery. Regret buying this.", -1),
    ("Cheap materials and bad craftsmanship. Avoid at all costs.", -1),
    ("Defective item, didn't work out of the box. Very disappointing.", -1),
    ("The worst online purchase experience I have ever had.", -1),
    ("Completely useless. Stopped working after one week.", -1),
    ("Absolute rubbish. Returned immediately. Zero stars.", -1),
    ("Do not waste your money on this. It is a scam.", -1),
    ("Broken on arrival. No response from seller. Furious.", -1),

    # Neutral (label = 0)
    ("Nothing special about this product, just average.", 0),
    ("It's okay, does what it says but nothing impressive.", 0),
    ("Received the item as described, no issues so far.", 0),
    ("Average product, meets expectations but doesn't exceed them.", 0),
    ("It works fine, nothing to complain about or praise.", 0),
    ("Decent quality for the price, got what I paid for.", 0),
    ("Not bad, not great. Just an ordinary product.", 0),
    ("It does the job, but nothing worth writing home about.", 0),
    ("Delivery was on time. Product is as expected.", 0),
    ("Fairly standard item. No complaints but no excitement either.", 0),
    ("It functions correctly. Nothing more, nothing less.", 0),
    ("Packaging was fine. Product looks as shown in the photo.", 0),
]

LABEL_NAMES = {1: "Positive", -1: "Negative", 0: "Neutral"}

# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Clean and normalize raw text."""
    text = text.lower()                          # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)       # remove mentions / hashtags
    text = re.sub(r"[^a-z\s]", " ", text)       # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()     # collapse whitespace
    return text

# ─────────────────────────────────────────────
# 3. BUILD DATASET
# ─────────────────────────────────────────────
texts  = [preprocess(t) for t, _ in DATA]
labels = [l for _, l in DATA]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────
models = {
    "Naive Bayes (TF-IDF)": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf",   MultinomialNB()),
    ]),
    "Naive Bayes (CountVectorizer)": Pipeline([
        ("cv",  CountVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", MultinomialNB()),
    ]),
    "Logistic Regression (TF-IDF)": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
    ]),
}

# ─────────────────────────────────────────────
# 5. EVALUATE MODELS
# ─────────────────────────────────────────────
def evaluate_all():
    print("\n" + "=" * 60)
    print("  SENTIMENT ANALYSIS TOOL  |  Made by Sameer Ahmed")
    print("=" * 60)
    print(f"\n  Dataset   : {len(DATA)} samples "
          f"({sum(1 for _,l in DATA if l==1)} pos | "
          f"{sum(1 for _,l in DATA if l==-1)} neg | "
          f"{sum(1 for _,l in DATA if l==0)} neu)")
    print(f"  Train/Test: {len(X_train)} / {len(X_test)} samples\n")

    trained = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds   = pipeline.predict(X_test)
        acc     = accuracy_score(y_test, preds)
        f1      = f1_score(y_test, preds, average="macro")

        print(f"  Model : {name}")
        print(f"  ├─ Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  └─ Macro F1 : {f1:.4f}\n")
        trained[name] = pipeline

    # Detailed report for best model (LR TF-IDF)
    best_name = "Logistic Regression (TF-IDF)"
    best      = trained[best_name]
    best_pred = best.predict(X_test)

    print("-" * 60)
    print(f"  Detailed Report — {best_name}")
    print("-" * 60)
    print(classification_report(
        y_test, best_pred,
        target_names=["Negative", "Neutral", "Positive"],
        labels=[-1, 0, 1]
    ))

    return trained

# ─────────────────────────────────────────────
# 6. CLI PREDICTION
# ─────────────────────────────────────────────
def cli(trained_models: dict):
    print("=" * 60)
    print("  CLI — Enter text to predict sentiment")
    print("  Commands : 'switch' to change model | 'quit' to exit")
    print("=" * 60)

    model_names = list(trained_models.keys())
    current_idx = 2  # default: Logistic Regression TF-IDF
    current_model = trained_models[model_names[current_idx]]

    print(f"\n  Active model: {model_names[current_idx]}\n")

    while True:
        try:
            user_input = input("  >> Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n  Goodbye!\n")
            break

        if user_input.lower() == "switch":
            print("\n  Available models:")
            for i, n in enumerate(model_names):
                marker = " <-- active" if i == current_idx else ""
                print(f"    [{i+1}] {n}{marker}")
            choice = input("  Select model number: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(model_names):
                current_idx   = int(choice) - 1
                current_model = trained_models[model_names[current_idx]]
                print(f"\n  Switched to: {model_names[current_idx]}\n")
            else:
                print("  Invalid choice.\n")
            continue

        cleaned = preprocess(user_input)

        # Prediction + confidence
        label      = current_model.predict([cleaned])[0]
        proba      = current_model.predict_proba([cleaned])[0]
        classes    = current_model.classes_
        confidence = max(proba) * 100

        print(f"\n  ┌─ Result     : {LABEL_NAMES[label]}")
        print(f"  ├─ Confidence : {confidence:.1f}%")
        print(f"  ├─ Model      : {model_names[current_idx]}")
        print(f"  └─ Probabilities:")
        for cls, prob in sorted(zip(classes, proba), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            print(f"       {LABEL_NAMES[cls]:<10}: {prob*100:5.1f}%  {bar}")
        print()

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    trained_models = evaluate_all()

    if "--eval-only" in sys.argv:
        sys.exit(0)

    cli(trained_models)
