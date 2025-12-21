import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import load_data
from src.data_cleaning import clean_dataframe, clean_text
from src.feature_engineering import preprocess_text
from src.modeling import train_models
from src.evaluation import evaluate

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def run_pipeline(data_path):
    print("🚀 Starting training pipeline...")
    
    # 1. Load
    print("📥 Loading data...")
    df = load_data(data_path)
    print(f"   Loaded {len(df)} rows")

    # 2. Clean
    print("🧹 Cleaning data...")
    df = clean_dataframe(df)
    df['text'] = df['text'].apply(clean_text)
    df['text'] = df['text'].apply(preprocess_text)
    print(f"   After cleaning: {len(df)} rows")

    # 3. Features
    print("🔧 Extracting features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    y = df['target']
    print(f"   Feature matrix shape: {X.shape}")

    # 4. Split
    print("✂️  Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # 5. Train
    print("🤖 Training models...")
    models = train_models(X_train, y_train)
    print("   Trained models: Logistic Regression, SVM, Naive Bayes")
    
    # Select best model (Logistic Regression) for deployment
    model = models['lr']
    
    # Evaluate all models
    print("\n📊 Model Evaluation:")
    for name, model_instance in models.items():
        print(f"\n{name.upper()}:")
        report = evaluate(model_instance, X_test, y_test)
        print(report)

    # 6. Save
    print("\n💾 Saving models...")
    # Save to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "final_sentiment_model.pkl")
    vectorizer_path = os.path.join(project_root, "tfidf_vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"   Saved: {model_path}")
    print(f"   Saved: {vectorizer_path}")

    print("\n✅ Training pipeline completed successfully!")

if __name__ == "__main__":
    # Get the project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "sentiment140.csv")
    run_pipeline(data_path)
