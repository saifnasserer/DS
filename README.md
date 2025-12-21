# Emergency Tweet Sentiment Classifier

A machine learning project for classifying emergency tweets as positive or negative/critical sentiment using NLP techniques.

## Project Structure

```
DS/
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── data/
│   └── sentiment140.csv   # Training dataset
├── src/
│   ├── train_pipeline.py      # Main training pipeline
│   ├── data_collection.py      # Data loading utilities
│   ├── data_cleaning.py        # Data cleaning functions
│   ├── eda.py                  # Exploratory data analysis
│   ├── feature_engineering.py  # Text preprocessing
│   ├── modeling.py             # Model training
│   ├── evaluation.py           # Model evaluation
│   ├── visualization.py        # Data visualization
│   └── clustering.py           # Clustering utilities
└── notebooks/
    └── exploration.ipynb      # Jupyter notebook for exploration
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (if not already downloaded):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### Training the Model

Run the training pipeline to train models and generate the model files:

```bash
python src/train_pipeline.py
```

This will:
- Load and clean the data
- Extract TF-IDF features
- Train multiple models (Logistic Regression, SVM, Naive Bayes)
- Evaluate all models
- Save the best model and vectorizer as `.pkl` files

### Running the Web App

After training, launch the Streamlit application:

```bash
streamlit run app.py
```

The app allows you to input tweets and get sentiment predictions in real-time.

## Models

The pipeline trains three models:
- **Logistic Regression** (selected for deployment)
- **Support Vector Machine (SVM)**
- **Naive Bayes**

All models are evaluated and the Logistic Regression model is saved for the web application.

## Data

The project uses the Sentiment140 dataset. Make sure the CSV file is placed in the `data/` directory.

## Output Files

After training, the following files will be generated:
- `final_sentiment_model.pkl` - Trained sentiment classification model
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer for text preprocessing

