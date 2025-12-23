# Disaster Tweet Classification & Volume Forecasting System

A comprehensive machine learning system for analyzing disaster-related tweets with two main components:
1. **Disaster Classification**: Binary classification of tweets to identify real disaster situations
2. **Volume Forecasting**: Time series forecasting of tweet volumes for specific topics

## Project Structure

```
DS/
├── api/
│   └── app.py                    # Streamlit web application
├── data/
│   ├── raw/                      # Original dataset
│   │   └── train.csv
│   ├── processed/                # Cleaned, ready-to-train data
│   └── test/                     # Test dataset
│       ├── test.csv
│       └── sample_submission.csv
├── ml_model/                     # Trained models & metrics
│   ├── model.pkl                 # Trained classification model
│   ├── vectorizer.pkl            # TF-IDF vectorizer
│   └── metrics.json              # Test performance report
├── notebooks/
│   └── 01_eda.ipynb             # Data exploration
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # Data cleaning, feature engineering
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Model testing & metrics
│   ├── predict.py                # Load model & predict on new data
│   ├── utils.py                  # Helper functions (visualization, forecasting)
│   └── modeling.py               # Model definitions
├── config.yaml                   # Hyperparameters & paths
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd DS
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (automatically handled):
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### Training the Classification Model

Train the disaster classification model:

```bash
python src/train.py
```

This will:
- Load data from `data/raw/train.csv`
- Clean and preprocess the data
- Extract TF-IDF features (5000 features)
- Train three models: Logistic Regression, SVM, and Naive Bayes
- Evaluate all models and display performance metrics
- Save the best model (Logistic Regression) to `ml_model/`
- Save metrics to `ml_model/metrics.json`

**Output Files:**
- `ml_model/model.pkl` - Trained classification model
- `ml_model/vectorizer.pkl` - TF-IDF vectorizer
- `ml_model/metrics.json` - Performance metrics

### Running the Web Application

Launch the Streamlit application:

```bash
streamlit run api/app.py
```

The application provides three main features:

1. **Data Exploration**: Visualize dataset characteristics and distributions
2. **Disaster Classification**: Real-time tweet classification with confidence scores
3. **Volume Forecasting**: Predict future tweet volumes with interactive charts

## Model Architecture

### Classification Models

The system trains and evaluates three classification models:

1. **Logistic Regression** (selected for deployment)
   - Regularized with max_iter=500
   - Provides probability estimates

2. **Support Vector Machine (LinearSVC)**
   - Linear kernel for efficiency
   - Good performance on high-dimensional sparse data

3. **Naive Bayes (MultinomialNB)**
   - Probabilistic classifier
   - Fast training and prediction

### Feature Engineering

- **Text Cleaning**: Remove URLs, mentions, special characters
- **Stopword Removal**: Remove common English stopwords
- **Lemmatization**: Reduce words to their root forms
- **TF-IDF Vectorization**: Extract 5000 most important features

## Configuration

Edit `config.yaml` to customize:
- Data paths
- Model directory
- Training parameters (test_size, random_state, max_features)
- Forecasting defaults

## Data Requirements

### Classification Training Data

CSV file with columns:
- `id`: Unique identifier
- `keyword`: Optional keyword
- `location`: Optional location
- `text`: Tweet text content
- `target`: Binary label (0 = not disaster, 1 = disaster)

Place training data in `data/raw/train.csv`

## API Documentation

### Core Modules

#### `src/preprocess.py`
- `load_data(path)` - Load dataset from CSV
- `clean_dataframe(df)` - Clean and prepare dataframe
- `clean_text(text)` - Clean individual text strings
- `preprocess_text(text)` - Preprocess with stopword removal and lemmatization
- `extract_volume_data(df, keyword)` - Extract daily tweet volumes

#### `src/train.py`
- `train(data_path, model_dir)` - Complete training pipeline

#### `src/evaluate.py`
- `evaluate(model, X_test, y_test)` - Generate classification report

#### `src/predict.py`
- `load_model(model_dir)` - Load trained model and vectorizer
- `predict(text, model, vectorizer)` - Predict on new text

#### `src/utils.py`
- `plot_label_distribution(df)` - Create label distribution chart
- `plot_text_length_distribution(df)` - Create text length histogram
- `plot_target_by_length(df)` - Compare lengths by tweet type
- `forecast_volume(df, periods, window)` - Generate volume forecast
- `visualize_forecast(historical_df, forecast_df, keyword)` - Create forecast chart

## Performance Metrics

Models are evaluated using:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

Metrics are saved to `ml_model/metrics.json` after training.

## Dependencies

- `streamlit` - Web application framework
- `scikit-learn` - Machine learning library
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `nltk` - Natural language processing
- `joblib` - Model serialization
- `plotly` - Interactive visualizations

## License

This project is intended for academic/educational purposes.
