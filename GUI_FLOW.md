# GUI Flow Explanation

## Application Overview

The Streamlit application has **three main screens** accessible via the sidebar navigation. Each screen serves a specific purpose in the disaster tweet analysis workflow.

---

## Screen 1: Data Exploration

### Purpose
Explore and visualize the dataset to understand its characteristics before model training.

### Flow:
1. **Page Load**
   - Automatically loads `data/train.csv`
   - Uses `src/data_collection.load_data()` to read the CSV
   - Uses `src/data_cleaning.clean_dataframe()` to clean the data

2. **Dataset Overview Section**
   - Displays 4 key metrics:
     - Total Tweets count
     - Disaster Tweets count (target=1)
     - Non-Disaster Tweets count (target=0)
     - Disaster Ratio percentage

3. **Label Distribution Visualization**
   - Calls `src/visualization.plot_label_distribution()`
   - Creates a pie chart showing the proportion of disaster vs non-disaster tweets
   - Uses Plotly for interactive visualization

4. **Text Length Analysis**
   - Left column: Histogram of tweet lengths (word count)
     - Uses `src/visualization.plot_text_length_distribution()`
   - Right column: Box plot comparing text lengths by tweet type
     - Uses `src/visualization.plot_target_by_length()`
   - Helps identify if disaster tweets have different characteristics

5. **Sample Data Viewer**
   - Expandable section showing first 20 tweets
   - Displays text and target label
   - Allows manual inspection of data quality

### Files Used:
- `src/data_collection.py` → `load_data()`
- `src/data_cleaning.py` → `clean_dataframe()`
- `src/visualization.py` → All three plotting functions

---

## Screen 2: Disaster Classification

### Purpose
Real-time classification of tweets to determine if they describe a real disaster.

### Prerequisites:
- Model files must exist: `final_sentiment_model.pkl` and `tfidf_vectorizer.pkl`
- These are created by running: `python src/train_pipeline.py`

### Flow:

1. **Page Load**
   - Attempts to load pre-trained model and vectorizer
   - If files not found, shows error message with instructions

2. **User Input**
   - Text area for entering tweet text
   - User types or pastes a tweet

3. **Classification Process** (when "Classify Tweet" button clicked):
   ```
   User Input → Text Preprocessing → Feature Extraction → Model Prediction → Display Results
   ```
   
   **Step-by-step:**
   - **Text Preprocessing**: 
     - Uses `preprocess()` function (defined in app.py)
     - Calls `clean_text()` to remove URLs, mentions, special characters
     - Removes stopwords and applies lemmatization
   
   - **Feature Extraction**:
     - Uses loaded `vectorizer` (TF-IDF vectorizer)
     - Transforms preprocessed text into numerical features
   
   - **Model Prediction**:
     - Uses loaded `model` (Logistic Regression)
     - Gets prediction (0 = Not Disaster, 1 = Disaster)
     - Gets probability scores for both classes
   
   - **Display Results**:
     - Shows classification result (Disaster/Not Disaster)
     - Displays confidence percentage
     - Shows appropriate message based on prediction

### Files Used:
- `final_sentiment_model.pkl` (pre-trained model)
- `tfidf_vectorizer.pkl` (pre-trained vectorizer)
- Preprocessing functions in `app.py`

### Training the Model:
Before using this screen, run:
```bash
python src/train_pipeline.py
```

This will:
1. Load and clean data from `data/train.csv`
2. Extract TF-IDF features
3. Train 3 models (Logistic Regression, SVM, Naive Bayes)
4. Evaluate all models
5. Save the best model (Logistic Regression) and vectorizer

---

## Screen 3: Volume Forecasting

### Purpose
Predict future tweet volumes based on historical data from the dataset.

### Flow:

1. **Page Load**
   - Checks if forecasting modules are available
   - Verifies `data/train.csv` exists

2. **Configuration Inputs**
   - **Keyword Filter** (optional): Filter tweets by keyword (e.g., "earthquake")
   - **Moving Average Window**: Number of days for moving average (1-30, default: 7)
   - **Forecast Periods**: Number of future days to predict (1-365, default: 30)

3. **Forecast Generation** (when "Generate Forecast" button clicked):
   ```
   Load Data → Extract Volume Data → Calculate Moving Average → Generate Forecast → Visualize
   ```
   
   **Step-by-step:**
   - **Load Data**:
     - Uses `src/data_collection.load_data()` to load `data/train.csv`
   
   - **Extract Volume Data**:
     - Uses `src/data_collection.extract_volume_data()`
     - If keyword provided, filters tweets containing that keyword
     - Groups tweets by date and counts daily volume
     - Creates date-based volume time series
   
   - **Generate Forecast**:
     - Uses `src/volume_forecasting.forecast_volume()`
     - Calculates moving average using specified window
     - Projects future volumes based on recent average
     - Creates confidence intervals (upper/lower bounds)
   
   - **Display Results**:
     - Shows forecast metrics (average, min, max predicted volume)
     - Displays interactive visualization
     - Provides CSV download option
     - Shows detailed forecast table

4. **Visualization**
   - Uses `src/forecast_pipeline.visualize_forecast()`
   - Creates Plotly chart with:
     - Blue line: Historical volume
     - Red dashed line: Forecast
     - Shaded area: Confidence interval

### Files Used:
- `src/data_collection.py` → `load_data()`, `extract_volume_data()`
- `src/volume_forecasting.py` → `forecast_volume()`
- `src/forecast_pipeline.py` → `visualize_forecast()`

---

## Complete User Journey Example

### Scenario: New User Exploring the System

1. **First Visit - Data Exploration**
   - Opens app → Selects "Data Exploration"
   - Views dataset statistics and visualizations
   - Understands data distribution and characteristics
   - Reviews sample tweets

2. **Training the Model** (if not done)
   - Runs: `python src/train_pipeline.py`
   - Waits for training to complete (~2-5 minutes)
   - Model files are created in project root

3. **Testing Classification**
   - Selects "Disaster Classification"
   - Enters test tweet: "Fire broke out in downtown building, need help!"
   - Clicks "Classify Tweet"
   - Sees result: "Real Disaster Tweet" with confidence score

4. **Volume Forecasting**
   - Selects "Volume Forecasting"
   - Optionally enters keyword: "earthquake"
   - Sets forecast period: 30 days
   - Clicks "Generate Forecast"
   - Views forecast chart and downloads CSV

---

## Data Flow Diagram

```
┌─────────────────┐
│  data/train.csv │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌────────────────┐  ┌──────────────────┐
│ Data           │  │ Disaster          │
│ Exploration    │  │ Classification    │
│ Screen         │  │ Screen            │
│                │  │                   │
│ • Load data    │  │ • Load model      │
│ • Visualize    │  │ • Preprocess      │
│ • Show stats   │  │ • Predict         │
└────────────────┘  └──────────────────┘
         │                 │
         │                 │
         └────────┬────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Volume           │
         │ Forecasting      │
         │ Screen           │
         │                  │
         │ • Extract volume │
         │ • Forecast       │
         │ • Visualize      │
         └──────────────────┘
```

---

## Key Dependencies Between Screens

1. **Data Exploration** → Independent (only needs data file)
2. **Disaster Classification** → Requires trained model files
3. **Volume Forecasting** → Independent (uses data file directly)

---

## Error Handling

Each screen includes error handling:
- **Data Exploration**: Handles missing data file, import errors
- **Disaster Classification**: Checks for model files, handles prediction errors
- **Volume Forecasting**: Validates data, handles forecasting errors

All errors are displayed with helpful messages and expandable error details.

