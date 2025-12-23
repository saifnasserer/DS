# Project Structure & GUI Flow

## New Project Structure (Following Best Practices)

```
DS/
├── api/
│   └── app.py                    # Streamlit web application
├── data/
│   ├── raw/                      # Original dataset (CSV files)
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
│   └── 01_eda.ipynb             # Data exploration notebook
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

## Code Organization

### Removed Unused Files:
- ❌ `src/clustering.py` - Not used anywhere
- ❌ `src/eda.py` - Functionality moved to `utils.py`
- ❌ `src/data_collection.py` - Consolidated into `preprocess.py`
- ❌ `src/data_cleaning.py` - Consolidated into `preprocess.py`
- ❌ `src/feature_engineering.py` - Consolidated into `preprocess.py`
- ❌ `src/train_pipeline.py` - Renamed to `train.py`
- ❌ `src/evaluation.py` - Renamed to `evaluate.py`
- ❌ `src/visualization.py` - Moved to `utils.py`
- ❌ `src/volume_forecasting.py` - Moved to `utils.py`
- ❌ `src/forecast_pipeline.py` - Moved to `utils.py`

### New Consolidated Structure:
- ✅ `src/preprocess.py` - All data loading, cleaning, and preprocessing
- ✅ `src/train.py` - Complete training pipeline
- ✅ `src/evaluate.py` - Model evaluation
- ✅ `src/predict.py` - Prediction functionality
- ✅ `src/utils.py` - Visualization and forecasting utilities
- ✅ `src/modeling.py` - Model definitions (unchanged)

## GUI Flow Explanation

### Screen 1: Data Exploration

**Purpose:** Explore dataset characteristics before model training

**Flow:**
1. **Load Data**
   - Path: `data/raw/train.csv`
   - Uses: `src.preprocess.load_data()`
   - Uses: `src.preprocess.clean_dataframe()`

2. **Display Metrics**
   - Total tweets, disaster tweets, non-disaster tweets, ratio

3. **Visualizations**
   - Label distribution: `src.utils.plot_label_distribution()`
   - Text length histogram: `src.utils.plot_text_length_distribution()`
   - Text length comparison: `src.utils.plot_target_by_length()`

4. **Sample Data**
   - Shows first 20 cleaned tweets

**Files Used:**
- `src/preprocess.py` → `load_data()`, `clean_dataframe()`
- `src/utils.py` → All visualization functions

---

### Screen 2: Disaster Classification

**Purpose:** Real-time tweet classification

**Flow:**
1. **Load Model**
   - Path: `ml_model/model.pkl`, `ml_model/vectorizer.pkl`
   - Uses: `src.predict.load_model()`
   - Checks if files exist

2. **User Input**
   - Text area for tweet input

3. **Classification**
   - Uses: `src.predict.predict()`
   - Internally calls: `src.preprocess.clean_text()` and `src.preprocess.preprocess_text()`
   - Returns: prediction (0/1) and confidence score

4. **Display Result**
   - Shows classification and confidence percentage

**Files Used:**
- `src/predict.py` → `load_model()`, `predict()`
- `src/preprocess.py` → `clean_text()`, `preprocess_text()`

**Prerequisites:**
- Must run `python src/train.py` first to create model files

---

### Screen 3: Volume Forecasting

**Purpose:** Predict future tweet volumes

**Flow:**
1. **Load Data**
   - Path: `data/raw/train.csv`
   - Uses: `src.preprocess.load_data()`

2. **Extract Volume Data**
   - Uses: `src.preprocess.extract_volume_data()`
   - Optionally filters by keyword
   - Groups by date and counts daily volumes

3. **Generate Forecast**
   - Uses: `src.utils.forecast_volume()`
   - Calculates moving average
   - Projects future volumes

4. **Visualize**
   - Uses: `src.utils.visualize_forecast()`
   - Creates interactive Plotly chart

5. **Download**
   - Provides CSV download of forecast results

**Files Used:**
- `src/preprocess.py` → `load_data()`, `extract_volume_data()`
- `src/utils.py` → `forecast_volume()`, `visualize_forecast()`

---

## Complete Workflow

### Step 1: Data Exploration
```
User → Select "Data Exploration" 
     → Loads data/raw/train.csv
     → Displays statistics and visualizations
```

### Step 2: Train Model (if needed)
```bash
python src/train.py
```
```
Loads data/raw/train.csv
     → Preprocesses (clean, feature engineering)
     → Trains models
     → Evaluates
     → Saves to ml_model/
```

### Step 3: Classification
```
User → Select "Disaster Classification"
     → Loads ml_model/model.pkl
     → User enters tweet
     → Predicts using src/predict.py
     → Displays result
```

### Step 4: Volume Forecasting
```
User → Select "Volume Forecasting"
     → Loads data/raw/train.csv
     → Extracts volume data
     → Generates forecast
     → Visualizes and allows download
```

---

## File Mapping to ML Workflow Sections

| Section | Files | Description |
|---------|-------|-------------|
| 1. Data Exploration | `src/utils.py` (visualization functions)<br>`notebooks/01_eda.ipynb` | EDA and visualization |
| 2. Data Preprocessing | `src/preprocess.py` | Data loading, cleaning, feature engineering |
| 3. Feature Engineering | `src/preprocess.py` | Text preprocessing, TF-IDF (in train.py) |
| 4. Model Training | `src/train.py`<br>`src/modeling.py` | Model selection and training |
| 5. Hyperparameter Tuning | `config.yaml` | Configuration (currently uses defaults) |
| 6. Model Evaluation | `src/evaluate.py`<br>`src/train.py` | Evaluation and metrics |
| 7. GUI Deployment | `api/app.py` | Streamlit web application |

---

## Running the Application

### Training:
```bash
python src/train.py
```

### Running GUI:
```bash
streamlit run api/app.py
```

All paths are now properly organized and follow the standard ML project structure!

