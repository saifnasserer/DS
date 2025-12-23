import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import load_model, predict
from src.preprocess import load_data, clean_dataframe, extract_volume_data
from src.utils import (
    plot_label_distribution,
    plot_text_length_distribution,
    plot_target_by_length,
    forecast_volume,
    visualize_forecast
)

try:
    model, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    model = None
    vectorizer = None

st.set_page_config(
    page_title="Disaster Tweet Analysis",
    page_icon="🚨",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", ["Data Exploration", "Disaster Classification", "Volume Forecasting"])

if page == "Data Exploration":
    st.title("Data Exploration and Visualization")
    
    try:
        data_path = os.path.join("data", "raw", "train.csv")
        
        if not os.path.exists(data_path):
            st.error(f"Dataset not found at {data_path}")
        else:
            with st.spinner("Loading data..."):
                df = load_data(data_path)
                df_clean = clean_dataframe(df)
                st.success(f"Loaded {len(df_clean)} tweets")
            
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tweets", len(df_clean))
            with col2:
                st.metric("Disaster Tweets", int(df_clean['target'].sum()))
            with col3:
                st.metric("Non-Disaster Tweets", int((df_clean['target'] == 0).sum()))
            with col4:
                st.metric("Disaster Ratio", f"{(df_clean['target'].mean()*100):.1f}%")
            
            st.subheader("Label Distribution")
            fig = plot_label_distribution(df_clean)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Text Length Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_text_length_distribution(df_clean)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plot_target_by_length(df_clean)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Sample Data")
            with st.expander("View Sample Tweets"):
                st.dataframe(df_clean[['text', 'target']].head(20))
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        with st.expander("Error Details"):
            st.exception(e)

elif page == "Disaster Classification":
    st.title("Disaster Tweet Classifier")
    st.markdown("Classify tweets to determine if they describe a real disaster or not using machine learning.")
    
    if not model_loaded:
        st.error("Model files not found. Please train the model first using: `python src/train.py`")
    else:
        tweet = st.text_area("Enter tweet text", height=100, placeholder="Type or paste tweet here...")
        
        if st.button("Classify Tweet", type="primary"):
            if not tweet or tweet.strip() == "":
                st.warning("Please enter tweet text")
            else:
                try:
                    pred, confidence = predict(tweet, model, vectorizer)
                    
                    if pred == 1:
                        st.error(f"**Classification: Real Disaster Tweet**")
                        st.info(f"Confidence: {confidence*100:.2f}%")
                        st.markdown("⚠️ This tweet appears to describe a real disaster situation.")
                    else:
                        st.success(f"**Classification: Not a Disaster**")
                        st.info(f"Confidence: {confidence*100:.2f}%")
                        st.markdown("✓ This tweet does not appear to describe a disaster.")
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")

elif page == "Volume Forecasting":
    st.title("Tweet Volume Forecasting")
    
    try:
        data_path = os.path.join("data", "raw", "train.csv")
        
        if not os.path.exists(data_path):
            st.error(f"Dataset not found at {data_path}")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                keyword = st.text_input("Filter by Keyword", value="", placeholder="e.g., earthquake (optional)")
            
            with col2:
                window = st.number_input("Moving Average Window", min_value=1, max_value=30, value=7)
            
            with col3:
                periods = st.number_input("Forecast Periods (days)", min_value=1, max_value=365, value=30)
            
            if st.button("Generate Forecast", type="primary"):
                try:
                    with st.spinner("Loading and processing data..."):
                        df = load_data(data_path)
                        volume_df = extract_volume_data(df, keyword=keyword if keyword else None)
                        st.success(f"Loaded {len(volume_df)} data points")
                    
                    with st.expander("Data Preview"):
                        st.dataframe(volume_df.head(10))
                        st.caption(f"Date range: {volume_df['date'].min()} to {volume_df['date'].max()}")
                    
                    with st.spinner("Generating forecast..."):
                        forecast_df = forecast_volume(volume_df, periods=periods, window=window)
                    
                    st.subheader("Forecast Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Predicted Volume", f"{forecast_df['yhat'].mean():.0f}")
                    with col2:
                        st.metric("Minimum Volume", f"{forecast_df['yhat'].min():.0f}")
                    with col3:
                        st.metric("Maximum Volume", f"{forecast_df['yhat'].max():.0f}")
                    
                    st.subheader("Forecast Visualization")
                    fig = visualize_forecast(volume_df, forecast_df, keyword=keyword)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Download Forecast")
                    csv = forecast_df.to_csv(index=False)
                    filename = f"forecast_{keyword}.csv" if keyword else "forecast.csv"
                    st.download_button(
                        label="Download Forecast CSV",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )
                    
                    with st.expander("Forecast Details"):
                        st.dataframe(forecast_df)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.exception(e)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        with st.expander("Error Details"):
            st.exception(e)
