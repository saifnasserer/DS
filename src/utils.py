import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta


def plot_label_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        df,
        names=df['target'].map({0: 'Not Disaster', 1: 'Disaster'}),
        title='Distribution of Disaster vs Non-Disaster Tweets',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    return fig


def plot_text_length_distribution(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    
    fig = px.histogram(
        df,
        x='text_length',
        nbins=50,
        title='Distribution of Tweet Length (Words)',
        labels={'text_length': 'Number of Words', 'count': 'Number of Tweets'},
        marginal='box',
        template='plotly_white'
    )
    return fig


def plot_target_by_length(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    df['target_label'] = df['target'].map({0: 'Not Disaster', 1: 'Disaster'})
    
    fig = px.box(
        df,
        x='target_label',
        y='text_length',
        title='Text Length by Tweet Type',
        labels={'target_label': 'Tweet Type', 'text_length': 'Number of Words'},
        template='plotly_white'
    )
    return fig


def forecast_volume(df: pd.DataFrame, periods: int = 30, window: int = 7) -> pd.DataFrame:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    daily_volume = df.groupby('date')['volume'].sum().reset_index()
    daily_volume = daily_volume.set_index('date')
    
    ma = daily_volume['volume'].rolling(window=window).mean()
    last_ma = ma.iloc[-1] if not pd.isna(ma.iloc[-1]) else daily_volume['volume'].mean()
    
    last_date = daily_volume.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': [last_ma] * periods,
        'yhat_lower': [last_ma * 0.8] * periods,
        'yhat_upper': [last_ma * 1.2] * periods
    })
    
    return forecast_df


def visualize_forecast(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, keyword: str = "") -> go.Figure:
    hist_df = historical_df.copy()
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date')
    
    forecast_df = forecast_df.copy()
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['volume'],
        mode='lines',
        name='Historical Volume',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(color='rgba(255,0,0,0.2)', width=1),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines',
        name='Lower Bound',
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0.2)', width=1),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Tweet Volume Forecast: {keyword}" if keyword else "Tweet Volume Forecast",
        xaxis_title="Date",
        yaxis_title="Tweet Volume",
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

