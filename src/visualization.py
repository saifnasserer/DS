import plotly.express as px

def plot_label_distribution(df):
    fig = px.pie(df, names='target', title="Label Distribution")
    fig.show()
