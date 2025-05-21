import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
import os

def create_dashboard(csv_file):
    # Read the CSV data
    df = pd.read_csv(csv_file)
    
    # Get the column names
    bpp_col = df.columns[0]  # Assuming first column is BPP
    model_cols = df.columns[1:]  # All other columns are models
    
    # Create a subplot figure with multiple plots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=(
            "Line Chart Comparison", 
            "Performance Differences",
            "Detailed Model Comparison"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Custom colors for the models
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    colors = colors[:len(model_cols)]  # Use only as many colors as needed
    
    # 1. Top-left: Line chart comparison
    for i, model in enumerate(model_cols):
        fig.add_trace(
            go.Scatter(
                x=df[bpp_col], 
                y=df[model],
                mode='lines+markers',
                name=model,
                line=dict(color=colors[i], width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # 2. Top-right: Bar chart showing average performance
    avg_performance = df[model_cols].mean().values
    fig.add_trace(
        go.Bar(
            x=model_cols,
            y=avg_performance,
            text=[f"{val:.3f}" for val in avg_performance],
            textposition='auto',
            marker_color=colors
        ),
        row=1, col=2
    )
    
    # 3. Bottom: Area chart for detailed comparison
    for i, model in enumerate(model_cols):
        fig.add_trace(
            go.Scatter(
                x=df[bpp_col],
                y=df[model],
                mode='lines',
                fill='tozeroy',
                name=f"{model} (Area)",
                line=dict(width=0.5, color=colors[i]),
                fillcolor=f"rgba({','.join(str(int(int(colors[i].lstrip('#')[j:j+2], 16))) for j in (0, 2, 4))},0.2)"
            ),
            row=2, col=1
        )
    
    # Update layout for better appearance
    fig.update_layout(
        title_text="VAE-GAN Models Comprehensive Analysis Dashboard",
        title_font=dict(size=24),
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text=bpp_col, row=2, col=1)
    fig.update_yaxes(title_text="Performance Metric", row=1, col=1)
    fig.update_yaxes(title_text="Avg Performance", row=1, col=2)
    fig.update_yaxes(title_text="Performance Metric", row=2, col=1)
    
    # Get the filename without extension for output
    output_base = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = f"{output_base}_dashboard.html"
    
    # Save as HTML file
    fig.write_html(output_file)
    print(f"Dashboard saved as {output_file}")
    
    # Return the figure object (useful in Jupyter notebooks)
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create an interactive dashboard for VAE-GAN comparison from CSV file.')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    fig = create_dashboard(args.csv_file)
    # In non-interactive environments, you can use fig.show() to display the plot