import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
import os

def create_dashboard(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get column names
    bpp_col = df.columns[0]  # BPP column
    model_cols = df.columns[1:]  # Model columns
    
    # Create dataframe for model comparison metrics
    metrics_df = pd.DataFrame(index=model_cols)
    
    # Calculate metrics for each model
    for model in model_cols:
        # Calculate average PSNR
        metrics_df.loc[model, 'Avg PSNR'] = df[model].mean()
        
        # Calculate PSNR improvement (last BPP - first BPP)
        metrics_df.loc[model, 'PSNR Improvement'] = df[model].iloc[-1] - df[model].iloc[0]
        
        # Calculate rate-distortion efficiency at middle point
        mid_idx = len(df) // 2
        metrics_df.loc[model, 'R-D Efficiency'] = df[model].iloc[mid_idx] / df[bpp_col].iloc[mid_idx]
    
    # Create a dashboard with 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "table"}]],
        subplot_titles=('PSNR vs BPP', 'Average PSNR', 'PSNR Gain', 'Performance Metrics'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors = colors[:len(model_cols)]
    
    # 1. Top-left: Main line plot
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
    
    # 2. Top-right: Average PSNR bar chart
    fig.add_trace(
        go.Bar(
            x=model_cols,
            y=metrics_df['Avg PSNR'],
            marker_color=colors,
            text=metrics_df['Avg PSNR'].round(2),
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # 3. Bottom-left: PSNR improvement (difference between first and last BPP)
    # Convert to long format for plotting
    improvement_df = pd.DataFrame({
        'Model': model_cols,
        'First BPP': [df[model].iloc[0] for model in model_cols],
        'Last BPP': [df[model].iloc[-1] for model in model_cols]
    })
    
    for i, model in enumerate(model_cols):
        fig.add_trace(
            go.Scatter(
                x=[df[bpp_col].iloc[0], df[bpp_col].iloc[-1]],
                y=[df[model].iloc[0], df[model].iloc[-1]],
                mode='lines+markers',
                name=f"{model} Gain",
                line=dict(color=colors[i], width=3, dash='dash'),
                marker=dict(size=12, symbol='circle'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add annotation showing the improvement
        improvement = df[model].iloc[-1] - df[model].iloc[0]
        fig.add_annotation(
            x=(df[bpp_col].iloc[0] + df[bpp_col].iloc[-1])/2,
            y=(df[model].iloc[0] + df[model].iloc[-1])/2,
            text=f"+{improvement:.2f}dB",
            showarrow=False,
            font=dict(size=12, color=colors[i]),
            row=2, col=1
        )
    
    # 4. Bottom-right: Performance metrics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Model', 'Avg PSNR (dB)', 'PSNR Gain (dB)', 'R-D Efficiency'],
                font=dict(size=12),
                align='left'
            ),
            cells=dict(
                values=[
                    model_cols,
                    metrics_df['Avg PSNR'].round(2),
                    metrics_df['PSNR Improvement'].round(2),
                    metrics_df['R-D Efficiency'].round(2)
                ],
                font=dict(size=11),
                align='left'
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Comprehensive VAE-GAN Performance Analysis Dashboard",
        title_font=dict(size=24),
        height=900,
        width=1200,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text=bpp_col, row=1, col=1)
    fig.update_yaxes(title_text='PSNR (dB)', row=1, col=1)
    fig.update_xaxes(title_text='Model', row=1, col=2)
    fig.update_yaxes(title_text='Average PSNR (dB)', row=1, col=2)
    fig.update_xaxes(title_text=bpp_col, row=2, col=1)
    fig.update_yaxes(title_text='PSNR (dB)', row=2, col=1)
    
    # Add a footer note with data source
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        text=f"Data source: {os.path.basename(csv_file)}",
        showarrow=False,
        font=dict(size=10)
    )
    
    # Save dashboard
    output_file = os.path.splitext(csv_file)[0] + '_dashboard.html'
    fig.write_html(output_file)
    print(f"Dashboard saved as {output_file}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create comprehensive dashboard from CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    fig = create_dashboard(args.csv_file)
    # fig.show() # Uncomment to display in browser