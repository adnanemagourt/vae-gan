import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
import os

def create_interactive_plot(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get column names
    bpp_col = df.columns[0]  # BPP column
    model_cols = df.columns[1:]  # Model columns
    
    # Create a Plotly figure with two subplots
    fig = make_subplots(rows=1, cols=2, 
                       specs=[[{"type": "scatter"}, {"type": "bar"}]],
                       column_widths=[0.7, 0.3],
                       subplot_titles=('PSNR vs Bits per Pixel', 'Rate-Distortion Performance'))
    
    # Custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = colors[:len(model_cols)]  # Use only as many colors as needed
    
    # Create the line plot for each model
    for i, model in enumerate(model_cols):
        # Add line plot
        fig.add_trace(
            go.Scatter(
                x=df[bpp_col],
                y=df[model],
                mode='lines+markers',
                name=model,
                line=dict(color=colors[i], width=3),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # Calculate rate-distortion efficiency (PSNR divided by BPP)
        # Using the middle point for a fair comparison
        mid_idx = len(df) // 2
        rd_efficiency = df[model].iloc[mid_idx] / df[bpp_col].iloc[mid_idx]
        
        # Add bar chart for rate-distortion efficiency
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[rd_efficiency],
                name=f"{model} R-D",
                marker_color=colors[i],
                showlegend=False,
                text=[f'{rd_efficiency:.2f}'],
                textposition='auto'
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive VAE-GAN Models PSNR Analysis',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14)
        ),
        height=700,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text=bpp_col, row=1, col=1)
    fig.update_yaxes(title_text='PSNR (dB)', row=1, col=1)
    fig.update_yaxes(title_text='PSNR/BPP Ratio', row=1, col=2)
    
    # Add annotations
    for i, model in enumerate(model_cols):
        # Find max PSNR
        max_val = df[model].max()
        max_idx = df[model].idxmax()
        max_x = df[bpp_col][max_idx]
        
        # Add annotation
        fig.add_annotation(
            x=max_x,
            y=max_val,
            text=f"Max: {max_val:.2f}dB",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=colors[i],
            ax=0,
            ay=-40,
            bgcolor="white",
            opacity=0.8,
            row=1, col=1
        )
    
    # Save interactive HTML
    output_file = os.path.splitext(csv_file)[0] + '_interactive.html'
    fig.write_html(output_file)
    print(f"Interactive plot saved as {output_file}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create interactive visualization from CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    fig = create_interactive_plot(args.csv_file)
    # fig.show() # Uncomment to display in browser