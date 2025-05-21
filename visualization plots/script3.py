import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import argparse
import os

def create_interactive_plot(csv_file):
    # Read the CSV data
    df = pd.read_csv(csv_file)
    
    # Get the column names
    bpp_col = df.columns[0]  # Assuming first column is BPP
    model_cols = df.columns[1:]  # All other columns are models
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Custom colors for the lines - add more if needed
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    colors = colors[:len(model_cols)]  # Use only as many colors as needed
    
    # Add traces for each model
    for i, model in enumerate(model_cols):
        fig.add_trace(go.Scatter(
            x=df[bpp_col],
            y=df[model],
            mode='lines+markers',
            name=model,
            line=dict(color=colors[i], width=3),
            marker=dict(size=10, symbol='circle')
        ))
    
    # Update layout with custom styling
    fig.update_layout(
        title={
            'text': 'Interactive Comparison of VAE-GAN Models',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title={
            'text': bpp_col,
            'font': dict(size=18)
        },
        yaxis_title={
            'text': 'Performance Metric',
            'font': dict(size=18)
        },
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            font=dict(size=16)
        ),
        plot_bgcolor='rgba(240,240,240,0.2)',
        paper_bgcolor='white',
        hovermode='closest'
    )
    
    # Add grid lines
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(200,200,200,0.2)',
        zeroline=False
    )
    
    # Set axis ranges with padding
    y_min = df[model_cols].min().min() * 0.95
    y_max = df[model_cols].max().max() * 1.05
    fig.update_yaxes(range=[y_min, y_max])
    
    # Add annotations for the highest value of each model
    for i, model in enumerate(model_cols):
        max_idx = df[model].idxmax()
        fig.add_annotation(
            x=df[bpp_col][max_idx],
            y=df[model][max_idx],
            text=f"Max: {df[model][max_idx]:.3f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=20,
            ay=-30,
            font=dict(size=12, color=colors[i]),
            bordercolor=colors[i],
            borderwidth=2,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
    
    # Get the filename without extension for output
    output_base = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = f"{output_base}_interactive.html"
    
    # Save as HTML file
    pio.write_html(fig, output_file)
    print(f"Interactive plot saved as {output_file}")
    
    # Return the figure object (useful in Jupyter notebooks)
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create interactive VAE-GAN comparison plots from CSV file.')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    fig = create_interactive_plot(args.csv_file)
    # In non-interactive environments, you can use fig.show() to display the plot