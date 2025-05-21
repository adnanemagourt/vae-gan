import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def plot_enhanced_comparison(csv_file):
    # Read the CSV data
    df = pd.read_csv(csv_file)
    
    # Set the style with a more modern look
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Get the column names
    bpp_col = df.columns[0]  # Assuming first column is BPP
    model_cols = df.columns[1:]  # All other columns are models
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [2, 1]})
    
    # Define custom colors - create more colors if needed
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', 
              '#1abc9c', '#d35400', '#34495e', '#7f8c8d', '#c0392b']
    colors = colors[:len(model_cols)]  # Use only as many colors as needed
    
    # Plot each model on the main graph
    for i, model in enumerate(model_cols):
        ax1.plot(df[bpp_col], df[model], marker='o', 
                linewidth=3, markersize=8, label=model, color=colors[i])
        
        # Fill the area under each curve
        ax1.fill_between(df[bpp_col], df[model], 
                        alpha=0.1, color=colors[i])
    
    # Set the title and labels with custom styling
    ax1.set_title('Performance Metrics vs. Bits per Pixel', fontsize=18, pad=20)
    ax1.set_xlabel(bpp_col, fontsize=14, labelpad=10)
    ax1.set_ylabel('Performance Metric', fontsize=14, labelpad=10)
    
    # Customize the grid
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Add a legend with custom styling
    ax1.legend(fontsize=12, framealpha=0.8, loc='best')
    
    # Set the axis limits
    y_min = df[model_cols].min().min()
    y_max = df[model_cols].max().max()
    padding = (y_max - y_min) * 0.05
    ax1.set_ylim(y_min - padding, y_max + padding)
    
    # Create a bar chart on the second subplot showing the range for each model
    model_ranges = []
    model_names = []
    
    for model in model_cols:
        model_range = df[model].max() - df[model].min()
        model_ranges.append(model_range)
        model_names.append(model)
    
    # Create a bar chart
    bars = ax2.bar(model_names, model_ranges, color=colors, alpha=0.7)
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11)
    
    # Set title and labels
    ax2.set_title('Performance Range (Max-Min)', fontsize=18, pad=20)
    ax2.set_ylabel('Range', fontsize=14, labelpad=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Rotate x-axis labels if they're too long
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_file = 'gan_comparison_advanced.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Enhanced plot saved as {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create enhanced VAE-GAN comparison plots from CSV file.')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    plot_enhanced_comparison(args.csv_file)