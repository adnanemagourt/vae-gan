import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def create_enhanced_plot(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Set aesthetic parameters
    sns.set(style="ticks", context="talk")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [2, 1]})
    
    # Get the column names
    bpp_col = df.columns[0]  # BPP column
    model_cols = df.columns[1:]  # Model columns
    
    # Color palette for models
    palette = sns.color_palette("muted", len(model_cols))
    
    # Line plot on the left
    for i, model in enumerate(model_cols):
        # Create the line plot
        sns.lineplot(x=bpp_col, y=model, data=df, ax=ax1, marker='o', 
                    markersize=10, linewidth=3, label=model, color=palette[i])
        
        # Add light colored region around each line
        ax1.fill_between(df[bpp_col], df[model], min(df[model_cols].min()) - 0.5, 
                        alpha=0.1, color=palette[i])
    
    # Set left subplot properties
    ax1.set_title('PSNR vs Bits per Pixel for Different VAE-GAN Models', fontsize=16)
    ax1.set_xlabel(bpp_col, fontsize=14)
    ax1.set_ylabel('PSNR (dB)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create bar chart on the right showing improvement from lowest BPP to highest
    improvements = []
    
    for model in model_cols:
        # Calculate improvement from lowest to highest BPP
        start_val = df[model].iloc[0]
        end_val = df[model].iloc[-1]
        improvement = end_val - start_val
        improvements.append(improvement)
    
    # Create the bar plot
    bars = ax2.bar(model_cols, improvements, color=palette, alpha=0.8)
    
    # Add data labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'+{height:.2f}dB', ha='center', va='bottom', fontsize=12)
    
    # Set right subplot properties
    ax2.set_title('PSNR Improvement\n(Lowest to Highest BPP)', fontsize=16)
    ax2.set_ylabel('PSNR Improvement (dB)', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if needed
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')
    
    # Add overall annotation
    plt.figtext(0.5, 0.01, f"Data source: {os.path.basename(csv_file)}", 
                ha='center', fontsize=10, style='italic')
    
    # Save the figure
    output_file = os.path.splitext(csv_file)[0] + '_enhanced_plot.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Enhanced plot saved as {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create enhanced visualization from CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    create_enhanced_plot(args.csv_file)