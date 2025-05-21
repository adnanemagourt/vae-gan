import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def create_basic_plot(csv_file):
    # Read data from CSV file
    df = pd.read_csv(csv_file)
    
    # Set styling
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Get x-axis column (BPP) and model columns
    x_col = df.columns[0]
    model_cols = df.columns[1:]
    
    # Create line plots for each model
    for model in model_cols:
        plt.plot(df[x_col], df[model], marker='o', linewidth=2.5, markersize=8, label=model)
    
    # Set labels and title
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel('PSNR (dB)', fontsize=14)
    plt.title('VAE-GAN Models Performance Comparison', fontsize=18)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Customize grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotations for min and max values
    for model in model_cols:
        # Find max value
        max_val = df[model].max()
        max_idx = df[model].idxmax()
        max_x = df[x_col][max_idx]
        
        # Annotate max value
        plt.annotate(f'{max_val:.2f}dB', 
                    xy=(max_x, max_val),
                    xytext=(max_x, max_val+0.3),
                    ha='center',
                    arrowprops=dict(arrowstyle='->'))
    
    # Save the figure
    output_file = os.path.splitext(csv_file)[0] + '_plot.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved as {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create visualization from CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    create_basic_plot(args.csv_file)