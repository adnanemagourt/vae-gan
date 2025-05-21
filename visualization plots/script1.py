import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_vae_gan_comparison(csv_file):
    # Set the style for the plot
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("notebook", font_scale=1.4)
    
    # Read the CSV data
    df = pd.read_csv(csv_file)
    
    # Create a figure with a specific size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get the column names
    bpp_col = df.columns[0]  # Assuming first column is BPP
    model_cols = df.columns[1:]  # All other columns are models
    
    # Plot each model as a separate line with markers
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    for i, model in enumerate(model_cols):
        marker = markers[i % len(markers)]  # Cycle through markers if more models
        ax.plot(df[bpp_col], df[model], '-', marker=marker, linewidth=2.5, 
                markersize=8, label=model)
    
    # Set the title and labels
    ax.set_title('Comparison of VAE-GAN Models', fontsize=20, pad=20)
    ax.set_xlabel(bpp_col, fontsize=16)
    ax.set_ylabel('Performance Metric', fontsize=16)
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    ax.legend(loc='best', fontsize=14, frameon=True, framealpha=0.9)
    
    # Set y-axis limits slightly beyond the data range
    min_val = df[model_cols].min().min()
    max_val = df[model_cols].max().max()
    padding = (max_val - min_val) * 0.1
    ax.set_ylim(min_val - padding, max_val + padding)
    
    # Adjust the layout
    plt.tight_layout()
    
    # Save the figure
    output_file = 'gan_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot VAE-GAN comparison from CSV file.')
    parser.add_argument('csv_file', help='Path to the CSV file containing the data')
    args = parser.parse_args()
    
    plot_vae_gan_comparison(args.csv_file)