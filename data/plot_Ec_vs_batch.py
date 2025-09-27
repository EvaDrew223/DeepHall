import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rcParams
import matplotlib.cm as cm
import pickle

rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'STIXGeneral'

# Physical parameters
N = 8
Q = 16/2
nu = 2/5
kappa = 1
N_layer = 2

# Generate x values from the formula: int([1, 2, 3, 4, 5, 6, 7, 8]*8*3*5*7)
base_values = [1, 2, 3, 4, 5, 6, 7, 8]
multiplier = 8 * 3 * 5 * 7  # = 840
x_values = [int(val * multiplier) for val in base_values]
print(f"GPU configurations to process: {x_values}")

# Base path
base_path = Path(__file__).parent / "seed_1758632847/tillicum/vs_batch_size"

# Smoothing parameters
window = 500

# Function to smooth data with IQR outlier removal
def smooth_data(energy, window_size):
    smoothed = np.empty_like(energy, dtype=float)
    for i in range(len(energy)):
        start = max(0, i - window_size + 1)
        w = energy[start:i + 1]
        # IQR-based outlier removal within the window
        q1 = np.nanpercentile(w, 25)
        q3 = np.nanpercentile(w, 75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        w_filtered = w[(w >= low) & (w <= high)]
        if w_filtered.size == 0:
            w_filtered = w  # fallback if all were filtered
        smoothed[i] = np.nanmean(w_filtered)
    return smoothed

# Data processing and saving
def process_and_save_data():
    """Process all data and save to pickle file"""
    processed_data = {}
    
    for i, x in enumerate(x_values):
        dict_path = base_path / f"DeepHall_n{N}l{int(2*Q)}_NN{N_layer}L_{x}GPU8"
        csv_path = dict_path / "train_stats.csv"
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                
                # Ensure numeric types
                steps = df["step"].astype(int).to_numpy()
                energy = df["energy"].astype(float).to_numpy()
                
                # Smooth the data
                smoothed = smooth_data(energy, window)
                
                # Transform to Ec/N units
                smoothed_plot = (smoothed - N**2/2/np.sqrt(Q) - N/2/kappa) * np.sqrt(2*Q*nu/N) / N
                
                # Store processed data
                processed_data[x] = {
                    'steps': steps,
                    'smoothed_plot': smoothed_plot,
                    'config_name': f"{x}GPU8"
                }
                
                print(f"Successfully processed: {x}GPU8")
                
            except Exception as e:
                print(f"Error processing {x}GPU8: {e}")
        else:
            print(f"File not found: {csv_path}")
    
    # Save processed data
    data_file = base_path / f"plotdata_EcVSbatch_n{N}l{int(Q*2)}_NN{N_layer}L.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Processed data saved to: {data_file}")
    
    return processed_data

# Load processed data
def load_processed_data():
    """Load processed data from pickle file"""
    # First check for the existing plotdata file
    existing_data_file = base_path / f"plotdata_EcVSbatch_n{N}l{int(Q*2)}_NN{N_layer}L.pkl"
    
    if existing_data_file.exists():
        with open(existing_data_file, 'rb') as f:
            processed_data = pickle.load(f)
        print(f"Loaded existing processed data from: {existing_data_file}")
        return processed_data
    
    # Fallback to the old filename
    data_file = base_path / f"plotdata_EcVSbatch_n{N}l{int(Q*2)}_NN{N_layer}L.pkl"
    
    if data_file.exists():
        with open(data_file, 'rb') as f:
            processed_data = pickle.load(f)
        print(f"Loaded processed data from: {data_file}")
        return processed_data
    else:
        print("No processed data found. Processing data...")
        return process_and_save_data()

# Plotting function
def create_plot(colormap_name='plasma'):
    """Create plot with specified colormap"""
    # Load processed data
    processed_data = load_processed_data()
    
    if not processed_data:
        print("No data to plot!")
        return
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Get colormap
    cmap = cm.get_cmap(colormap_name)
    colors = cmap(np.linspace(0, 1, len(processed_data)))
    
    # Plot data for each configuration
    for i, (x, data) in enumerate(processed_data.items()):
        steps = data['steps']
        smoothed_plot = data['smoothed_plot']
        config_name = data['config_name']
        
        plt.plot(steps, smoothed_plot, color=colors[i], linewidth=2, 
                label=config_name, alpha=0.8)
    
    # Set plot properties
    plt.xlim(-5000, 200000)
    # plt.ylim(-0.4115-0.0004, -0.4095+0.0001)
    plt.ylim(-0.434-0.0005, -0.426 +0.001)

    
    # Set up axes
    ax = plt.gca()
    # major_ticks_y = np.arange(-0.4115, -0.4095 + 1e-12, 0.0005)
    # minor_ticks_y = np.arange(-0.4115-0.0004, -0.4095 + 1e-12, 0.0001)
    major_ticks_y = np.arange(-0.434, -0.426 + 1e-12, 0.002)
    minor_ticks_y = np.arange(-0.434-0.001, -0.426 +0.001 + 1e-12, 0.0005)
    ax.set_yticks(major_ticks_y)
    ax.tick_params(axis='y', which='major', width=1.5, length=5)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    major_ticks_x = np.arange(0, 200000+1, 100000)
    minor_ticks_x = np.arange(0, 200000+ 1e-12, 20000)
    ax.set_xticks(major_ticks_x)
    ax.tick_params(axis='x', which='major', width=1.5, length=5)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.tick_params(axis='both', which='both', labelsize=18)
    
    # Labels and title
    plt.xlabel("training step", fontsize=18)
    plt.ylabel(r"$E_c/N\hbar\omega_c \kappa$", fontsize=18)
    plt.title(f"NN layers = {N_layer}, vs. batch size (ν=2/5)", fontsize=18)
    
    # Legend
    plt.legend(fontsize=14, loc='best')
    
    # Grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = base_path / f"EcVSbatch_n{N}l{int(Q*2)}_NN{N_layer}L.png"
    plt.savefig(str(output_path), bbox_inches='tight', dpi=300)
    print(f"Figure saved to: {output_path}")
    
    plt.close()

    # Show summary
    print(f"\nSummary:")
    print(f"Total configurations plotted: {len(processed_data)}")
    print(f"Valid configurations: {list(processed_data.keys())}")
    print(f"Colormap used: {colormap_name}")
    
    # plt.show()

# Main execution
# %%
if __name__ == "__main__":
    # Process data first (only if needed)
    processed_data = load_processed_data()
    
    # Create plots with different colormaps
    colormaps = ['plasma_r']#, 'viridis', 'tab10', 'Set1', 'Dark2']
    
    for cmap in colormaps:
        print(f"\nCreating plot with {cmap} colormap...")
        create_plot(cmap)
    
    print("\nAll plots created!")

# %%
