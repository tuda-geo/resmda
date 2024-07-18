#%%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pickle

# File paths
nc_file_path = '/samoa/data/smrserraoseabr/resmda/examples/resmda-darts/darts-geomech-runnable_working_new_darts/runs/ccus_64x64x1/DARTS_simulation_realization_2.nc'
pkl_file_path = '/samoa/data/smrserraoseabr/resmda/examples/resmda-darts/darts-geomech-runnable_working_new_darts/runs/ccus_64x64x1/DARTS_simulation_realization_2_observations.pkl'

# Load the NetCDF file
ds = xr.open_dataset(nc_file_path)

# Load the PKL file
with open(pkl_file_path, 'rb') as f:
    observations = pickle.load(f)

# Function to create and display a 2D map plot
def plot_2d_map(ds, var_name, title):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{title} over time', fontsize=16)

    for i, time in enumerate([0, 3, 7, 11]):  # Plot for different time steps
        ax = axes[i//2, i%2]
        ds[var_name].isel(time=time, Z=0).plot(ax=ax, cmap='viridis')
        ax.set_title(f'Time step: {ds.time.values[time]:.2f}')

    plt.tight_layout()
    plt.show()

# Function to create and display a line plot
def plot_line(ds, var_name, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ds[var_name].isel(Y=32, Z=0).mean(dim='X').plot(ax=ax)
    ax.set_title(f'{title} - Mean along X-axis at Y=32')
    ax.set_xlabel('Time')
    ax.set_ylabel(var_name)
    plt.tight_layout()
    plt.show()

# Function to create and display a heatmap
def plot_heatmap(ds, var_name, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    ds[var_name].mean(dim='time').plot(ax=ax, cmap='viridis')
    ax.set_title(f'{title} - Mean over time')
    plt.tight_layout()
    plt.show()

# Updated function to plot observation points
def plot_observations(obs_data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for point, data in obs_data['dict'].items():
        values = data['data']
        if values.ndim == 2:
            values = values.squeeze()
        ax.plot(range(len(values)), values, label=f'{point}')
    
    ax.set_title(title)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Function to analyze and print observation statistics
def analyze_observations(obs_data, title):
    print(f"\nAnalysis of {title}:")
    for point, data in obs_data['dict'].items():
        values = data['data']
        if values.ndim == 2:
            values = values.squeeze()
        print(f"  {point}:")
        print(f"    Shape: {values.shape}")
        print(f"    Mean: {np.mean(values)}")
        print(f"    Max: {np.max(values)}")
        print(f"    Min: {np.min(values)}")
        print(f"    Standard deviation: {np.std(values)}")

# Create plots for pressure, temperature, U_z, and H2O
variables = ['pressure', 'TEMPERATURE', 'U_z', 'H2O']
for var in variables:
    plot_2d_map(ds, var, var.capitalize())
    plot_line(ds, var, var.capitalize())
    plot_heatmap(ds, var, var.capitalize())

# Plot and analyze observations
plot_observations(observations['U_z'], 'U_z Observations')
analyze_observations(observations['U_z'], 'U_z Observations')

plot_observations(observations['BHP'], 'BHP Observations')
analyze_observations(observations['BHP'], 'BHP Observations')

# Create and display a correlation heatmap
correlation_vars = ['pressure', 'TEMPERATURE', 'U_z', 'H2O', 'Perm', 'Por']
corr_data = ds[correlation_vars].to_array().values
corr_matrix = np.corrcoef(corr_data.reshape(len(correlation_vars), -1))

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(len(correlation_vars)))
ax.set_yticks(range(len(correlation_vars)))
ax.set_xticklabels(correlation_vars, rotation=45, ha='right')
ax.set_yticklabels(correlation_vars)
plt.colorbar(im)
ax.set_title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Close the dataset
ds.close()

print("All plots have been displayed.")


# Function to flatten all observation data
def flatten_all_observations(observations):
    all_data = []
    for obs_type in observations.values():
        for point_data in obs_type['dict'].values():
            all_data.append(point_data['data'].flatten())
    return np.concatenate(all_data)

# Generate flattened array of all observations
flattened_observations = flatten_all_observations(observations)