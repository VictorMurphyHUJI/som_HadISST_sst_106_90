import xarray as xr
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import os
import pandas as pd
import csv

from scipy.spatial.distance import cdist

from scipy import stats
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from scipy import spatial
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

from matplotlib import gridspec

import pickle
import json
import traceback

from som_pattern_clustering import (
    apply_clustering_to_som,
    analyze_cluster_occurrences,
    plot_cluster_frequency_over_time,
    create_cluster_composites,
    get_nao_index_data,
    correlate_clusters_with_indices,
    create_cluster_network,
    create_physical_interpretation_summary,
    create_climate_mode_correlations
)

import argparse

from cluster_index_generator import create_cluster_indices_for_applications
from som_HadISST_sst_106_81_lead_lag_001 import analyze_cluster_z500_lead_lag_with_indices



######################################################################################################################
######################################################################################################################

def load_and_preprocess_sst(file_path):
    print("Loading dataset...")
    ds = xr.open_dataset(file_path)

    print(f"Initial data shape: {ds.sst.shape}")
    print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")
    print(f"Latitude range: {ds.latitude.min().values} to {ds.latitude.max().values}")
    print(f"Longitude range: {ds.longitude.min().values} to {ds.longitude.max().values}")

    # Print some latitude values for debugging
    print("\nFirst few latitude values:")
    print(ds.latitude.values[:10])

    # Select time range 1950-present
    ds = ds.sel(time=slice('1950-01-01', None))
    print(f"\nShape after time selection: {ds.sst.shape}")

    # Convert latitude array to numpy for easier manipulation
    lats = ds.latitude.values
    lons = ds.longitude.values

    # Find indices for latitude range (0 to 60N)
    # Note: Since latitudes are in descending order (90 to -90), we need to flip the comparison
    lat_indices = np.where((lats >= 0) & (lats <= 60))[0]
    lat_min_idx = lat_indices[0]
    lat_max_idx = lat_indices[-1]

    # Find indices for longitude range (-80 to -5)
    lon_indices = np.where((lons >= -80) & (lons <= -5))[0]
    lon_min_idx = lon_indices[0]
    lon_max_idx = lon_indices[-1]

    print(f"\nSelected latitude indices: {lat_min_idx} to {lat_max_idx}")
    print(f"Corresponding latitudes: {lats[lat_min_idx]} to {lats[lat_max_idx]}")
    print(f"Selected longitude indices: {lon_min_idx} to {lon_max_idx}")
    print(f"Corresponding longitudes: {lons[lon_min_idx]} to {lons[lon_max_idx]}")

    # Select the region using integer indexing
    ds = ds.isel(
        latitude=slice(lat_min_idx, lat_max_idx + 1),
        longitude=slice(lon_min_idx, lon_max_idx + 1)
    )

    print(f"\nShape after spatial selection: {ds.sst.shape}")
    print(f"Selected latitude range: {ds.latitude.values.min()} to {ds.latitude.values.max()}")
    print(f"Selected longitude range: {ds.longitude.values.min()} to {ds.longitude.values.max()}")

    if ds.sst.size == 0:
        raise ValueError("No data found after spatial and temporal selection")

    # Select DJF months
    time_coords = pd.to_datetime(ds.time.values)
    is_djf = (time_coords.month == 12) | (time_coords.month == 1) | (time_coords.month == 2)
    ds = ds.isel(time=is_djf)
    print(f"\nShape after DJF selection: {ds.sst.shape}")

    if ds.sst.size == 0:
        raise ValueError("No data found after DJF selection")

    # Calculate climatology and anomalies
    print("\nCalculating climatology and anomalies...")
    monthly_clim = ds.groupby('time.month').mean('time')
    anomalies = ds.groupby('time.month') - monthly_clim

    # Detrend the anomalies
    print("Detrending anomalies...")
    time_idx = np.arange(len(ds.time))
    trend = anomalies.sst.polyfit('time', deg=1)
    trend_fitted = xr.polyval(coord=anomalies.time, coeffs=trend.polyfit_coefficients)
    detrended_anomalies = anomalies.sst - trend_fitted

    # Create new dataset with detrended DJF anomalies
    ds_processed = xr.Dataset({
        'sst': detrended_anomalies
    })

    print(f"\nFinal processed data shape: {ds_processed.sst.shape}")

    # Check for all-NaN slices
    if np.all(np.isnan(ds_processed.sst.values)):
        raise ValueError("All values are NaN in processed dataset")

    return ds_processed


def prepare_som_input(ds):
    print("\nPreparing data for SOM...")
    sst_array = ds.sst.values
    print(f"Input array shape: {sst_array.shape}")

    # Reshape to 2D array (samples x features)
    n_times, n_lats, n_lons = sst_array.shape
    X = sst_array.reshape(n_times, n_lats * n_lons)
    print(f"Reshaped array: {X.shape}")

    # Handle NaN values
    nan_cols = np.all(np.isnan(X), axis=0)
    if np.all(nan_cols):
        raise ValueError("All columns contain only NaN values")

    # Replace NaN values with column means
    col_means = np.nanmean(X, axis=0)
    for col in range(X.shape[1]):
        nan_mask = np.isnan(X[:, col])
        if np.all(nan_mask):
            X[:, col] = 0  # Set to 0 if entire column is NaN
        else:
            X[nan_mask, col] = col_means[col]

    # Normalize
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)

    print(f"Final prepared data shape: {X.shape}")
    return X


#######################################################################################################################
#######################################################################################################################


def train_som(X, map_size=(7, 4), n_iterations=10000, random_seed=42):
    """
    Train SOM with fixed random seed for reproducibility

    Parameters:
    -----------
    X : numpy.ndarray
        Input data array of shape (n_samples, n_features)
    map_size : tuple
        Size of the SOM grid (height, width)
    n_iterations : int
        Number of training iterations
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    MiniSom
        Trained SOM object
    """
    print(f"\nTraining SOM with input shape: {X.shape}")
    n_features = X.shape[1]

    # Set random seed for numpy operations
    np.random.seed(random_seed)

    # Initialize SOM with fixed random seed
    som = MiniSom(map_size[0], map_size[1], n_features,
                  sigma=1, learning_rate=0.7,
                  neighborhood_function='bubble',  # or 'triangle' instead of 'gaussian'
                  random_seed=random_seed)

    # Initialize weights with fixed random seed
    som.random_weights_init(X)

    # Train SOM
    som.train(X, n_iterations, verbose=True)

    return som


#######################################################################################################################
#######################################################################################################################
######### 10000, 20000, 50000, 75000, 100000, 200000, 500000 ############ 100000, 200000,

def test_som_iterations(X, iterations_list=[20000, 25000, 50000], map_size=(7, 4), random_seed=42):
    """
    Test SOM training with different numbers of iterations

    Parameters:
    -----------
    X : numpy.ndarray
        Input data array
    iterations_list : list
        List of iteration numbers to test
    map_size : tuple
        Size of the SOM grid
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing errors for each iteration count
    """
    results = {}

    for n_iterations in iterations_list:
        print(f"\nTraining SOM with {n_iterations} iterations...")

        # Set random seed for numpy operations
        np.random.seed(random_seed)

        # Initialize and train SOM
        som = MiniSom(map_size[0], map_size[1], X.shape[1],
                      sigma=1, learning_rate=0.7,
                      neighborhood_function='bubble',
                      random_seed=random_seed)

        som.random_weights_init(X)
        som.train(X, n_iterations, verbose=True)

        # Calculate quantization error
        error = som.quantization_error(X)
        results[n_iterations] = error
        print(f"Quantization error with {n_iterations} iterations: {error:.4f}")

    return results


#######################################################################################################################
#######################################################################################################################

def plot_som_patterns(som, original_shape, ds, output_dir, pattern_results, n_patterns=9):
    """
    Plot SOM patterns with proper coordinate handling and a single colorbar
    """
    # Get top patterns by variance
    variance_by_pattern = {k: v['relative_variance']
                           for k, v in pattern_results['pattern_variance'].items()}
    top_patterns = sorted(variance_by_pattern.items(),
                          key=lambda x: x[1], reverse=True)[:n_patterns]

    fig = plt.figure(figsize=(15, 15))
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape
    spatial_shape = original_shape[1:]

    # Create coordinate meshgrid
    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

    # Create land mask
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray')

    # Find global min/max for consistent color scaling
    all_patterns = []
    for pattern_num, _ in top_patterns:
        i, j = (pattern_num - 1) // n_cols, (pattern_num - 1) % n_cols
        pattern = weights[i, j].reshape(spatial_shape)
        all_patterns.append(pattern)

    vmin = min(p.min() for p in all_patterns)
    vmax = max(p.max() for p in all_patterns)
    abs_max = max(abs(vmin), abs(vmax))

    # Plot patterns based on variance ranking
    for idx, (pattern_num, variance) in enumerate(top_patterns):
        row = idx // 3
        col = idx % 3

        # Convert pattern number to grid indices
        i, j = (pattern_num - 1) // n_cols, (pattern_num - 1) % n_cols

        ax = fig.add_subplot(3, 3, idx + 1, projection=ccrs.PlateCarree())
        pattern = weights[i, j].reshape(spatial_shape)

        # Plot SST data with consistent color scaling
        im = ax.pcolormesh(lons, lats, pattern,
                           transform=ccrs.PlateCarree(),
                           cmap='RdBu_r',
                           vmin=-abs_max,
                           vmax=abs_max,
                           shading='auto')

        # Add land feature and coastlines
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        ax.set_title(f'DJF Pattern {pattern_num}\nVar Explained: {variance:.1f}%')

        # Set extent to match data domain
        ax.set_extent([ds.longitude.min(), ds.longitude.max(),
                       ds.latitude.min(), ds.latitude.max()],
                      crs=ccrs.PlateCarree())

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    plt.colorbar(im, cax=cbar_ax, label='Normalized SST Anomaly')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for colorbar
    plt.savefig(os.path.join(output_dir, 'som_patterns_djf.png'),
                dpi=300, bbox_inches='tight')

    # Save individual patterns
    for pattern_num, variance in top_patterns:
        fig_single = plt.figure(figsize=(8, 6))
        ax = fig_single.add_subplot(111, projection=ccrs.PlateCarree())

        i, j = (pattern_num - 1) // n_cols, (pattern_num - 1) % n_cols
        pattern = weights[i, j].reshape(spatial_shape)

        im = ax.pcolormesh(lons, lats, pattern,
                           transform=ccrs.PlateCarree(),
                           cmap='RdBu_r',
                           vmin=-abs_max,
                           vmax=abs_max,
                           shading='auto')

        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        plt.colorbar(im, ax=ax, label='Normalized SST Anomaly')
        ax.set_title(f'DJF Pattern {pattern_num}\nVar Explained: {variance:.1f}%')

        ax.set_extent([ds.longitude.min(), ds.longitude.max(),
                       ds.latitude.min(), ds.latitude.max()],
                      crs=ccrs.PlateCarree())

        plt.savefig(os.path.join(output_dir, f'pattern_{pattern_num}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig_single)

    return fig


#######################################################################################################################
#######################################################################################################################

def analyze_pattern_statistics(som, X, original_shape):
    """Calculate basic statistics for each pattern"""
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    # Reshape patterns
    patterns_2d = weights.reshape(n_rows * n_cols, n_features)
    patterns_3d = np.array([p.reshape(original_shape[1:]) for p in patterns_2d])

    stats = {
        'spatial_variance': [],
        'pattern_amplitude': [],
        'pattern_std': [],
        'activation_frequency': []
    }

    # Calculate statistics for each pattern
    for i in range(len(patterns_2d)):
        pattern = patterns_2d[i]
        stats['spatial_variance'].append(np.var(pattern))
        stats['pattern_amplitude'].append(np.max(pattern) - np.min(pattern))
        stats['pattern_std'].append(np.std(pattern))
        # Calculate how often this pattern is the best match
        winners = np.array([som.winner(x) for x in X])
        freq = np.mean(winners == np.unravel_index(i, (n_rows, n_cols)))
        stats['activation_frequency'].append(freq)

    return stats


########################################################################################################################
########################################################################################################################

def calculate_corrected_variance(som, X, bmu_indices=None):
    """
    Calculate corrected variance explained by SOM patterns.

    Parameters:
    -----------
    som : MiniSom
        Trained SOM object
    X : numpy.ndarray
        Input data array (n_samples x n_features)
    bmu_indices : numpy.ndarray, optional
        Pre-computed BMU indices. If None, will be calculated.

    Returns:
    --------
    dict
        Dictionary containing corrected variance metrics
    """
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    # Get BMUs if not provided
    if bmu_indices is None:
        bmu_indices = np.array([som.winner(x) for x in X])

    # Get BMU weights for all data points
    bmu_weights = np.array([weights[i, j] for i, j in bmu_indices])

    # Calculate total variance
    X_mean = np.mean(X, axis=0, keepdims=True)
    total_ss = np.sum(np.square(X - X_mean))
    total_var = total_ss / (X.shape[0] - 1)

    # Calculate residual variance (unexplained variance)
    residual = X - bmu_weights
    residual_ss = np.sum(np.square(residual))
    residual_var = residual_ss / (X.shape[0] - 1)

    # Calculate total explained variance ratio
    total_explained_var = 1 - (residual_var / total_var)

    # Initialize pattern-specific variance dictionary
    pattern_vars = {}

    # Calculate individual pattern contributions
    total_reduction = 0
    for i in range(n_rows):
        for j in range(n_cols):
            pattern_idx = i * n_cols + j + 1
            pattern_mask = (bmu_indices[:, 0] == i) & (bmu_indices[:, 1] == j)
            pattern_data = X[pattern_mask]

            if len(pattern_data) > 0:
                # Calculate variance reduction when using this pattern
                pattern_weights = weights[i, j]
                pattern_reduction = np.sum(np.square(pattern_data - X_mean)) - \
                                    np.sum(np.square(pattern_data - pattern_weights))

                # Weight by pattern frequency
                pattern_weight = len(pattern_data) / len(X)
                weighted_reduction = pattern_reduction * pattern_weight

                # Store pattern statistics
                pattern_vars[pattern_idx] = {
                    'variance_reduction': pattern_reduction,
                    'weighted_reduction': weighted_reduction,
                    'relative_variance': (weighted_reduction / total_ss) * 100,
                    'frequency': pattern_weight,
                    'n_samples': len(pattern_data)
                }

                total_reduction += weighted_reduction
            else:
                pattern_vars[pattern_idx] = {
                    'variance_reduction': 0,
                    'weighted_reduction': 0,
                    'relative_variance': 0,
                    'frequency': 0,
                    'n_samples': 0
                }

    # Normalize pattern contributions to sum to total explained variance
    scaling_factor = total_explained_var * total_var * 100 / total_reduction
    for pattern_idx in pattern_vars:
        pattern_vars[pattern_idx]['relative_variance'] *= scaling_factor

    quality_metrics = {
        'quantization_error': som.quantization_error(X),
        'topographic_error': som.topographic_error(X),
        'neighborhood_preservation': calculate_neighborhood_preservation(som, X)
    }

    return {
        'total_variance': total_var,
        'residual_variance': residual_var,
        'total_explained_variance_ratio': total_explained_var,
        'total_explained_variance_percent': total_explained_var * 100,
        'pattern_variances': pattern_vars,
        'quality_metrics': quality_metrics
    }


########################################################################################################################
########################################################################################################################

def calculate_neighborhood_preservation(som, X, k=5):
    """Calculate neighborhood preservation metric"""
    # Get pairwise distances in input space
    input_distances = cdist(X, X)

    # Get BMUs for all points
    bmu_indices = np.array([som.winner(x) for x in X])
    bmu_coords = np.array([[i, j] for i, j in bmu_indices])

    # Get pairwise distances in SOM space
    som_distances = cdist(bmu_coords, bmu_coords)

    # Calculate neighborhood preservation
    preservation_scores = []
    for idx in range(len(X)):
        input_neighbors = set(np.argsort(input_distances[idx])[1:k + 1])
        som_neighbors = set(np.argsort(som_distances[idx])[1:k + 1])
        preservation = len(input_neighbors.intersection(som_neighbors)) / k
        preservation_scores.append(preservation)

    return np.mean(preservation_scores)


########################################################################################################################
########################################################################################################################

def print_variance_summary(variance_results):
    """
    Print a summary of the variance analysis results
    """
    print(f"\nTotal variance explained: {variance_results['total_explained_variance_percent']:.2f}%")
    print("\nTop patterns by variance explained:")

    # Sort patterns by variance explained
    pattern_vars = variance_results['pattern_variances']
    sorted_patterns = sorted(pattern_vars.items(),
                             key=lambda x: x[1]['relative_variance'],
                             reverse=True)

    # Print top 10 patterns
    for pattern_idx, stats in sorted_patterns[:10]:
        print(f"\nPattern {pattern_idx}:")
        print(f"- Variance explained: {stats['relative_variance']:.2f}%")
        print(f"- Frequency: {stats['frequency'] * 100:.2f}%")
        print(f"- Sample count: {stats['n_samples']}")


########################################################################################################################
########################################################################################################################

def validate_variance_results(variance_results):
    """
    Validate the variance calculation results
    """
    pattern_vars = variance_results['pattern_variances']
    total_explained = variance_results['total_explained_variance_percent']

    # Sum of individual variances
    sum_individual = sum(stats['relative_variance']
                         for stats in pattern_vars.values())

    print("\nVariance Calculation Validation:")
    print(f"Total explained variance: {total_explained:.2f}%")
    print(f"Sum of pattern variances: {sum_individual:.2f}%")
    print(f"Difference: {abs(total_explained - sum_individual):.4f}%")

    if abs(total_explained - sum_individual) > 0.01:
        print("Warning: Sum of pattern variances differs from total explained variance")


########################################################################################################################
########################################################################################################################

def plot_quality_metrics(variance_results, pattern_results, output_dir):
    """
    Create visualizations for SOM quality metrics using corrected variance calculations.

    Parameters:
    -----------
    variance_results : dict
        Results from calculate_corrected_variance
    pattern_results : dict
        Results from analyze_pattern_occurrences
    output_dir : str
        Directory to save output
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Pattern Frequency Distribution',
            'Quality Metrics',
            'Pattern Variance Distribution',
            'Total Explained Variance'
        )
    )

    # Define base_rows based on the SOM grid dimensions
    n_rows, n_cols = variance_results['pattern_variances'].shape if hasattr(variance_results['pattern_variances'],
                                                                            'shape') else som.get_weights().shape[:2]
    base_rows = n_rows

    # Add metrics if available
    if variance_results:
        metrics_row = base_rows + 1

        # Pattern usage heatmap
        pattern_freq_matrix = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                pattern_idx = i * n_cols + j + 1
                if pattern_idx in pattern_results['pattern_occurrences']:
                    pattern_freq_matrix[i, j] = pattern_results['pattern_occurrences'][pattern_idx]['frequency']

        fig.add_trace(
            go.Heatmap(
                z=pattern_freq_matrix,
                colorscale='Viridis',
                text=[[f'{val:.3f}' for val in row] for row in pattern_freq_matrix],
                texttemplate='%{text}',
                showscale=True,
                colorbar=dict(
                    title="Pattern Usage",
                    len=0.2,
                    thickness=10,
                    x=1.02,
                    y=0.15,
                    tickformat=".3f"
                ),
                name='Pattern Usage'
            ),
            row=metrics_row, col=1
        )

    # 2. Quality metrics bar chart
    key_metrics = {
        'Quantization Error': variance_results['quality_metrics']['quantization_error'],
        'Topographic Error': variance_results['quality_metrics']['topographic_error'],
        'Neighborhood Preservation': variance_results['quality_metrics']['neighborhood_preservation']
    }

    fig.add_trace(
        go.Bar(
            x=list(key_metrics.keys()),
            y=list(key_metrics.values()),
            text=[f'{val:.3f}' for val in key_metrics.values()],
            textposition='auto',
        ),
        row=1, col=2
    )

    # 3. Pattern variance distribution
    pattern_variances = [stats['relative_variance']
                         for stats in pattern_results['pattern_variance'].values()]

    fig.add_trace(
        go.Histogram(
            x=pattern_variances,
            nbinsx=20,
            name='Pattern Variance Distribution'
        ),
        row=2, col=1
    )

    # 4. Total explained variance gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=variance_results['total_explained_variance_percent'],
            title={'text': "Total Explained Variance (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "darkgray"}
                ]
            }
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        showlegend=False,
        title_text="SOM Quality Metrics Analysis"
    )

    # Save the plot
    fig.write_html(os.path.join(output_dir, 'som_quality_metrics.html'))
    return fig


########################################################################################################################
########################################################################################################################

def analyze_pattern_occurrences(som, X, ds):
    """
    Analyze when each pattern occurs and its variance contribution using corrected variance calculation.

    Parameters:
    -----------
    som : MiniSom
        Trained SOM object
    X : numpy.ndarray
        Input data array (n_samples x n_features)
    ds : xarray.Dataset
        Original dataset with time coordinates

    Returns:
    --------
    dict
        Dictionary containing pattern occurrences and variance information
    """
    # Get time coordinates
    times = pd.to_datetime(ds.time.values)

    # Get SOM weights
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    # Get BMUs
    bmu_indices = np.array([som.winner(x) for x in X])
    bmu_weights = np.array([weights[i, j] for i, j in bmu_indices])

    # Calculate mean and total variance
    X_mean = np.mean(X, axis=0, keepdims=True)
    total_ss = np.sum(np.square(X - X_mean))
    total_var = total_ss / (X.shape[0] - 1)

    # Calculate residual variance
    residual = X - bmu_weights
    residual_ss = np.sum(np.square(residual))
    residual_var = residual_ss / (X.shape[0] - 1)

    # Calculate total explained variance
    total_explained_var = 1 - (residual_var / total_var)
    total_explained_percent = total_explained_var * 100

    # Initialize results storage
    results = {
        'pattern_occurrences': {},
        'pattern_variance': {},
        'total_variance_explained': total_explained_percent
    }

    # Calculate pattern-specific contributions
    total_reduction = 0
    for i in range(n_rows):
        for j in range(n_cols):
            pattern_idx = i * n_cols + j + 1
            pattern_mask = (bmu_indices[:, 0] == i) & (bmu_indices[:, 1] == j)
            pattern_times = times[pattern_mask]
            pattern_data = X[pattern_mask]

            # Store occurrence information
            results['pattern_occurrences'][pattern_idx] = {
                'dates': [t.strftime('%Y-%m') for t in pattern_times],
                'count': len(pattern_times),
                'frequency': len(pattern_times) / len(times)
            }

            if len(pattern_data) > 0:
                # Calculate variance reduction
                pattern_weights = weights[i, j]
                pattern_reduction = np.sum(np.square(pattern_data - X_mean)) - \
                                    np.sum(np.square(pattern_data - pattern_weights))

                # Weight by frequency
                pattern_weight = len(pattern_data) / len(X)
                weighted_reduction = pattern_reduction * pattern_weight

                results['pattern_variance'][pattern_idx] = {
                    'absolute_variance': pattern_reduction,
                    'relative_variance': (weighted_reduction / total_ss) * 100
                }

                total_reduction += weighted_reduction
            else:
                results['pattern_variance'][pattern_idx] = {
                    'absolute_variance': 0,
                    'relative_variance': 0
                }

    # Normalize pattern variances to sum to total explained variance
    if total_reduction > 0:
        scaling_factor = total_explained_percent / (total_reduction / total_ss * 100)
        for pattern_idx in results['pattern_variance']:
            results['pattern_variance'][pattern_idx]['relative_variance'] *= scaling_factor

    return results


#######################################################################################################################
#######################################################################################################################

def analyze_pattern_transitions(som, X, ds):
    """
    Analyze transitions between SOM patterns over time

    Parameters:
    -----------
    som : MiniSom
        Trained SOM object
    X : numpy.ndarray
        Input data array (n_samples x n_features)
    ds : xarray.Dataset
        Original dataset with time coordinates

    Returns:
    --------
    dict
        Dictionary containing transition information
    """
    # Get time coordinates in chronological order
    times = pd.to_datetime(ds.time.values)
    sorted_indices = np.argsort(times)

    # Get SOM weights and dimensions
    weights = som.get_weights()
    n_rows, n_cols = weights.shape[:2]
    n_patterns = n_rows * n_cols

    # Get BMUs for all points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert BMU grid indices to pattern numbers (1-based)
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Sort by time
    chronological_patterns = pattern_sequence[sorted_indices]
    chronological_times = times[sorted_indices]

    # Create transition matrix
    transition_matrix = np.zeros((n_patterns, n_patterns))
    for i in range(len(chronological_patterns) - 1):
        from_pattern = chronological_patterns[i] - 1  # Convert to 0-based
        to_pattern = chronological_patterns[i + 1] - 1  # Convert to 0-based
        transition_matrix[from_pattern, to_pattern] += 1

    # Calculate transition probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_probabilities = np.divide(transition_matrix, row_sums,
                                             where=row_sums != 0)
        transition_probabilities[np.isnan(transition_probabilities)] = 0

    # Calculate pattern persistence (how many consecutive times a pattern appears)
    persistence = {}
    current_pattern = chronological_patterns[0]
    streak_length = 1

    for i in range(1, len(chronological_patterns)):
        if chronological_patterns[i] == current_pattern:
            streak_length += 1
        else:
            if current_pattern not in persistence:
                persistence[current_pattern] = []
            persistence[current_pattern].append(streak_length)
            current_pattern = chronological_patterns[i]
            streak_length = 1

    # Add the last streak
    if current_pattern not in persistence:
        persistence[current_pattern] = []
    persistence[current_pattern].append(streak_length)

    # Calculate mean persistence for each pattern
    mean_persistence = {p: np.mean(streaks) for p, streaks in persistence.items()}

    # Analyze seasonal preferences
    monthly_counts = {}
    for i, time in enumerate(chronological_times):
        month = time.month
        pattern = chronological_patterns[i]

        if month not in monthly_counts:
            monthly_counts[month] = {}

        if pattern not in monthly_counts[month]:
            monthly_counts[month][pattern] = 0

        monthly_counts[month][pattern] += 1

    return {
        "pattern_sequence": chronological_patterns,
        "transition_matrix": transition_matrix,
        "transition_probabilities": transition_probabilities,
        "persistence": persistence,
        "mean_persistence": mean_persistence,
        "monthly_counts": monthly_counts,
        "times": chronological_times
    }


#######################################################################################################################
#######################################################################################################################

def plot_pattern_transitions(transition_results, som, output_dir):
    """
    Create visualizations for pattern transitions

    Parameters:
    -----------
    transition_results : dict
        Results from analyze_pattern_transitions
    som : MiniSom
        Trained SOM object
    output_dir : str
        Directory to save output
    """
    # Get transitions
    transition_probs = transition_results["transition_probabilities"]
    n_rows, n_cols = som.get_weights().shape[:2]

    # Create heatmap of transition probabilities
    fig = go.Figure(data=go.Heatmap(
        z=transition_probs,
        x=[f"Pattern {i + 1}" for i in range(transition_probs.shape[1])],
        y=[f"Pattern {i + 1}" for i in range(transition_probs.shape[0])],
        colorscale='Viridis',
        colorbar=dict(title="Transition Probability")
    ))

    fig.update_layout(
        title="Pattern Transition Probabilities",
        xaxis_title="To Pattern",
        yaxis_title="From Pattern",
        height=800,
        width=800
    )

    fig.write_html(os.path.join(output_dir, 'pattern_transitions.html'))

    # Plot mean persistence
    mean_persistence = transition_results["mean_persistence"]
    patterns = list(mean_persistence.keys())
    persistence_values = [mean_persistence[p] for p in patterns]

    fig = go.Figure(data=go.Bar(
        x=[f"Pattern {p}" for p in patterns],
        y=persistence_values
    ))

    fig.update_layout(
        title="Mean Pattern Persistence",
        xaxis_title="Pattern",
        yaxis_title="Mean Consecutive Occurrences",
        height=600,
        width=1000
    )

    fig.write_html(os.path.join(output_dir, 'pattern_persistence.html'))

    # Create monthly pattern distribution
    monthly_counts = transition_results["monthly_counts"]
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Create a plot for seasonal pattern distribution
    fig = make_subplots(rows=1, cols=1)

    for pattern in range(1, n_rows * n_cols + 1):
        pattern_monthly = []
        for month in months:
            if month in monthly_counts and pattern in monthly_counts[month]:
                pattern_monthly.append(monthly_counts[month][pattern])
            else:
                pattern_monthly.append(0)

        fig.add_trace(
            go.Bar(
                name=f"Pattern {pattern}",
                x=month_names,
                y=pattern_monthly
            )
        )

    fig.update_layout(
        barmode='stack',
        title="Monthly Pattern Distribution",
        xaxis_title="Month",
        yaxis_title="Count",
        height=600,
        width=1000
    )

    fig.write_html(os.path.join(output_dir, 'monthly_pattern_distribution.html'))

    return fig


#######################################################################################################################
#######################################################################################################################

def print_pattern_summary(results, top_n=4):
    """
    Print a summary of pattern analysis results
    """
    print(f"\nTotal variance explained by all patterns: {results['total_variance_explained']:.2f}%")

    print("\nTop patterns by variance explained:")
    variance_by_pattern = {k: v['relative_variance']
                           for k, v in results['pattern_variance'].items()}
    top_patterns = sorted(variance_by_pattern.items(),
                          key=lambda x: x[1], reverse=True)[:top_n]

    for pattern, variance in top_patterns:
        occurrences = results['pattern_occurrences'][pattern]
        print(f"\nPattern {pattern}:")
        print(f"- Variance explained: {variance:.2f}%")
        print(f"- Occurrence count: {occurrences['count']}")
        print(f"- Frequency: {occurrences['frequency'] * 100:.2f}%")
        print(f"- Example months: {', '.join(occurrences['dates'][:5])}...")


#######################################################################################################################
#######################################################################################################################

def plot_scree_diagram(variance_results, pattern_results):
    """Create a scree plot showing explained variance by pattern"""
    # Calculate individual explained variances
    individual_variances = [v['relative_variance']
                            for v in pattern_results['pattern_variance'].values()]
    individual_variances.sort(reverse=True)

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(individual_variances)

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add bar plot for individual explained variance
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(individual_variances) + 1)),
            y=individual_variances,
            name='Individual',
            marker_color='rgb(106, 137, 247)',
            yaxis='y'
        )
    )

    # Add line plot for cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance,
            name='Cumulative',
            line=dict(color='red', width=2),
            yaxis='y2'
        )
    )

    # Update layout with secondary y-axis
    fig.update_layout(
        title='Scree Diagram - Explained Variance by Pattern',
        xaxis_title='Pattern Number',
        yaxis_title='Individual Explained Variance (%)',
        yaxis2=dict(
            title='Cumulative Explained Variance (%)',
            overlaying='y',
            side='right'
        ),
        showlegend=True,
        height=600,
        width=1000
    )

    return fig


#######################################################################################################################
#######################################################################################################################

def create_pattern_filters(pattern_results, stats):
    """Create advanced filtering options for patterns"""
    filters = {
        'high_variance': {
            'name': 'High Variance Patterns',
            'description': 'Patterns with above median variance explained',
            'threshold': np.median([v['relative_variance']
                                    for v in pattern_results['pattern_variance'].values()])
        },
        'high_frequency': {
            'name': 'High Frequency Patterns',
            'description': 'Patterns that occur more frequently',
            'threshold': np.median([v['frequency']
                                    for v in pattern_results['pattern_occurrences'].values()])
        },
        'top_patterns': {
            'name': 'Top Contributing Patterns',
            'description': 'Patterns with highest combined impact',
            'threshold': np.median([
                pattern_results['pattern_variance'][k]['relative_variance'] *
                pattern_results['pattern_occurrences'][k]['frequency']
                for k in pattern_results['pattern_variance'].keys()
            ])
        }
    }
    return filters


#######################################################################################################################
#######################################################################################################################

def plot_interactive_patterns(som, original_shape, ds, stats, X, iteration_results=None, variance_results=None):
    """Enhanced interactive plot with quality metrics and temporal information visualization"""
    pattern_results = analyze_pattern_occurrences(som, X, ds)
    weights = som.get_weights()
    n_rows, n_cols = weights.shape[:2]

    # Calculate base rows for patterns
    base_rows = n_rows
    total_rows = base_rows + (1 if variance_results else 0)

    # Create scree plot
    scree_fig = plot_scree_diagram(variance_results, pattern_results)

    # Create pattern filters
    filters = create_pattern_filters(pattern_results, stats)

    # Create subplot titles
    subplot_titles = []
    for i in range(n_rows * n_cols):
        pattern_var = pattern_results['pattern_variance'][i + 1]['relative_variance']
        pattern_freq = pattern_results['pattern_occurrences'][i + 1]['frequency'] * 100
        subplot_titles.append(
            f'Pattern {i + 1}<br>'
            f'Var Explained: {pattern_var:.1f}%<br>'
            f'Frequency: {pattern_freq:.1f}%'
        )

    # Create base specs for pattern rows
    specs = [[{"type": "heatmap"} for _ in range(n_cols)] for _ in range(base_rows)]

    if variance_results:
        # This needs to be updated for a 7×4 grid
        metrics_row = [
            {"type": "heatmap", "colspan": 2},
            None,  # Needed because of colspan above
            {"type": "bar", "colspan": 1},
            {"type": "indicator"}  # Last element
        ]
        specs.append(metrics_row)
        subplot_titles.extend(['Pattern Usage', 'Quality Metrics', 'Explained Variance'])

    # Create subplots
    fig = make_subplots(
        rows=total_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        specs=specs,
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )

    # Create mask for coastlines
    lons = ds.longitude.values
    lats = ds.latitude.values

    # Add pattern heatmaps
    for i in range(n_rows):
        for j in range(n_cols):
            pattern_idx = i * n_cols + j + 1
            pattern = weights[i, j].reshape(original_shape[1:])

            # Get occurrence dates for this pattern
            pattern_dates = pattern_results['pattern_occurrences'][pattern_idx]['dates']
            recent_dates = pattern_dates[-5:] if len(pattern_dates) > 5 else pattern_dates

            # Create hover text
            hover_text = [[
                f'Lat: {lat:.1f}°N<br>'
                f'Lon: {lon:.1f}°W<br>'
                f'SST Anomaly: {val:.2f}<br>'
                f'Pattern Stats:<br>'
                f'- Variance Explained: {pattern_results["pattern_variance"][pattern_idx]["relative_variance"]:.1f}%<br>'
                f'- Frequency: {pattern_results["pattern_occurrences"][pattern_idx]["frequency"] * 100:.1f}%<br>'
                f'- Total Occurrences: {pattern_results["pattern_occurrences"][pattern_idx]["count"]}<br>'
                f'- Recent Months: {", ".join(recent_dates)}'
                for lon, val in zip(ds.longitude.values, row)
            ] for lat, row in zip(ds.latitude.values, pattern)]

            # Add SST pattern
            fig.add_trace(
                go.Heatmap(
                    z=pattern,
                    x=ds.longitude.values,
                    y=ds.latitude.values,
                    colorscale='RdBu_r',
                    zmid=0,
                    showscale=True if (i == 0 and j == 0) else False,
                    colorbar=dict(
                        title="SST Anomaly",
                        len=0.6,
                        thickness=15,
                        x=1.02,
                        y=0.8,
                        tickformat=".1f"
                    ) if (i == 0 and j == 0) else None,
                    hoverongaps=False,
                    hoverinfo='text',
                    text=hover_text
                ),
                row=i + 1,
                col=j + 1
            )

            # Update axes properties
            fig.update_xaxes(showgrid=False, zeroline=False, row=i + 1, col=j + 1)
            fig.update_yaxes(showgrid=False, zeroline=False, row=i + 1, col=j + 1)

    # Add metrics if available
    if variance_results:
        metrics_row = base_rows + 1

        # Determine pattern frequency matrix
        if 'pattern_frequency' in variance_results:
            pattern_freq = variance_results['pattern_frequency']
        else:
            # Create pattern frequency matrix from pattern_results
            pattern_freq = np.zeros((n_rows, n_cols))
            for i in range(n_rows):
                for j in range(n_cols):
                    pattern_idx = i * n_cols + j + 1
                    if pattern_idx in pattern_results['pattern_occurrences']:
                        pattern_freq[i, j] = pattern_results['pattern_occurrences'][pattern_idx]['frequency']

        # Add pattern usage heatmap trace
        fig.add_trace(
            go.Heatmap(
                z=pattern_freq,
                colorscale='Viridis',
                text=[[f'{val:.3f}' for val in row] for row in pattern_freq],
                texttemplate='%{text}',
                showscale=True,
                colorbar=dict(
                    title="Pattern Usage",
                    len=0.2,
                    thickness=10,
                    x=1.02,
                    y=0.15,
                    tickformat=".3f"
                ),
                name='Pattern Usage'
            ),
            row=metrics_row, col=1
        )

        # Quality metrics bar chart
        metrics_data = {
            'QE': variance_results['quality_metrics']['quantization_error'],
            'TE': variance_results['quality_metrics']['topographic_error'],
            'NP': variance_results['quality_metrics']['neighborhood_preservation']
        }

        fig.add_trace(
            go.Bar(
                x=list(metrics_data.keys()),
                y=list(metrics_data.values()),
                text=[f'{val:.3f}' for val in metrics_data.values()],
                textposition='auto',
                name='Quality Metrics'
            ),
            row=metrics_row, col=3
        )

        # Weighted explained variance indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=variance_results['total_explained_variance_ratio'] * 100,
                title={'text': "Total Explained Variance (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ]
                }
            ),
            row=metrics_row, col=4
        )

    # Update layout
    fig.update_layout(
        height=400 * total_rows,
        width=400 * n_cols,
        title_text=f"Interactive SOM Patterns",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=100, b=50, l=50, r=50)
    )

    updatemenus = [
        dict(
            buttons=list([
                dict(
                    args=[{"visible": [True] * (n_rows * n_cols)}],
                    label="All Patterns",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [
                        pattern_results['pattern_variance'][i + 1]['relative_variance'] >
                        filters['high_variance']['threshold']
                        for i in range(n_rows * n_cols)
                    ]}],
                    label="High Variance Patterns",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [
                        pattern_results['pattern_occurrences'][i + 1]['frequency'] >
                        filters['high_frequency']['threshold']
                        for i in range(n_rows * n_cols)
                    ]}],
                    label="High Frequency Patterns",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [
                        pattern_results['pattern_variance'][i + 1]['relative_variance'] *
                        pattern_results['pattern_occurrences'][i + 1]['frequency'] >
                        filters['top_patterns']['threshold']
                        for i in range(n_rows * n_cols)
                    ]}],
                    label="Top Contributing Patterns",
                    method="restyle"
                )
            ]),
            direction="down",
            showactive=True,
            x=0.1,
            y=1.1,
            xanchor="left",
            yanchor="top"
        )
    ]

    # Update layout to include scree plot
    fig.update_layout(
        updatemenus=updatemenus,
        grid=dict(rows=total_rows + 1, columns=n_cols),  # Add row for scree plot
        height=400 * (total_rows + 1)  # Increase height to accommodate scree plot
    )

    return fig, scree_fig


#######################################################################################################################
#######################################################################################################################

def plot_top_patterns_with_distribution(som, original_shape, ds, pattern_results, variance_results, output_dir):
    """
    Create interactive visualization of top 9 patterns with a single colorbar
    """
    # Get top 9 patterns by variance explained
    variance_by_pattern = {k: v['relative_variance']
                           for k, v in pattern_results['pattern_variance'].items()}
    top_patterns = sorted(variance_by_pattern.items(),
                          key=lambda x: x[1], reverse=True)[:9]

    # Create figure with subplot grid
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[f'Pattern {p[0]}: {p[1]:.1f}%' for p in top_patterns],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )

    # Get SOM weights
    weights = som.get_weights()

    # Find global min and max for consistent color scaling
    all_patterns = []
    for pattern_idx, _ in top_patterns:
        i, j = (pattern_idx - 1) // weights.shape[1], (pattern_idx - 1) % weights.shape[1]
        pattern = weights[i, j].reshape(original_shape[1:])
        all_patterns.append(pattern)

    global_min = min(p.min() for p in all_patterns)
    global_max = max(p.max() for p in all_patterns)
    abs_max = max(abs(global_min), abs(global_max))

    # Add pattern heatmaps
    for idx, (pattern_idx, variance) in enumerate(top_patterns):
        row = idx // 3 + 1
        col = idx % 3 + 1

        i, j = (pattern_idx - 1) // weights.shape[1], (pattern_idx - 1) % weights.shape[1]
        pattern = weights[i, j].reshape(original_shape[1:])

        # Get pattern dates
        pattern_dates = pattern_results['pattern_occurrences'][pattern_idx]['dates']
        recent_dates = pattern_dates[-5:] if len(pattern_dates) > 5 else pattern_dates

        # Create hover text
        hover_text = [[
            f'Lat: {lat:.1f}°N<br>'
            f'Lon: {lon:.1f}°W<br>'
            f'SST Anomaly: {val:.2f}<br>'
            f'Pattern Stats:<br>'
            f'- Variance Explained: {variance:.1f}%<br>'
            f'- Frequency: {pattern_results["pattern_occurrences"][pattern_idx]["frequency"] * 100:.1f}%<br>'
            f'- Recent Months: {", ".join(recent_dates)}'
            for lon, val in zip(ds.longitude.values, row)
        ] for lat, row in zip(ds.latitude.values, pattern)]

        fig.add_trace(
            go.Heatmap(
                z=pattern,
                x=ds.longitude.values,
                y=ds.latitude.values,
                colorscale='RdBu_r',
                zmid=0,
                zmin=-abs_max,
                zmax=abs_max,
                showscale=idx == 0,  # Only show colorbar for first pattern
                colorbar=dict(
                    title="SST Anomaly",
                    len=0.6,
                    thickness=15,
                    x=1.02,
                    y=0.5,
                    tickformat=".1f"
                ) if idx == 0 else None,
                hoverongaps=False,
                hoverinfo='text',
                text=hover_text
            ),
            row=row,
            col=col
        )

        # Update axes for each subplot
        fig.update_xaxes(showgrid=False, zeroline=False, row=row, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, row=row, col=col)

    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        title_text="Top 9 SOM Patterns",
        showlegend=False,
        plot_bgcolor='white'
    )

    # Save the plot
    fig.write_html(os.path.join(output_dir, 'top_patterns.html'))
    return fig


#######################################################################################################################
#######################################################################################################################


def load_nao_data_from_file(file_path, start_year=1950):
    """
    Load NAO index data from downloaded file

    Parameters:
    -----------
    file_path : str
        Path to the NAO index data file
    start_year : int
        Starting year to filter data (optional)

    Returns:
    --------
    numpy.ndarray
        NAO index values for DJF months only
    """
    # Read the data file with comma delimiter
    nao_data = pd.read_csv(file_path)

    # Create list to store values
    nao_values = []
    djf_months = [12, 1, 2]  # December, January, February

    # If there are no column headers, add them
    if len(nao_data.columns) == 1:
        # It seems the file is comma-delimited but read as a single column
        # Split the first column into multiple columns
        first_row = nao_data.iloc[0, 0].split(',')

        if len(first_row) == 13:  # Year + 12 months
            # Create proper column structure
            new_data = []
            for _, row in nao_data.iterrows():
                values = row.iloc[0].split(',')
                new_data.append(values)

            # Create new DataFrame
            nao_data = pd.DataFrame(new_data, columns=['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                                                       'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Convert Year column to integer
    nao_data['Year'] = nao_data['Year'].astype(int)

    # Filter data by start year
    nao_data = nao_data[nao_data['Year'] >= start_year]

    # Extract DJF values
    for idx, row in nao_data.iterrows():
        year = row['Year']

        # Add December
        if idx < len(nao_data) - 1:  # Not the last row
            nao_values.append(float(row['Dec']))

        # Add January and February
        if 'Jan' in nao_data.columns and 'Feb' in nao_data.columns:
            nao_values.append(float(row['Jan']))
            nao_values.append(float(row['Feb']))

    return np.array(nao_values)


#######################################################################################################################
#######################################################################################################################


def save_pattern_summary(results, output_dir):
    """
    Save comprehensive pattern summary to both txt and csv files
    """
    # Save detailed text summary
    with open(os.path.join(output_dir, 'patterns_by_variance_detailed.txt'), 'w') as f:
        f.write(f"Total variance explained by all patterns: {results['total_variance_explained']:.2f}%\n\n")
        f.write("All patterns by variance explained:\n\n")

        # Sort patterns by variance
        variance_by_pattern = {k: v['relative_variance']
                               for k, v in results['pattern_variance'].items()}
        sorted_patterns = sorted(variance_by_pattern.items(),
                                 key=lambda x: x[1], reverse=True)

        for pattern, variance in sorted_patterns:
            occurrences = results['pattern_occurrences'][pattern]
            f.write(f"Pattern {pattern}:\n")
            f.write(f"- Variance explained: {variance:.2f}%\n")
            f.write(f"- Occurrence count: {occurrences['count']}\n")
            f.write(f"- Frequency: {occurrences['frequency'] * 100:.2f}%\n")
            f.write("- All occurrences: " + ", ".join(occurrences['dates']) + "\n\n")

    # Save CSV format

    with open(os.path.join(output_dir, 'patterns_by_variance.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Pattern', 'Variance_Explained_%', 'Occurrence_Count',
                         'Frequency_%', 'Occurrences'])

        # Write data for each pattern
        for pattern, variance in sorted_patterns:
            occurrences = results['pattern_occurrences'][pattern]
            writer.writerow([
                pattern,
                f"{variance:.2f}",
                occurrences['count'],
                f"{occurrences['frequency'] * 100:.2f}",
                ";".join(occurrences['dates'])  # Use semicolon to separate dates for easier CSV parsing
            ])

    # Also keep the original top 10 summary
    with open(os.path.join(output_dir, 'Top_patterns_by_variance_explained.txt'), 'w') as f:
        f.write(f"Total variance explained by all patterns: {results['total_variance_explained']:.2f}%\n\n")
        f.write("Top patterns by variance explained:\n\n")

        # Get top 10 patterns
        top_patterns = sorted_patterns[:10]

        for pattern, variance in top_patterns:
            occurrences = results['pattern_occurrences'][pattern]
            f.write(f"Pattern {pattern}:\n")
            f.write(f"- Variance explained: {variance:.2f}%\n")
            f.write(f"- Occurrence count: {occurrences['count']}\n")
            f.write(f"- Frequency: {occurrences['frequency'] * 100:.2f}%\n")
            f.write(f"- Example months: {', '.join(occurrences['dates'][:5])}...\n\n")


#######################################################################################################################
#######################################################################################################################

def plot_som_patterns_grid(som, original_shape, ds, output_dir, pattern_results):
    """
    Plot all SOM patterns in a grid with panel labels using p1, p2, etc.
    """
    # Get SOM weights and dimensions
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape
    n_patterns = n_rows * n_cols

    # Calculate grid layout - using 4 columns
    grid_cols = 4
    grid_rows = (n_patterns + grid_cols - 1) // grid_cols

    # Create figure and axes
    fig = plt.figure(figsize=(16, 4 * grid_rows))

    # Create coordinate meshgrid
    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

    # Find global min/max for consistent color scaling
    all_patterns = []
    for pattern_num in range(1, n_patterns + 1):
        i, j = (pattern_num - 1) // n_cols, (pattern_num - 1) % n_cols
        pattern = weights[i, j].reshape(original_shape[1:])
        all_patterns.append(pattern)

    vmin = min(p.min() for p in all_patterns)
    vmax = max(p.max() for p in all_patterns)
    abs_max = max(abs(vmin), abs(vmax))

    # Create colormap
    cmap = plt.cm.RdBu_r

    # Plot each pattern
    for pattern_num in range(1, n_patterns + 1):
        i, j = (pattern_num - 1) // n_cols, (pattern_num - 1) % n_cols
        pattern = weights[i, j].reshape(original_shape[1:])

        # Calculate row and column in the subplot grid
        grid_row = (pattern_num - 1) // grid_cols
        grid_col = (pattern_num - 1) % grid_cols

        # Create subplot with cartopy projection
        ax = fig.add_subplot(grid_rows, grid_cols, pattern_num, projection=ccrs.PlateCarree())

        # Plot the pattern
        im = ax.pcolormesh(lons, lats, pattern,
                           transform=ccrs.PlateCarree(),
                           cmap=cmap,
                           vmin=-abs_max, vmax=abs_max,
                           shading='auto')

        # Add coastlines
        ax.coastlines(resolution='50m', linewidth=0.5)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True if grid_col == 0 else False
        gl.bottom_labels = True if grid_row == grid_rows - 1 else False

        # Get pattern variance and frequency information
        pattern_var = np.var(pattern).item() * 100  # Convert to percentage

        # Get frequency if available in pattern_results
        pattern_freq = 0.0
        if pattern_results and 'pattern_occurrences' in pattern_results:
            if pattern_num in pattern_results['pattern_occurrences']:
                pattern_freq = pattern_results['pattern_occurrences'][pattern_num]['frequency'] * 100

        # Add pattern label in the corner with p1, p2, etc.
        label_text = f"(p{pattern_num})"
        ax.text(0.03, 0.97, label_text, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # Add title with pattern number, variance, and frequency
        ax.set_title(f"Pattern {pattern_num}\nVar: {pattern_var:.1f}%, Freq: {pattern_freq:.1f}%")

        # Set extent
        ax.set_extent([ds.longitude.min(), ds.longitude.max(),
                       ds.latitude.min(), ds.latitude.max()],
                      crs=ccrs.PlateCarree())

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, label='Normalized SST Anomaly')

    # Adjust layout
    fig.suptitle('Complete SOM Pattern Grid', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save figure
    plt.savefig(os.path.join(output_dir, 'som_patterns_complete_grid.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    return fig


#######################################################################################################################
#######################################################################################################################

def save_som_model(som, output_dir):
    """
    Save the trained SOM model with error handling

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    output_dir : str
        Directory to save the model
    """
    try:
        model_path = os.path.join(output_dir, 'trained_som_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(som, f)
        print(f"Successfully saved SOM model to: {model_path}")

        # Verify the save by trying to load it
        try:
            with open(model_path, 'rb') as f:
                _ = pickle.load(f)
            print("Verified model save - file is readable")
        except Exception as e:
            print(f"Warning: Saved model file may be corrupted: {str(e)}")

    except Exception as e:
        print(f"Error saving SOM model: {str(e)}")
        raise


#######################################################################################################################
#######################################################################################################################

def load_som_model(model_path):
    """
    Load a trained SOM model with validation

    Parameters:
    -----------
    model_path : str
        Path to the saved model file

    Returns:
    --------
    MiniSom
        Loaded SOM model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        with open(model_path, 'rb') as f:
            som = pickle.load(f)

        # Verify it's a MiniSom object
        if not hasattr(som, 'winner') or not hasattr(som, 'quantization_error'):
            raise ValueError("Loaded object does not appear to be a valid SOM model")

        print(f"Successfully loaded SOM model from: {model_path}")
        print(f"Model dimensions: {som.get_weights().shape}")
        return som

    except Exception as e:
        print(f"Error loading SOM model: {str(e)}")
        raise


#######################################################################################################################
#######################################################################################################################

def save_patterns_for_projection(som, ds, variance_results, pattern_results, output_dir):
    """
    Save SOM patterns and metadata for projection onto other datasets (e.g., ERA5)

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    ds : xarray.Dataset
        Original dataset with coordinates
    variance_results : dict
        Results from improved_variance_calculation
    pattern_results : dict
        Results from analyze_pattern_occurrences
    output_dir : str
        Directory to save pattern information
    """

    patterns_dir = os.path.join(output_dir, 'patterns_for_projection')
    os.makedirs(patterns_dir, exist_ok=True)

    # 1. Save SOM weights and spatial information
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    patterns_info = {
        'grid_shape': (n_rows, n_cols),
        'spatial_shape': ds.sst.shape[1:],  # lat, lon dimensions
        'lat_range': [float(ds.latitude.min().values), float(ds.latitude.max().values)],
        'lon_range': [float(ds.longitude.min().values), float(ds.longitude.max().values)],
        'n_patterns': n_rows * n_cols,
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source_data': 'HadISST'
    }

    # Save patterns metadata
    with open(os.path.join(patterns_dir, 'patterns_metadata.json'), 'w') as f:
        json.dump(patterns_info, f, indent=4)

    # 2. Save pattern weights
    pattern_weights = {
        'weights': weights,
        'grid_shape': (n_rows, n_cols),
        'n_features': n_features
    }
    with open(os.path.join(patterns_dir, 'pattern_weights.pkl'), 'wb') as f:
        pickle.dump(pattern_weights, f)

    # 3. Save pattern statistics
    pattern_stats = {}
    for i in range(n_rows * n_cols):
        pattern_idx = i + 1
        pattern_stats[pattern_idx] = {
            'variance_explained': pattern_results['pattern_variance'][pattern_idx]['relative_variance'],
            'frequency': pattern_results['pattern_occurrences'][pattern_idx]['frequency'],
            'occurrence_count': pattern_results['pattern_occurrences'][pattern_idx]['count'],
            'occurrence_dates': pattern_results['pattern_occurrences'][pattern_idx]['dates']
        }

    with open(os.path.join(patterns_dir, 'pattern_statistics.json'), 'w') as f:
        json.dump(pattern_stats, f, indent=4)

    # 4. Save pattern coordinates
    coords = {
        'latitude': ds.latitude.values.tolist(),
        'longitude': ds.longitude.values.tolist()
    }
    with open(os.path.join(patterns_dir, 'coordinates.json'), 'w') as f:
        json.dump(coords, f, indent=4)

    # 5. Save variance results
    variance_summary = {
        'total_variance': float(variance_results['total_variance']),
        'explained_variance_ratio': float(variance_results['total_explained_variance_ratio']),
        'quality_metrics': {
            'quantization_error': float(variance_results['quality_metrics']['quantization_error']),
            'topographic_error': float(variance_results['quality_metrics']['topographic_error']),
            'neighborhood_preservation': float(variance_results['quality_metrics']['neighborhood_preservation'])
        }
    }
    with open(os.path.join(patterns_dir, 'variance_summary.json'), 'w') as f:
        json.dump(variance_summary, f, indent=4)

    print(f"\nSaved pattern information for projection in: {patterns_dir}")
    print("Files saved:")
    print("- patterns_metadata.json: General information about the patterns")
    print("- pattern_weights.pkl: SOM pattern weights for projection")
    print("- pattern_statistics.json: Detailed statistics for each pattern")
    print("- coordinates.json: Spatial coordinates information")
    print("- variance_summary.json: Variance and quality metrics")


#######################################################################################################################
#######################################################################################################################

def load_patterns_for_projection(patterns_dir):
    """
    Load saved patterns and metadata for projection

    Parameters:
    -----------
    patterns_dir : str
        Directory containing saved pattern information

    Returns:
    --------
    dict
        Dictionary containing all pattern information
    """
    # Load metadata
    with open(os.path.join(patterns_dir, 'patterns_metadata.json'), 'r') as f:
        metadata = json.load(f)

    # Load weights
    with open(os.path.join(patterns_dir, 'pattern_weights.pkl'), 'rb') as f:
        weights_data = pickle.load(f)

    # Load statistics
    with open(os.path.join(patterns_dir, 'pattern_statistics.json'), 'r') as f:
        statistics = json.load(f)

    # Load coordinates
    with open(os.path.join(patterns_dir, 'coordinates.json'), 'r') as f:
        coordinates = json.load(f)

    # Load variance summary
    with open(os.path.join(patterns_dir, 'variance_summary.json'), 'r') as f:
        variance = json.load(f)

    return {
        'metadata': metadata,
        'weights': weights_data,
        'statistics': statistics,
        'coordinates': coordinates,
        'variance': variance
    }


#######################################################################################################################
#######################################################################################################################

def project_som_patterns(model_dir, new_data_file, output_dir=None):
    """
    Project saved SOM patterns onto new data

    Parameters:
    -----------
    model_dir : str
        Directory containing saved SOM model and patterns
    new_data_file : str
        Path to new data file to project onto
    output_dir : str or None
        Directory to save projection results (if None, uses model_dir/projections)

    Returns:
    --------
    dict
        Dictionary containing projection results
    """
    print(f"\nProjecting SOM patterns onto new data: {new_data_file}")

    # Create output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(model_dir, 'projections')
    os.makedirs(output_dir, exist_ok=True)

    # Load SOM model
    model_path = os.path.join(model_dir, 'trained_som_model.pkl')
    som = load_som_model(model_path)

    # Load pattern information
    patterns_dir = os.path.join(model_dir, 'patterns_for_projection')
    pattern_info = load_patterns_for_projection(patterns_dir)

    # Load clustering results if available
    clustering_path = os.path.join(model_dir, 'pattern_clustering_results.pkl')
    if os.path.exists(clustering_path):
        with open(clustering_path, 'rb') as f:
            clustering_results = pickle.load(f)
    else:
        clustering_results = None

    # Load and preprocess new data
    new_ds = load_and_preprocess_sst(new_data_file)
    X_new = prepare_som_input(new_ds)

    # Get BMUs for all new data points
    bmu_indices = np.array([som.winner(x) for x in X_new])

    # Convert BMU grid indices to pattern numbers (1-based)
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Convert to cluster sequence if clustering results available
    cluster_sequence = None
    if clustering_results is not None and 'pattern_clusters' in clustering_results:
        pattern_clusters = clustering_results['pattern_clusters']
        cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Associate with time information
    times = pd.to_datetime(new_ds.time.values)

    # Create projection results
    projection_results = {
        'times': times,
        'pattern_sequence': pattern_sequence,
        'bmu_indices': bmu_indices,
        'cluster_sequence': cluster_sequence
    }

    # Save results
    with open(os.path.join(output_dir, 'projection_results.pkl'), 'wb') as f:
        pickle.dump(projection_results, f)

    # Create summary CSV
    results_df = pd.DataFrame({
        'time': times,
        'pattern': pattern_sequence,
        'cluster': cluster_sequence if cluster_sequence is not None else np.nan
    })
    results_df.to_csv(os.path.join(output_dir, 'pattern_projection_summary.csv'), index=False)

    # Generate visualization of pattern/cluster frequency over time
    create_projection_visualizations(projection_results, output_dir)

    print(f"Projection complete. Results saved to: {output_dir}")
    return projection_results


#######################################################################################################################
#######################################################################################################################

def create_projection_visualizations(projection_results, output_dir):
    """Create visualizations of pattern/cluster projections over time"""
    # Extract data
    times = projection_results['times']
    pattern_sequence = projection_results['pattern_sequence']
    cluster_sequence = projection_results['cluster_sequence']

    # Convert to pandas DataFrame for easier analysis
    df = pd.DataFrame({
        'time': times,
        'pattern': pattern_sequence,
        'cluster': cluster_sequence if cluster_sequence is not None else None
    })

    # Add year and month columns
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    # Create pattern frequency over time plot
    fig = go.Figure()

    # Group by year and count patterns
    pattern_counts = df.groupby(['year', 'pattern']).size().unstack(fill_value=0)

    # Calculate relative frequency
    pattern_freq = pattern_counts.div(pattern_counts.sum(axis=1), axis=0)

    # Plot each pattern
    for pattern in pattern_freq.columns:
        fig.add_trace(
            go.Scatter(
                x=pattern_freq.index,
                y=pattern_freq[pattern],
                mode='lines',
                name=f'Pattern {pattern}'
            )
        )

    fig.update_layout(
        title='Pattern Frequency Over Time',
        xaxis_title='Year',
        yaxis_title='Relative Frequency',
        height=600,
        width=1000
    )

    fig.write_html(os.path.join(output_dir, 'pattern_frequency_over_time.html'))

    # Also create cluster frequency plot if available
    if cluster_sequence is not None:
        fig = go.Figure()

        # Group by year and count clusters
        cluster_counts = df.groupby(['year', 'cluster']).size().unstack(fill_value=0)

        # Calculate relative frequency
        cluster_freq = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

        # Plot each cluster
        for cluster in cluster_freq.columns:
            fig.add_trace(
                go.Scatter(
                    x=cluster_freq.index,
                    y=cluster_freq[cluster],
                    mode='lines',
                    name=f'Cluster {cluster}'
                )
            )

        fig.update_layout(
            title='Cluster Frequency Over Time',
            xaxis_title='Year',
            yaxis_title='Relative Frequency',
            height=600,
            width=1000
        )

        fig.write_html(os.path.join(output_dir, 'cluster_frequency_over_time.html'))


#######################################################################################################################
#######################################################################################################################

def analyze_pattern_time_series(som, X, ds, clustering_results, output_dir):
    """
    Analyze the time series of SOM patterns and clusters

    Parameters:
    -----------
    som : MiniSom
        Trained SOM object
    X : numpy.ndarray
        Input data array
    ds : xarray.Dataset
        Original dataset with time coordinates
    clustering_results : dict
        Results from cluster_som_patterns
    output_dir : str
        Directory to save analysis results

    Returns:
    --------
    dict
        Dictionary containing time series analysis results
    """
    print("\nAnalyzing pattern time series (1950 onwards)...")

    # Create time series directory
    time_series_dir = os.path.join(output_dir, 'time_series_analysis')
    os.makedirs(time_series_dir, exist_ok=True)

    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert BMU grid indices to pattern numbers (1-based)
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)

    # Map patterns to clusters if clustering results available
    cluster_sequence = None
    if clustering_results and 'pattern_clusters' in clustering_results:
        pattern_clusters = clustering_results['pattern_clusters']
        cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Create DataFrame for analysis
    df = pd.DataFrame({
        'time': times,
        'pattern': pattern_sequence,
        'cluster': cluster_sequence
    })

    # Add year and month
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    # Create annual pattern frequency
    annual_pattern_freq = df.groupby(['year', 'pattern']).size().unstack(fill_value=0)
    annual_pattern_freq_norm = annual_pattern_freq.div(annual_pattern_freq.sum(axis=1), axis=0)

    # Create annual cluster frequency if available
    annual_cluster_freq = None
    annual_cluster_freq_norm = None
    if cluster_sequence is not None:
        annual_cluster_freq = df.groupby(['year', 'cluster']).size().unstack(fill_value=0)
        annual_cluster_freq_norm = annual_cluster_freq.div(annual_cluster_freq.sum(axis=1), axis=0)

    # Save results
    df.to_csv(os.path.join(time_series_dir, 'pattern_time_series.csv'), index=False)
    annual_pattern_freq.to_csv(os.path.join(time_series_dir, 'annual_pattern_frequency.csv'))
    annual_pattern_freq_norm.to_csv(os.path.join(time_series_dir, 'annual_pattern_frequency_normalized.csv'))

    if annual_cluster_freq is not None:
        annual_cluster_freq.to_csv(os.path.join(time_series_dir, 'annual_cluster_frequency.csv'))
        annual_cluster_freq_norm.to_csv(os.path.join(time_series_dir, 'annual_cluster_frequency_normalized.csv'))

    # Create visualizations
    create_time_series_visualizations(df, time_series_dir)

    # Analyze trends
    trend_results = analyze_pattern_trends(df, time_series_dir)

    print(f"Time series analysis complete. Results saved to: {time_series_dir}")

    return {
        'time_series': df,
        'annual_pattern_freq': annual_pattern_freq_norm,
        'annual_cluster_freq': annual_cluster_freq_norm,
        'trends': trend_results
    }


#######################################################################################################################
#######################################################################################################################

def create_time_series_visualizations(df, output_dir):
    """Create visualizations of pattern/cluster time series"""
    print("  Creating time series visualizations...")

    # Create pattern frequency over time plot
    fig = go.Figure()

    # Group by year and calculate pattern frequency
    pattern_counts = df.groupby(['year', 'pattern']).size().unstack(fill_value=0)
    pattern_freq = pattern_counts.div(pattern_counts.sum(axis=1), axis=0)

    # Plot each pattern
    for pattern in pattern_freq.columns:
        fig.add_trace(
            go.Scatter(
                x=pattern_freq.index,
                y=pattern_freq[pattern],
                mode='lines',
                name=f'Pattern {pattern}'
            )
        )

    fig.update_layout(
        title='Pattern Frequency Over Time (1950 onwards)',
        xaxis_title='Year',
        yaxis_title='Relative Frequency',
        height=600,
        width=1000
    )

    fig.write_html(os.path.join(output_dir, 'pattern_frequency_over_time.html'))

    # Create cluster frequency plot if available
    if 'cluster' in df.columns and not df['cluster'].isna().all():
        fig = go.Figure()

        # Group by year and calculate cluster frequency
        cluster_counts = df.groupby(['year', 'cluster']).size().unstack(fill_value=0)
        cluster_freq = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

        # Plot each cluster
        for cluster in cluster_freq.columns:
            fig.add_trace(
                go.Scatter(
                    x=cluster_freq.index,
                    y=cluster_freq[cluster],
                    mode='lines',
                    name=f'Cluster {cluster}'
                )
            )

        fig.update_layout(
            title='Cluster Frequency Over Time (1950 onwards)',
            xaxis_title='Year',
            yaxis_title='Relative Frequency',
            height=600,
            width=1000
        )

        fig.write_html(os.path.join(output_dir, 'cluster_frequency_over_time.html'))

    # Create stacked area chart for patterns
    fig = go.Figure()

    for pattern in pattern_freq.columns:
        fig.add_trace(
            go.Scatter(
                x=pattern_freq.index,
                y=pattern_freq[pattern],
                mode='lines',
                stackgroup='one',
                name=f'Pattern {pattern}'
            )
        )

    fig.update_layout(
        title='Pattern Composition Over Time',
        xaxis_title='Year',
        yaxis_title='Relative Frequency',
        height=600,
        width=1000
    )

    fig.write_html(os.path.join(output_dir, 'pattern_composition_over_time.html'))


#######################################################################################################################
#######################################################################################################################


def analyze_pattern_trends(df, output_dir):
    """Analyze trends in pattern/cluster frequencies"""
    print("  Analyzing pattern frequency trends...")

    from scipy import stats

    # Create results dictionary
    trend_results = {
        'pattern_trends': {},
        'cluster_trends': {}
    }

    # Analyze pattern trends
    pattern_counts = df.groupby(['year', 'pattern']).size().unstack(fill_value=0)
    pattern_freq = pattern_counts.div(pattern_counts.sum(axis=1), axis=0)

    years = pattern_freq.index.values

    for pattern in pattern_freq.columns:
        freq = pattern_freq[pattern].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, freq)

        trend_results['pattern_trends'][int(pattern)] = {
            'slope': slope,
            'p_value': p_value,
            'r_squared': r_value ** 2,
            'trend_per_decade': slope * 10  # Change per decade
        }

    # Analyze cluster trends if available
    if 'cluster' in df.columns and not df['cluster'].isna().all():
        cluster_counts = df.groupby(['year', 'cluster']).size().unstack(fill_value=0)
        cluster_freq = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

        for cluster in cluster_freq.columns:
            freq = cluster_freq[cluster].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, freq)

            trend_results['cluster_trends'][int(cluster)] = {
                'slope': slope,
                'p_value': p_value,
                'r_squared': r_value ** 2,
                'trend_per_decade': slope * 10  # Change per decade
            }

    # Save trend results
    with open(os.path.join(output_dir, 'trend_analysis.txt'), 'w') as f:
        f.write("Pattern Frequency Trends (1950 onwards)\n")
        f.write("======================================\n\n")

        for pattern, stats in sorted(trend_results['pattern_trends'].items()):
            f.write(f"Pattern {pattern}:\n")
            f.write(f"  Trend per decade: {stats['trend_per_decade']:.4f}\n")
            f.write(f"  p-value: {stats['p_value']:.4f}")
            if stats['p_value'] < 0.05:
                f.write(" (statistically significant)")
            f.write("\n")
            f.write(f"  R-squared: {stats['r_squared']:.4f}\n\n")

        if trend_results['cluster_trends']:
            f.write("\nCluster Frequency Trends (1950 onwards)\n")
            f.write("======================================\n\n")

            for cluster, stats in sorted(trend_results['cluster_trends'].items()):
                f.write(f"Cluster {cluster}:\n")
                f.write(f"  Trend per decade: {stats['trend_per_decade']:.4f}\n")
                f.write(f"  p-value: {stats['p_value']:.4f}")
                if stats['p_value'] < 0.05:
                    f.write(" (statistically significant)")
                f.write("\n")
                f.write(f"  R-squared: {stats['r_squared']:.4f}\n\n")

    # Create trend visualization
    fig = go.Figure()

    significant_trends = []
    for pattern, stats in trend_results['pattern_trends'].items():
        # Only plot significant trends
        if stats['p_value'] < 0.05:
            significant_trends.append((pattern, stats['trend_per_decade'], stats['p_value']))

    # Sort by trend magnitude
    significant_trends.sort(key=lambda x: abs(x[1]), reverse=True)

    for pattern, trend, p_value in significant_trends:
        fig.add_trace(
            go.Bar(
                x=[f'Pattern {pattern}'],
                y=[trend],
                name=f'Pattern {pattern}',
                text=[f'p={p_value:.3f}'],
                textposition='auto',
                marker_color='blue' if trend > 0 else 'red'
            )
        )

    fig.update_layout(
        title='Significant Pattern Frequency Trends (Change per Decade)',
        xaxis_title='Pattern',
        yaxis_title='Change per Decade',
        height=500,
        width=800
    )

    fig.write_html(os.path.join(output_dir, 'pattern_trend_analysis.html'))

    return trend_results


#######################################################################################################################
#######################################################################################################################

def analyze_z500_by_pattern(som, X, ds, z500_file, output_dir):
    """
    Analyze Z500 anomalies associated with each SOM pattern directly

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array (n_samples x n_features)
    ds : xarray.Dataset
        Original SST dataset with coordinates
    z500_file : str
        Path to Z500 data file
    output_dir : str
        Directory to save results
    """
    # Create output directory
    pattern_z500_dir = os.path.join(output_dir, 'pattern_z500_associations')
    os.makedirs(pattern_z500_dir, exist_ok=True)

    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers (1-based)
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    years = times.year

    # Load Z500 data
    print("Loading Z500 data...")
    ds_z500 = xr.open_dataset(z500_file)

    # Select 500 hPa level if necessary
    if 'pressure_level' in ds_z500.dims:
        ds_z500 = ds_z500.sel(pressure_level=500.0)

    # Select DJF months
    z500_times = pd.to_datetime(ds_z500.valid_time.values)
    is_djf = (z500_times.month == 12) | (z500_times.month == 1) | (z500_times.month == 2)
    ds_z500_djf = ds_z500.isel(valid_time=is_djf)

    # Create year coordinate for seasonal averages
    year_coords = [t.year if t.month > 2 else t.year - 1
                   for t in pd.to_datetime(ds_z500_djf.valid_time.values)]
    ds_z500_djf = ds_z500_djf.assign_coords(year=('valid_time', year_coords))

    # Calculate seasonal means
    z500_yearly = ds_z500_djf.groupby('year').mean('valid_time')

    # Calculate climatology
    z500_climatology = z500_yearly.z.mean('year')

    # Find common years between SST and Z500
    unique_years = np.unique(years)
    common_years = sorted(list(set(unique_years).intersection(set(z500_yearly.year.values))))

    print(f"Found {len(common_years)} common years between SST and Z500 data")

    # Create mapping of year to pattern
    year_pattern = {}
    for year in common_years:
        # Find dominant pattern for this year
        year_mask = (years == year)
        if np.any(year_mask):
            year_patterns = pattern_sequence[year_mask]
            pattern_counts = np.bincount(year_patterns)
            dominant_pattern = np.argmax(pattern_counts[1:]) + 1  # Adjust for 1-based indexing
            year_pattern[year] = dominant_pattern

    # Create composites for each pattern
    all_patterns = range(1, n_rows * n_cols + 1)

    # Create land feature for maps
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3
                                        )

    # For each pattern, create Z500 composite
    pattern_years = {}
    for pattern in all_patterns:
        # Get years when this pattern was dominant
        pattern_years[pattern] = [year for year, pat in year_pattern.items() if pat == pattern]

        if len(pattern_years[pattern]) < 3:
            print(f"Pattern {pattern}: Not enough years for analysis ({len(pattern_years[pattern])})")
            continue

        print(f"Pattern {pattern}: {len(pattern_years[pattern])} years")

        # Create Z500 composite
        z500_composite = z500_yearly.sel(year=pattern_years[pattern]).z.mean('year')

        # Calculate anomaly from climatology
        z500_anomaly = z500_composite - z500_climatology

        # Create figure with both maps
        fig = plt.figure(figsize=(12, 10))

        # 1. Plot absolute Z500 field
        ax1 = fig.add_subplot(211, projection=ccrs.PlateCarree())

        # Choose appropriate contour levels
        levels = np.arange(5000, 6000, 50)

        print(f"Z500 composite range: {z500_composite.min().values} to {z500_composite.max().values}")

        # Adjust the contour levels to match the actual data range
        vmin = float(z500_composite.min().values)
        vmax = float(z500_composite.max().values)
        levels = np.linspace(vmin, vmax, 21)

        # Plot Z500
        # Use pcolormesh instead of contourf for better rendering
        cf1 = ax1.pcolormesh(z500_composite.longitude, z500_composite.latitude, z500_composite,
                             cmap='jet', transform=ccrs.PlateCarree())

        # cf1 = ax1.contourf(z500_composite.longitude, z500_composite.latitude, z500_composite,
        #                   levels=levels, cmap='jet', transform=ccrs.PlateCarree())

        # Add contour lines levels=levels,
        cs1 = ax1.contour(z500_composite.longitude, z500_composite.latitude, z500_composite,
                          colors='black', linewidths=0.5, transform=ccrs.PlateCarree())

        # Label contours
        ax1.clabel(cs1, cs1.levels[::2], inline=True, fontsize=8, fmt='%d')

        # Add coastlines and borders
        ax1.add_feature(land)
        ax1.coastlines(resolution='50m')

        # Add gridlines
        gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl1.top_labels = False
        gl1.right_labels = False

        # Add title
        ax1.set_title(f'Z500 Composite for Pattern {pattern} (n={len(pattern_years[pattern])} years)')

        # 2. Plot Z500 anomaly
        ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree())

        # Find min/max for anomaly coloring
        vmax = max(20, np.max(np.abs(z500_anomaly.values)))
        levels = np.linspace(-vmax, vmax, 21)

        # Plot Z500 anomaly
        p_values = np.ones_like(z500_anomaly.values)

        # If you want to calculate statistical significance
        # This would require more data points than just the composite
        # You would need individual years' anomalies to do a proper t-test

        # Mask non-significant values (if p-values are calculated)
        masked_anomaly = np.ma.masked_where(p_values > 0.05, z500_anomaly.values)

        # Plot using masked array
        cf2 = ax2.pcolormesh(z500_anomaly.longitude, z500_anomaly.latitude, masked_anomaly,
                             cmap='RdBu_r', transform=ccrs.PlateCarree())

        # cf2 = ax2.contourf(z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
        #                   levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree())

        # Add contour lines
        cs2 = ax2.contour(z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                          levels=np.arange(-100, 101, 20), colors='black', linewidths=0.5,
                          transform=ccrs.PlateCarree())

        # Label contours
        ax2.clabel(cs2, cs2.levels[::2], inline=True, fontsize=8, fmt='%d')

        # Add coastlines and borders
        ax2.add_feature(land)
        ax2.coastlines(resolution='50m')

        # Add gridlines
        gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl2.top_labels = False
        gl2.right_labels = False

        # Add title
        ax2.set_title(f'Z500 Anomaly for Pattern {pattern}')

        # Add colorbars
        plt.colorbar(cf1, ax=ax1, orientation='horizontal', pad=0.05, label='Z500 (m)')
        plt.colorbar(cf2, ax=ax2, orientation='horizontal', pad=0.05, label='Z500 Anomaly (m)')

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(pattern_z500_dir, f'pattern_{pattern}_z500_composite.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Create summary table with pattern-year associations
    summary_data = []
    for pattern in all_patterns:
        summary_data.append({
            'Pattern': pattern,
            'Years': ', '.join(map(str, pattern_years.get(pattern, []))),
            'Count': len(pattern_years.get(pattern, []))
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(pattern_z500_dir, 'pattern_year_associations.csv'), index=False)

    return pattern_years


#######################################################################################################################
#######################################################################################################################

def analyze_z500_by_cluster(som, X, ds, clustering_results, z500_file, output_dir):
    """
    Analyze Z500 anomalies associated with each SOM cluster directly

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array (n_samples x n_features)
    ds : xarray.Dataset
        Original SST dataset with coordinates
    clustering_results : dict
        Results from cluster_som_patterns function
    z500_file : str
        Path to Z500 data file
    output_dir : str
        Directory to save results
    """
    # Create output directory
    cluster_z500_dir = os.path.join(output_dir, 'cluster_z500_associations')
    os.makedirs(cluster_z500_dir, exist_ok=True)

    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers (1-based)
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results['pattern_clusters']

    # Map patterns to clusters
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    years = times.year

    # Load Z500 data
    print("Loading Z500 data...")
    ds_z500 = xr.open_dataset(z500_file)

    # Select 500 hPa level if necessary
    if 'pressure_level' in ds_z500.dims:
        ds_z500 = ds_z500.sel(pressure_level=500.0)

    # Select DJF months
    z500_times = pd.to_datetime(ds_z500.valid_time.values)
    is_djf = (z500_times.month == 12) | (z500_times.month == 1) | (z500_times.month == 2)
    ds_z500_djf = ds_z500.isel(valid_time=is_djf)

    # Create year coordinate for seasonal averages
    year_coords = [t.year if t.month > 2 else t.year - 1
                   for t in pd.to_datetime(ds_z500_djf.valid_time.values)]
    ds_z500_djf = ds_z500_djf.assign_coords(year=('valid_time', year_coords))

    # Calculate seasonal means
    z500_yearly = ds_z500_djf.groupby('year').mean('valid_time')

    # Calculate climatology
    z500_climatology = z500_yearly.z.mean('year')

    # Find common years between SST and Z500
    unique_years = np.unique(years)
    common_years = sorted(list(set(unique_years).intersection(set(z500_yearly.year.values))))

    print(f"Found {len(common_years)} common years between SST and Z500 data")

    # Create mapping of year to dominant cluster
    year_cluster = {}
    for year in common_years:
        # Find dominant cluster for this year
        year_mask = (years == year)
        if np.any(year_mask):
            year_clusters = cluster_sequence[year_mask]
            cluster_counts = np.bincount(year_clusters)
            dominant_cluster = np.argmax(cluster_counts)
            year_cluster[year] = dominant_cluster

    # Get number of clusters
    n_clusters = clustering_results['n_clusters']

    # Create land feature for maps
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3
                                        )

    # For each cluster, create Z500 composite
    cluster_years = {}
    for cluster in sorted(set(pattern_clusters.values())):
        # Get years when this cluster was dominant
        cluster_years[cluster] = [year for year, clus in year_cluster.items() if clus == cluster]

        if len(cluster_years[cluster]) < 3:
            print(f"Cluster {cluster}: Not enough years for analysis ({len(cluster_years[cluster])})")
            continue

        print(f"Cluster {cluster}: {len(cluster_years[cluster])} years")

        # Create Z500 composite
        z500_composite = z500_yearly.sel(year=cluster_years[cluster]).z.mean('year')

        # Calculate anomaly from climatology
        z500_anomaly = z500_composite - z500_climatology

        # Create figure with both maps
        fig = plt.figure(figsize=(12, 10))

        # 1. Plot absolute Z500 field
        ax1 = fig.add_subplot(211, projection=ccrs.PlateCarree())

        # Choose appropriate contour levels
        levels = np.arange(5000, 6000, 50)

        # Plot Z500

        print(f"Z500 composite range: {z500_composite.min().values} to {z500_composite.max().values}")

        # Adjust the contour levels to match the actual data range
        vmin = float(z500_composite.min().values)
        vmax = float(z500_composite.max().values)
        levels = np.linspace(vmin, vmax, 21)  # Or use np.arange with appropriate step

        # Use pcolormesh instead of contourf for better rendering
        cf1 = ax1.pcolormesh(z500_composite.longitude, z500_composite.latitude, z500_composite,
                             cmap='jet', transform=ccrs.PlateCarree())

        # Add contour lines separately
        cs1 = ax1.contour(z500_composite.longitude, z500_composite.latitude, z500_composite,
                          colors='black', linewidths=0.5, transform=ccrs.PlateCarree())

        # Label contours
        ax1.clabel(cs1, cs1.levels[::2], inline=True, fontsize=8, fmt='%d')

        # Add coastlines and borders
        ax1.add_feature(land)
        ax1.coastlines(resolution='50m')

        # Add gridlines
        gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl1.top_labels = False
        gl1.right_labels = False

        # Add title
        ax1.set_title(f'Z500 Composite for Cluster {cluster} (n={len(cluster_years[cluster])} years)')

        # 2. Plot Z500 anomaly
        ax2 = fig.add_subplot(212, projection=ccrs.PlateCarree())

        # Find min/max for anomaly coloring
        vmax = max(20, np.max(np.abs(z500_anomaly.values)))
        levels = np.linspace(-vmax, vmax, 21)

        # Plot Z500 anomaly
        p_values = np.ones_like(z500_anomaly.values)

        # If you want to calculate statistical significance
        # This would require more data points than just the composite
        # You would need individual years' anomalies to do a proper t-test

        # Mask non-significant values (if p-values are calculated)
        masked_anomaly = np.ma.masked_where(p_values > 0.05, z500_anomaly.values)

        # Plot using masked array
        cf2 = ax2.pcolormesh(z500_anomaly.longitude, z500_anomaly.latitude, masked_anomaly,
                             cmap='RdBu_r', transform=ccrs.PlateCarree())

        # Add contour lines
        cs2 = ax2.contour(z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                          levels=np.arange(-100, 101, 20), colors='black', linewidths=0.5,
                          transform=ccrs.PlateCarree())

        # Label contours
        ax2.clabel(cs2, cs2.levels[::2], inline=True, fontsize=8, fmt='%d')

        # Add coastlines and borders
        ax2.add_feature(land)
        ax2.coastlines(resolution='50m')

        # Add gridlines
        gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl2.top_labels = False
        gl2.right_labels = False

        # Add title
        ax2.set_title(f'Z500 Anomaly for Cluster {cluster}')

        # Add colorbars
        plt.colorbar(cf1, ax=ax1, orientation='horizontal', pad=0.05, label='Z500 (m)')
        plt.colorbar(cf2, ax=ax2, orientation='horizontal', pad=0.05, label='Z500 Anomaly (m)')

        # Adjust layout
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(cluster_z500_dir, f'cluster_{cluster}_z500_composite.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Create summary table with cluster-year associations
    summary_data = []
    for cluster in sorted(set(pattern_clusters.values())):
        summary_data.append({
            'Cluster': cluster,
            'Years': ', '.join(map(str, cluster_years.get(cluster, []))),
            'Count': len(cluster_years.get(cluster, []))
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(cluster_z500_dir, 'cluster_year_associations.csv'), index=False)

    return cluster_years


#######################################################################################################################
#######################################################################################################################

def create_combined_cluster_view(clustering_results, som, ds, output_dir):
    """
    Create a single 2x2 panel figure showing all cluster patterns with enhanced contrast

    Parameters:
    -----------
    clustering_results : dict
        Results from cluster_som_patterns
    som : MiniSom
        Trained SOM model
    ds : xarray.Dataset
        Original dataset with coordinates
    output_dir : str
        Directory to save output plot
    """
    # Create output directory
    os.makedirs(os.path.join(output_dir, 'cluster_visualization'), exist_ok=True)

    # Get pattern to cluster mapping and number of clusters
    pattern_clusters = clustering_results["pattern_clusters"]
    unique_clusters = sorted(set(pattern_clusters.values()))
    n_clusters = clustering_results["n_clusters"]

    print(f"Creating combined view for {n_clusters} clusters: {unique_clusters}")

    # Get SOM weights
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    # Create coordinate meshgrid
    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

    # Prepare figure layout - 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 12),
                            subplot_kw={'projection': ccrs.PlateCarree()})
    axs = axs.flatten()  # Flatten to easily iterate

    # Create a list to collect all pattern data for global scaling
    all_patterns = []

    # First collect representative patterns for each cluster
    rep_patterns = {}
    for cluster in sorted(set(pattern_clusters.values())):
        # Get patterns in this cluster
        cluster_patterns = [idx for idx, c in pattern_clusters.items() if c == cluster]

        if cluster_stats := clustering_results.get("cluster_stats", {}):
            # If we have representative patterns from clustering
            if 'representative_patterns' in cluster_stats and cluster in cluster_stats['representative_patterns']:
                rep_idx = cluster_stats['representative_patterns'][cluster]
                i, j = (rep_idx - 1) // n_cols, (rep_idx - 1) % n_cols
                pattern = weights[i, j].reshape(ds.sst.shape[1:])
                rep_patterns[cluster] = pattern
                all_patterns.append(pattern)
        else:
            # If no representative pattern is defined, create a composite
            cluster_sst = np.zeros(ds.sst.shape[1:])
            for pattern_idx in cluster_patterns:
                i, j = (pattern_idx - 1) // n_cols, (pattern_idx - 1) % n_cols
                pattern = weights[i, j].reshape(ds.sst.shape[1:])
                cluster_sst += pattern

            if len(cluster_patterns) > 0:
                cluster_sst /= len(cluster_patterns)
                rep_patterns[cluster] = cluster_sst
                all_patterns.append(cluster_sst)

    # Determine global min/max for consistent color scaling
    if all_patterns:
        vmin = min(p.min() for p in all_patterns)
        vmax = max(p.max() for p in all_patterns)
        abs_max = max(abs(vmin), abs(vmax))
    else:
        abs_max = 1.0  # Fallback

    # Create enhanced colormap - RdBu_r with white center change or maybe to seismic maybe
    enhanced_cmap = plt.cm.RdBu_r.copy()
    enhanced_cmap.set_bad('gray')

    # Plot each cluster
    for cluster in sorted(set(pattern_clusters.values())):
        if cluster >= len(axs):  # Safety check
            continue

        ax = axs[cluster]

        if cluster in rep_patterns:
            pattern = rep_patterns[cluster]

            # Get patterns in this cluster for title
            cluster_patterns = [idx for idx, c in pattern_clusters.items() if c == cluster]
            pattern_list = ', '.join(map(str, cluster_patterns))

            # Plot with enhanced contrast
            im = ax.pcolormesh(lons, lats, pattern,
                               transform=ccrs.PlateCarree(),
                               cmap=enhanced_cmap,
                               vmin=-abs_max, vmax=abs_max,
                               shading='auto')

            # Get cohesion if available
            cohesion = "N/A"
            if cluster_stats and 'cluster_cohesion' in cluster_stats and cluster in cluster_stats['cluster_cohesion']:
                cohesion = f"{cluster_stats['cluster_cohesion'][cluster]:.3f}"

            # Add title
            rep_pattern_id = "Composite"
            if cluster_stats and 'representative_patterns' in cluster_stats and cluster in cluster_stats[
                'representative_patterns']:
                rep_pattern_id = cluster_stats['representative_patterns'][cluster]

            ax.set_title(f"Cluster {cluster} - Representative Pattern {rep_pattern_id}\n"
                         f"Size: {len(cluster_patterns)} patterns, Cohesion: {cohesion}\n"
                         f"Patterns: {pattern_list}")
        else:
            # Empty cluster
            ax.text(0.5, 0.5, f"Cluster {cluster} - No patterns",
                    transform=ax.transAxes, ha='center', va='center')

        # Add land and coastlines
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Set common extent
        ax.set_extent([ds.longitude.min(), ds.longitude.max(),
                       ds.latitude.min(), ds.latitude.max()],
                      crs=ccrs.PlateCarree())

    # Add single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cbar_ax, label='Normalized SST Anomaly')

    # Adjust spacing between subplots
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save the figure
    fig_path = os.path.join(output_dir, 'cluster_visualization', 'clusters_2x2_panel.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Combined cluster visualization saved to: {fig_path}")
    return fig_path


#######################################################################################################################
#######################################################################################################################

def create_cluster_indices_for_z500_correlation(som, X, ds, clustering_results, z500_file, output_dir):
    """
    Calculate cluster-based indices and correlate them with Z500 geopotential height data

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array (n_samples x n_features)
    ds : xarray.Dataset
        Original SST dataset with coordinates
    clustering_results : dict
        Results from cluster_som_patterns function
    z500_file : str
        Path to Z500 data file
    output_dir : str
        Directory to save results

    Returns:
    --------
    dict
        Dictionary with cluster indices and correlation results
    """
    print("\nCalculating cluster-based indices for Z500 correlation...")

    # Create output directory
    indices_dir = os.path.join(output_dir, 'cluster_indices_z500')
    os.makedirs(indices_dir, exist_ok=True)

    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers (1-based)
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get pattern to cluster mapping
    if 'pattern_clusters' not in clustering_results:
        print("Error: No pattern_clusters in clustering results")
        return None

    pattern_clusters = clustering_results['pattern_clusters']

    # Map patterns to clusters (should be 1-indexed)
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Group by year & month to create cluster frequency time series
    df = pd.DataFrame({
        'year': years,
        'month': months,
        'cluster': cluster_sequence
    })

    # Filter to DJF months for annual indices
    is_djf = (df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)
    df_djf = df[is_djf].copy()

    # Create "year" for DJF seasons (using year of January)
    def get_season_year(row):
        if row['month'] == 12:  # December belongs to next year's DJF
            return row['year'] + 1
        return row['year']

    df_djf['season_year'] = df_djf.apply(get_season_year, axis=1)

    # Calculate cluster frequency by season
    seasonal_counts = df_djf.groupby(['season_year', 'cluster']).size().unstack(fill_value=0)
    seasonal_freq = seasonal_counts.div(seasonal_counts.sum(axis=1), axis=0)

    # Get unique clusters (should be 1-indexed)
    unique_clusters = sorted(set(pattern_clusters.values()))

    # Create indices dictionary
    indices = {}

    # 1. Simple indices - frequency of each cluster
    for cluster in unique_clusters:
        if cluster in seasonal_freq.columns:
            indices[f'cluster_{cluster}_freq'] = seasonal_freq[cluster]

    # 2. Dipole indices - differences between clusters
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            cluster_i = unique_clusters[i]
            cluster_j = unique_clusters[j]
            if cluster_i in seasonal_freq.columns and cluster_j in seasonal_freq.columns:
                indices[f'cluster_{cluster_i}_minus_{cluster_j}_dipole'] = seasonal_freq[cluster_i] - seasonal_freq[
                    cluster_j]

    # 3. Standardized versions of all indices
    for name, index in list(indices.items()):
        std_index = (index - index.mean()) / index.std()
        indices[f'{name}_std'] = std_index

    # Save indices to CSV
    indices_df = pd.DataFrame(indices)
    indices_df.to_csv(os.path.join(indices_dir, 'cluster_indices.csv'))

    # Load Z500 data for correlation
    print("Loading Z500 data for correlation...")
    ds_z500 = xr.open_dataset(z500_file)

    # Select 500 hPa level if necessary
    if 'pressure_level' in ds_z500.dims:
        ds_z500 = ds_z500.sel(pressure_level=500.0)

    # Select DJF months
    z500_times = pd.to_datetime(ds_z500.valid_time.values)
    is_djf = (z500_times.month == 12) | (z500_times.month == 1) | (z500_times.month == 2)
    ds_z500_djf = ds_z500.isel(valid_time=is_djf)

    # Create season year for Z500 data
    year_coords = []
    for t in pd.to_datetime(ds_z500_djf.valid_time.values):
        if t.month == 12:
            year_coords.append(t.year + 1)
        else:
            year_coords.append(t.year)

    ds_z500_djf = ds_z500_djf.assign_coords(season_year=('valid_time', year_coords))

    # Calculate seasonal means
    z500_seasonal = ds_z500_djf.groupby('season_year').mean('valid_time')

    # Find common years between indices and Z500
    common_years = sorted(list(set(indices_df.index).intersection(set(z500_seasonal.season_year.values))))

    if len(common_years) < 3:
        print(f"Warning: Only {len(common_years)} common years between indices and Z500 data")
        return indices

    print(f"Computing correlations using {len(common_years)} years of common data")

    # Calculate correlation maps for each index
    correlation_results = {}

    # Create a land feature for maps
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)

    for index_name, index_values in indices.items():
        # Only process standardized indices
        if not index_name.endswith('_std'):
            continue

        # Get index values for common years
        index_subset = index_values.loc[common_years]

        # Select Z500 data for common years
        z500_subset = z500_seasonal.sel(season_year=common_years)

        # Initialize correlation map
        corr_map = np.zeros(z500_subset.z.isel(season_year=0).shape)
        p_map = np.ones(corr_map.shape)

        # Calculate point-by-point correlation
        for i in range(corr_map.shape[0]):
            for j in range(corr_map.shape[1]):
                # Extract Z500 time series at this grid point
                z_series = z500_subset.z[:, i, j].values

                # Skip if NaNs are present
                if np.isnan(z_series).any() or np.isnan(index_subset.values).any():
                    corr_map[i, j] = np.nan
                    continue

                # Calculate correlation
                r, p = pearsonr(index_subset.values, z_series)
                corr_map[i, j] = r
                p_map[i, j] = p

        # Store result
        correlation_results[index_name] = {
            'corr_map': corr_map,
            'p_map': p_map,
            'lats': z500_subset.latitude.values,
            'lons': z500_subset.longitude.values
        }

        # Create correlation plot
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Determine color scale
        vmax = min(0.8, np.nanmax(np.abs(corr_map)))

        # Mask non-significant correlations
        masked_corr = np.ma.masked_where(p_map > 0.05, corr_map)

        # Plot correlation
        cf = ax.contourf(z500_subset.longitude, z500_subset.latitude, masked_corr,
                         levels=np.linspace(-vmax, vmax, 17),
                         cmap='RdBu_r', transform=ccrs.PlateCarree())

        # Add contour lines for all data
        cs = ax.contour(z500_subset.longitude, z500_subset.latitude, corr_map,
                        levels=[-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
                        colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

        ax.clabel(cs, cs.levels, inline=True, fontsize=8, fmt='%.1f')

        # Add coastlines and borders
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xlines = True
        gl.ylines = True
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        # Add colorbar and title
        plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05,
                     label='Correlation Coefficient (p < 0.05)')

        ax.set_title(f'Z500 Correlation with {index_name}\n(n={len(common_years)} years)')

        # Save figure
        plt.savefig(os.path.join(indices_dir, f'z500_corr_{index_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Save correlation results
    with open(os.path.join(indices_dir, 'z500_correlations.pkl'), 'wb') as f:
        pickle.dump(correlation_results, f)

    print(f"Cluster indices and Z500 correlations saved to: {indices_dir}")
    return indices


#######################################################################################################################
#######################################################################################################################

def create_combined_z500_view(som, X, ds, clustering_results, z500_file, output_dir):
    """
    Create combined visualization of Z500 composites and anomalies for all clusters
    """
    # Create output directory
    z500_dir = os.path.join(output_dir, 'combined_z500_visualization')
    os.makedirs(z500_dir, exist_ok=True)

    # Get BMUs, patterns, and clusters
    bmu_indices = np.array([som.winner(x) for x in X])
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])
    pattern_clusters = clustering_results["pattern_clusters"]
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    years = times.year

    # Load Z500 data
    print("Loading Z500 data for combined visualization...")
    ds_z500 = xr.open_dataset(z500_file)
    if 'pressure_level' in ds_z500.dims:
        ds_z500 = ds_z500.sel(pressure_level=500.0)

    # Process Z500 data
    z500_times = pd.to_datetime(ds_z500.valid_time.values)
    is_djf = (z500_times.month == 12) | (z500_times.month == 1) | (z500_times.month == 2)
    ds_z500_djf = ds_z500.isel(valid_time=is_djf)

    year_coords = [t.year if t.month > 2 else t.year - 1
                   for t in pd.to_datetime(ds_z500_djf.valid_time.values)]
    ds_z500_djf = ds_z500_djf.assign_coords(year=('valid_time', year_coords))

    z500_yearly = ds_z500_djf.groupby('year').mean('valid_time')
    z500_climatology = z500_yearly.z.mean('year')

    # Get common years and create year-to-cluster mapping
    common_years = sorted(list(set(np.unique(years)).intersection(set(z500_yearly.year.values))))
    year_cluster = {}
    for year in common_years:
        year_mask = (years == year)
        if np.any(year_mask):
            year_clusters = cluster_sequence[year_mask]
            # Use proper handling for 1-indexed clusters
            unique_year_clusters = np.unique(year_clusters)
            cluster_counts = np.zeros(max(unique_year_clusters) + 1)
            for cluster in year_clusters:
                cluster_counts[cluster] += 1
            # Find dominant cluster (skip index 0)
            dominant_cluster = np.argmax(cluster_counts[1:]) + 1
            year_cluster[year] = dominant_cluster

    # Get unique clusters that actually exist (1-indexed)
    unique_clusters = sorted(set(pattern_clusters.values()))
    n_clusters = len(unique_clusters)

    # Initialize cluster_years dictionary
    cluster_years = {}

    # Create land feature for maps
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)

    # Prepare composite data for all clusters
    anomaly_data = {}  # Use a dictionary keyed by cluster number, not a list

    for cluster in unique_clusters:
        # Get years for this cluster
        cluster_years[cluster] = [year for year, clus in year_cluster.items() if clus == cluster]

        if len(cluster_years[cluster]) < 3:
            print(f"Cluster {cluster}: Not enough years for analysis ({len(cluster_years[cluster])})")
            anomaly_data[cluster] = None
            continue

        print(f"Cluster {cluster}: {len(cluster_years[cluster])} years")
        z500_composite = z500_yearly.sel(year=cluster_years[cluster]).z.mean('year')
        z500_anomaly = z500_composite - z500_climatology

        anomaly_data[cluster] = z500_anomaly

    # Create figure for anomalies only
    fig = plt.figure(figsize=(18, 6))  # Adjusted size for single row

    # Find global min/max for anomalies
    valid_anomalies = [anom for anom in anomaly_data.values() if anom is not None]
    if valid_anomalies:
        anom_max = max(np.max(np.abs(anom.values)) for anom in valid_anomalies)
    else:
        anom_max = 100  # Fallback

    # Plot each anomaly
    for i, cluster in enumerate(unique_clusters):
        if anomaly_data[cluster] is None:
            continue

        anomaly = anomaly_data[cluster]

        # Set up subplot position - 1-based subplot index
        subplot_idx = i + 1
        ax = fig.add_subplot(1, n_clusters, subplot_idx, projection=ccrs.PlateCarree())

        # Plot the anomaly
        cf = ax.pcolormesh(anomaly.longitude, anomaly.latitude, anomaly,
                           cmap='RdBu_r', vmin=-anom_max, vmax=anom_max,
                           transform=ccrs.PlateCarree())

        # Add contour lines
        cs = ax.contour(anomaly.longitude, anomaly.latitude, anomaly,
                        levels=np.arange(-100, 101, 20), colors='black', linewidths=0.5,
                        transform=ccrs.PlateCarree())

        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%d')

        # Add coastlines and land
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Add title with cluster info
        ax.set_title(f'Z500 Anomaly for Cluster {cluster}')

        # Add panel label
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    plt.colorbar(cf, cax=cbar_ax, label='Z500 Anomaly (m)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(z500_dir, 'combined_z500_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    return z500_dir


#######################################################################################################################
#######################################################################################################################

def create_grid_correlation_plots(correlation_results, output_dir):
    """
    Create 2×2 grid visualizations of Z500 correlations with cluster indices

    Parameters:
    -----------
    correlation_results : dict
        Dictionary containing correlation maps and metadata
    output_dir : str
        Directory to save output plots
    """

    # Create subdirectory for grid plots
    grid_plots_dir = os.path.join(output_dir, 'grid_correlation_plots')
    os.makedirs(grid_plots_dir, exist_ok=True)

    # Get land feature for maps
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3
                                        )

    # Separate frequency and dipole indices
    freq_indices = [k for k in correlation_results.keys() if k.endswith('_freq_std')]
    dipole_indices = [k for k in correlation_results.keys() if 'dipole_std' in k]

    # Create 2×2 grid for frequency correlations
    if len(freq_indices) > 0:
        fig_freq = plt.figure(figsize=(12, 10))

        # Find global min/max for consistent color scale
        corr_values = []
        for idx in freq_indices[:4]:  # Use at most 4 indices
            if idx in correlation_results:
                corr_values.extend(correlation_results[idx]['corr_map'].flatten())

        corr_values = [v for v in corr_values if not np.isnan(v)]
        if corr_values:
            vmax = min(0.8, max(abs(min(corr_values)), abs(max(corr_values))))
        else:
            vmax = 0.5

        # Plot each correlation map
        for i, idx in enumerate(freq_indices[:4]):  # Use at most 4 indices
            if idx not in correlation_results:
                continue

            ax = fig_freq.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

            corr_map = correlation_results[idx]['corr_map']
            p_map = correlation_results[idx]['p_map']
            lats = correlation_results[idx]['lats']
            lons = correlation_results[idx]['lons']

            # Mask non-significant correlations
            masked_corr = np.ma.masked_where(p_map > 0.05, corr_map)

            # Plot correlation
            cf = ax.contourf(lons, lats, masked_corr,
                             levels=np.linspace(-vmax, vmax, 17),
                             cmap='RdBu_r', transform=ccrs.PlateCarree(),
                             extend='both')

            # Add contour lines for all data
            cs = ax.contour(lons, lats, corr_map,
                            levels=[-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
                            colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

            ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.1f')

            # Add coastlines and land
            ax.add_feature(land)
            ax.coastlines(resolution='50m')

            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False  # Remove longitude labels at the top
            gl.right_labels = False  # Remove latitude labels on the right
            gl.bottom_labels = True  # Ensure longitude labels at the bottom are shown
            gl.left_labels = True  # Ensure latitude labels on the left are shown
            gl.xlines = True  # Show vertical gridlines
            gl.ylines = True  # Show horizontal gridlines
            gl.xlabel_style = {'size': 8}  # Customize longitude label size
            gl.ylabel_style = {'size': 8}  # Customize latitude label size

            # Extract cluster number for title
            if '_freq_std' in idx:
                cluster_num = idx.split('_')[2]
                title = f'Cluster {cluster_num} Frequency'
            else:
                title = idx.replace('_std', '')

            ax.set_title(title)

        # Add common colorbar
        cbar_ax = fig_freq.add_axes([0.92, 0.15, 0.02, 0.7])
        plt.colorbar(cf, cax=cbar_ax, label='Correlation Coefficient (p < 0.05)')

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(os.path.join(grid_plots_dir, 'frequency_correlations_grid.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig_freq)

    # Create 2×2 grid for dipole correlations
    if len(dipole_indices) > 0:
        for start_idx in range(0, len(dipole_indices), 4):
            fig_dipole = plt.figure(figsize=(12, 10))

            # Find global min/max for consistent color scale
            dipole_subset = dipole_indices[start_idx:start_idx + 4]
            corr_values = []
            for idx in dipole_subset:
                if idx in correlation_results:
                    corr_values.extend(correlation_results[idx]['corr_map'].flatten())

            corr_values = [v for v in corr_values if not np.isnan(v)]
            if corr_values:
                vmax = min(0.8, max(abs(min(corr_values)), abs(max(corr_values))))
            else:
                vmax = 0.5

            # Plot each correlation map
            for i, idx in enumerate(dipole_subset):
                if idx not in correlation_results:
                    continue

                ax = fig_dipole.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

                corr_map = correlation_results[idx]['corr_map']
                p_map = correlation_results[idx]['p_map']
                lats = correlation_results[idx]['lats']
                lons = correlation_results[idx]['lons']

                # Mask non-significant correlations
                masked_corr = np.ma.masked_where(p_map > 0.05, corr_map)

                # Plot correlation
                cf = ax.contourf(lons, lats, masked_corr,
                                 levels=np.linspace(-vmax, vmax, 17),
                                 cmap='RdBu_r', transform=ccrs.PlateCarree(),
                                 extend='both')

                # Add contour lines for all data
                cs = ax.contour(lons, lats, corr_map,
                                levels=[-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
                                colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

                ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.1f')

                # Add coastlines and land
                ax.add_feature(land)
                ax.coastlines(resolution='50m')

                # Add gridlines
                gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                gl.bottom_labels = True

                gl.xlines = True
                gl.ylines = True

                gl.xlabel_style = {'size': 8}  #
                gl.ylabel_style = {'size': 8}

                # Extract cluster numbers for title
                if 'dipole' in idx:
                    parts = idx.split('_')
                    c1, c2 = parts[2], parts[4]
                    title = f'Cluster {c1} - Cluster {c2}'
                else:
                    title = idx.replace('_std', '')

                ax.set_title(title)

            # Add common colorbar
            cbar_ax = fig_dipole.add_axes([0.92, 0.15, 0.02, 0.7])
            plt.colorbar(cf, cax=cbar_ax, label='Correlation Coefficient (p < 0.05)')

            plt.tight_layout(rect=[0, 0, 0.9, 1])
            plot_num = start_idx // 4 + 1
            plt.savefig(os.path.join(grid_plots_dir, f'dipole_correlations_grid_{plot_num}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig_dipole)

    return grid_plots_dir


#######################################################################################################################
#######################################################################################################################

def create_improved_correlation_plots(correlation_results, clustering_results, som, ds, output_dir):
    """
    Create enhanced visualization of Z500 correlations with cluster frequencies

    Parameters:
    -----------
    correlation_results : dict
        Dictionary containing correlation maps and metadata
    clustering_results : dict
        Results from cluster_som_patterns
    som : MiniSom
        Trained SOM model
    ds : xarray.Dataset
        Original dataset with coordinates
    output_dir : str
        Directory to save output
    """
    # Create output directory
    corr_dir = os.path.join(output_dir, 'improved_correlation_plots')
    os.makedirs(corr_dir, exist_ok=True)

    # Create land feature for maps
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)

    # Get only cluster frequency indices
    freq_indices = [k for k in correlation_results.keys() if k.endswith('_freq_std')]

    # Get SOm weights and number of clusters
    weights = som.get_weights()
    n_clusters = clustering_results['n_clusters']

    # Create the figure
    fig = plt.figure(figsize=(15, 12))

    # Find global min/max for consistent color scale
    corr_values = []
    for idx in freq_indices:
        if idx in correlation_results:
            corr_values.extend(correlation_results[idx]['corr_map'].flatten())

    corr_values = [v for v in corr_values if not np.isnan(v)]
    if corr_values:
        vmax = min(0.8, max(abs(min(corr_values)), abs(max(corr_values))))
    else:
        vmax = 0.5

    # Plot each correlation map
    for i, idx in enumerate(freq_indices[:n_clusters]):  # Limit to number of clusters
        if idx not in correlation_results:
            continue

        # Main plot
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

        corr_map = correlation_results[idx]['corr_map']
        p_map = correlation_results[idx]['p_map']
        lats = correlation_results[idx]['lats']
        lons = correlation_results[idx]['lons']

        # Mask non-significant correlations for coloring
        masked_corr = np.ma.masked_where(p_map > 0.05, corr_map)

        # Plot correlation
        cf = ax.contourf(lons, lats, masked_corr,
                         levels=np.linspace(-vmax, vmax, 17),
                         cmap='RdBu_r', transform=ccrs.PlateCarree(),
                         extend='both')

        # Add contour lines for all data (both significant and non-significant)
        cs = ax.contour(lons, lats, corr_map,
                        levels=[-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
                        colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.1f')

        # Add stippling for significant areas
        significance_mask = p_map < 0.05
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
        significant_lons = lons_mesh[significance_mask]
        significant_lats = lats_mesh[significance_mask]
        stride = 5  # Adjust stride to control density of stippling
        ax.scatter(significant_lons[::stride], significant_lats[::stride],
                   color='black', s=1, alpha=0.7, transform=ccrs.PlateCarree())

        # Add coastlines and borders
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Extract cluster number for title
        cluster_num = idx.split('_')[2]

        # Add title with physical interpretation
        physical_patterns = ["NAO+ Pattern", "NAO- Pattern", "Blocking Pattern", "Subtropical High"]
        if i < len(physical_patterns):
            title = f"Cluster {cluster_num} Frequency\n({physical_patterns[i]})"
        else:
            title = f"Cluster {cluster_num} Frequency"

        ax.set_title(title)

        # Add panel label
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

        # Add inset with SST pattern
        if cluster_num.isdigit():
            cluster_idx = int(cluster_num)
            # Get SST pattern for this cluster
            if cluster_idx < n_clusters:
                # Get patterns in this cluster
                pattern_clusters = clustering_results['pattern_clusters']
                cluster_patterns = [idx for idx, c in pattern_clusters.items() if c == cluster_idx]

                if cluster_patterns:
                    # Create an inset axes for the SST pattern
                    inset_ax = fig.add_axes([ax.get_position().x0 + 0.02,
                                             ax.get_position().y0 + 0.02,
                                             0.10, 0.10],
                                            projection=ccrs.PlateCarree())

                    # Create composite SST pattern
                    pattern_sum = np.zeros(ds.sst.shape[1:])
                    for pattern_idx in cluster_patterns:
                        i_idx, j_idx = (pattern_idx - 1) // weights.shape[1], (pattern_idx - 1) % weights.shape[1]
                        pattern = weights[i_idx, j_idx].reshape(ds.sst.shape[1:])
                        pattern_sum += pattern

                    if len(cluster_patterns) > 0:
                        pattern_avg = pattern_sum / len(cluster_patterns)

                        # Plot SST pattern
                        inset_ax.pcolormesh(ds.longitude, ds.latitude, pattern_avg,
                                            cmap='RdBu_r', transform=ccrs.PlateCarree())
                        inset_ax.coastlines(resolution='50m', linewidth=0.5)
                        inset_ax.set_extent([ds.longitude.min(), ds.longitude.max(),
                                             ds.latitude.min(), ds.latitude.max()],
                                            crs=ccrs.PlateCarree())

                        # Turn off axis labels for inset
                        inset_ax.set_xticks([])
                        inset_ax.set_yticks([])

        # Add text box with key correlations
        # Find max correlation points (absolute value)
        abs_corr = np.abs(corr_map)
        max_idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
        max_lat, max_lon = lats[max_idx[0]], lons[max_idx[1]]
        max_corr = corr_map[max_idx]

        # Add text box
        ax.text(0.02, 0.08, f"Max corr: {max_corr:.2f}\nLat: {max_lat:.1f}°, Lon: {max_lon:.1f}°",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = plt.colorbar(cf, cax=cbar_ax, label='Correlation Coefficient (p < 0.05)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(corr_dir, 'cluster_frequency_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Improved correlation plots saved to: {corr_dir}")
    return corr_dir


#######################################################################################################################
#######################################################################################################################

def create_sst_cluster_composite_figure(clustering_results, som, ds, output_dir):
    """
    Create a four-panel figure showing SST anomaly patterns for each cluster

    Parameters:
    -----------
    clustering_results : dict
        Results from cluster_som_patterns
    som : MiniSom
        Trained SOM model
    ds : xarray.Dataset
        Original dataset with coordinates
    output_dir : str
        Directory to save output
    """
    # Create output directory
    sst_dir = os.path.join(output_dir, 'sst_cluster_patterns')
    os.makedirs(sst_dir, exist_ok=True)

    # Get SOM weights and number of clusters
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape
    n_clusters = clustering_results['n_clusters']

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results['pattern_clusters']

    # Create the figure
    fig = plt.figure(figsize=(16, 12))

    # Create land feature
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray')

    # Prepare data for all clusters
    all_patterns = []
    for cluster in sorted(set(pattern_clusters.values())):
        # Get patterns in this cluster
        cluster_patterns = [idx for idx, c in pattern_clusters.items() if c == cluster]

        # Create composite SST pattern
        pattern_sum = np.zeros(ds.sst.shape[1:])
        for pattern_idx in cluster_patterns:
            i_idx, j_idx = (pattern_idx - 1) // n_cols, (pattern_idx - 1) % n_cols
            pattern = weights[i_idx, j_idx].reshape(ds.sst.shape[1:])
            pattern_sum += pattern

        if len(cluster_patterns) > 0:
            pattern_avg = pattern_sum / len(cluster_patterns)
            all_patterns.append(pattern_avg)

    # Find global min/max for consistent color scaling
    if all_patterns:
        vmin = min(p.min() for p in all_patterns)
        vmax = max(p.max() for p in all_patterns)
        abs_max = max(abs(vmin), abs(vmax))
    else:
        abs_max = 1.0  # Fallback

    # Plot each cluster's SST pattern
    for i, pattern in enumerate(all_patterns):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

        # Get patterns in this cluster for title
        cluster_patterns = [idx for idx, c in pattern_clusters.items() if c == i]

        # Plot with RdBu_r colormap centered at zero
        im = ax.pcolormesh(ds.longitude, ds.latitude, pattern,
                           transform=ccrs.PlateCarree(),
                           cmap='RdBu_r',
                           vmin=-abs_max, vmax=abs_max,
                           shading='auto')

        # Add contour lines
        cs = ax.contour(ds.longitude, ds.latitude, pattern,
                        colors='black', linewidths=0.5,
                        transform=ccrs.PlateCarree())

        ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

        # Add coastlines and borders
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Physical interpretations - placeholder, update with your actual interpretations
        interpretations = [
            "Gulf Stream Displacement",
            "Negative AMO-like Pattern",
            "Tripole Pattern",
            "Subtropical Warming"
        ]

        # Add title
        if i < len(interpretations):
            title = f"Cluster {i} SST Pattern (n={len(cluster_patterns)} patterns)\n{interpretations[i]}"
        else:
            title = f"Cluster {i} SST Pattern (n={len(cluster_patterns)} patterns)"

        ax.set_title(title)

        # Add panel label
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax, label='Normalized SST Anomaly')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(sst_dir, 'sst_cluster_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"SST cluster pattern figure saved to: {sst_dir}")
    return sst_dir


#######################################################################################################################
#######################################################################################################################

# def enhance_z500_anomaly_plots(clustering_results, output_dir, z500_dir):
# """
# Add additional statistical information to existing Z500 anomaly plots
# """
# This function would add:
# 1. Percentage of variance explained by each pattern
# 2. Mean duration of each cluster
# 3. Trend information (if any clusters show significant trends)
# 4. Relationship to known indices (NAO, EA, etc.)

# Implementation details would depend on your existing Z500 plots
# and what additional statistics you've calculated

def enhance_z500_anomaly_plots(cluster_years, z500_yearly, z500_climatology, clustering_results,
                               time_series_results, output_dir):
    """
    Create enhanced Z500 anomaly plots with additional statistical information

    Parameters:
    -----------
    cluster_years : dict
        Dictionary mapping clusters to years
    z500_yearly : xarray.Dataset
        Yearly Z500 data
    z500_climatology : xarray.DataArray
        Z500 climatology
    clustering_results : dict
        Results from cluster_som_patterns
    time_series_results : dict
        Results from analyze_pattern_time_series
    output_dir : str
        Directory to save output
    """
    # Create output directory
    enhanced_z500_dir = os.path.join(output_dir, 'enhanced_z500_analysis')
    os.makedirs(enhanced_z500_dir, exist_ok=True)

    # Create land feature
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)

    # Get number of clusters
    n_clusters = clustering_results['n_clusters']

    # Calculate additional statistics for each cluster
    cluster_stats = {}

    for cluster in sorted(set(pattern_clusters.values())):
        # Skip if not enough years
        if len(cluster_years.get(cluster, [])) < 3:
            continue

        # Years where this cluster was dominant
        years = cluster_years[cluster]

        # Calculate mean duration (consecutive years with same cluster)
        durations = []
        current_duration = 1
        for i in range(1, len(years)):
            if years[i] == years[i - 1] + 1:  # Consecutive years
                current_duration += 1
            else:
                durations.append(current_duration)
                current_duration = 1
        durations.append(current_duration)  # Add the last streak

        # Calculate trend if enough data
        if len(years) >= 10:
            try:
                # Simple linear trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    np.arange(len(years)), years
                )
                trend_significant = p_value < 0.05
                trend_per_decade = slope * 10
            except:
                trend_significant = False
                trend_per_decade = 0
        else:
            trend_significant = False
            trend_per_decade = 0

        # Store statistics
        cluster_stats[cluster] = {
            'years': years,
            'count': len(years),
            'mean_duration': np.mean(durations) if durations else 0,
            'trend_per_decade': trend_per_decade,
            'trend_significant': trend_significant
        }

    # Create enhanced plots
    fig = plt.figure(figsize=(16, 12))

    # Prepare composite data for all clusters
    anomaly_data = []
    all_anomalies = []

    for cluster in sorted(set(pattern_clusters.values())):
        if cluster not in cluster_stats:
            anomaly_data.append(None)
            continue

        # Calculate Z500 composite and anomaly
        z500_composite = z500_yearly.sel(year=cluster_stats[cluster]['years']).z.mean('year')
        z500_anomaly = z500_composite - z500_climatology
        anomaly_data.append(z500_anomaly)
        all_anomalies.append(z500_anomaly)

    # Find global min/max for anomalies
    if all_anomalies:
        anom_max = max(np.max(np.abs(anom.values)) for anom in all_anomalies if anom is not None)
    else:
        anom_max = 100  # Fallback

    # Physical pattern names
    pattern_names = {
        0: "Positive NAO-like",
        1: "Negative NAO-like",
        2: "Arctic Blocking",
        3: "Subtropical Ridge"
    }

    # Plot each anomaly
    for i, anomaly in enumerate(anomaly_data):
        if anomaly is None:
            continue

        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

        # Calculate statistical significance
        all_year_anomalies = []
        for year in cluster_stats[i]['years']:
            year_anomaly = z500_yearly.sel(year=year).z - z500_climatology
            all_year_anomalies.append(year_anomaly)

        if all_year_anomalies:
            all_years_anom = xr.concat(all_year_anomalies, dim='year')
            t_stat = (anomaly / (all_years_anom.std('year') / np.sqrt(len(cluster_stats[i]['years']))))
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(cluster_stats[i]['years']) - 1))
            significance_mask = p_values < 0.05
        else:
            significance_mask = np.zeros_like(anomaly.values, dtype=bool)

        # Plot the anomaly
        cf = ax.pcolormesh(anomaly.longitude, anomaly.latitude, anomaly,
                           cmap='RdBu_r', vmin=-anom_max, vmax=anom_max,
                           transform=ccrs.PlateCarree())

        # Add contour lines
        cs = ax.contour(anomaly.longitude, anomaly.latitude, anomaly,
                        levels=np.arange(-100, 101, 20), colors='black', linewidths=0.5,
                        transform=ccrs.PlateCarree())

        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%d')

        # Add stippling for significance
        lons_mesh, lats_mesh = np.meshgrid(anomaly.longitude, anomaly.latitude)
        significant_lons = lons_mesh[significance_mask]
        significant_lats = lats_mesh[significance_mask]
        stride = 5
        ax.scatter(significant_lons[::stride], significant_lats[::stride],
                   color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

        # Add coastlines and borders
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Get pattern name
        pattern_name = pattern_names.get(i, f"Cluster {i}")

        # Add title with statistics
        title = (f"Z500 Anomaly: {pattern_name}\n"
                 f"n={cluster_stats[i]['count']} years, "
                 f"Mean duration={cluster_stats[i]['mean_duration']:.1f} years")

        if cluster_stats[i]['trend_significant']:
            title += f", Trend={cluster_stats[i]['trend_per_decade']:.2f}/decade*"

        ax.set_title(title)

        # Add panel label
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

        # Add stats box
        stats_text = (
            f"Years: {', '.join(map(str, cluster_stats[i]['years'][:5]))}"
            f"{' ...' if len(cluster_stats[i]['years']) > 5 else ''}\n"
            f"Freq: {cluster_stats[i]['count'] / sum(len(y) for y in cluster_years.values()) * 100:.1f}%"
        )

        ax.text(0.98, 0.02, stats_text,
                transform=ax.transAxes, fontsize=8, ha='right',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(cf, cax=cbar_ax, label='Z500 Anomaly (m)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(enhanced_z500_dir, 'enhanced_z500_anomalies.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Enhanced Z500 anomaly plots saved to: {enhanced_z500_dir}")
    return enhanced_z500_dir


pass


#######################################################################################################################
#######################################################################################################################

def create_enhanced_correlation_plot(correlation_map, p_value_map, lons, lats,
                                     cluster_index, time_series_data, cluster_name=None,
                                     output_file=None):
    """
    Create an enhanced Z500 correlation plot with a time series table below it

    Parameters:
    -----------
    correlation_map : ndarray
        2D array of correlation values between Z500 and cluster index
    p_value_map : ndarray
        2D array of p-values for the correlations
    lons, lats : ndarray
        1D arrays of longitude and latitude coordinates
    cluster_index : int
        The cluster number being visualized
    time_series_data : pandas.DataFrame
        DataFrame containing time series data with columns for years, frequency, and other metrics
    cluster_name : str, optional
        Physical name for this cluster (e.g., "NAO+ Pattern")
    output_file : str, optional
        If provided, save the figure to this file
    """
    # Create figure with GridSpec to control layout
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 0.3, 1])  # Map, time series, table

    # 1. Main correlation map
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

    # Get max correlation for color scaling
    vmax = min(0.8, np.nanmax(np.abs(correlation_map)))

    # Create mask for non-significant correlations
    masked_corr = np.ma.masked_where(p_value_map > 0.05, correlation_map)

    # Plot correlation
    cf = ax_map.contourf(lons, lats, masked_corr,
                         levels=np.linspace(-vmax, vmax, 17),
                         cmap='RdBu_r', transform=ccrs.PlateCarree(),
                         extend='both')

    # Add contour lines for all data
    cs = ax_map.contour(lons, lats, correlation_map,
                        levels=[-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8],
                        colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

    ax_map.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.1f')

    # Add stippling for significance with reduced density
    significance_mask = p_value_map < 0.05
    lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    significant_lons = lons_mesh[significance_mask]
    significant_lats = lats_mesh[significance_mask]
    stride = 5  # Adjust this to control stippling density
    ax_map.scatter(significant_lons[::stride], significant_lats[::stride],
                   color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

    # Add coastlines and land
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)
    ax_map.add_feature(land)
    ax_map.coastlines(resolution='50m')

    # Add gridlines
    gl = ax_map.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Add title with cluster name if provided
    title = f'Z500 Correlation with Cluster {cluster_index} Frequency'
    if cluster_name:
        title += f' ({cluster_name})'
    ax_map.set_title(title)

    # Add panel label
    ax_map.text(0.02, 0.98, f"(a)", transform=ax_map.transAxes,
                fontsize=14, fontweight='bold', va='top')

    # Find max correlation location
    y_max, x_max = np.unravel_index(np.nanargmax(np.abs(correlation_map)), correlation_map.shape)
    max_lat, max_lon = lats[y_max], lons[x_max]
    max_corr = correlation_map[y_max, x_max]

    # Add text box with key statistics
    stats_text = (
        f"Max correlation: {max_corr:.2f}\n"
        f"Location: {max_lat:.1f}°N, {max_lon:.1f}°E\n"
        f"Mean frequency: {time_series_data['Frequency'].mean():.1f}%"
    )
    ax_map.text(0.98, 0.02, stats_text, transform=ax_map.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                va='bottom', ha='right')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.68, 0.02, 0.25])  # Adjust position as needed
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Correlation Coefficient (p < 0.05)')

    # 2. Add time series plot
    ax_ts = fig.add_subplot(gs[1])
    years = time_series_data['Year']
    freq = time_series_data['Frequency']

    # Plot frequency time series
    ax_ts.plot(years, freq, 'b-', linewidth=1.5)
    ax_ts.fill_between(years, freq, alpha=0.3, color='blue')

    # Add NAO correlation if available
    if 'NAO_Corr' in time_series_data.columns:
        ax_nao = ax_ts.twinx()
        ax_nao.plot(years, time_series_data['NAO_Corr'], 'r--', linewidth=1.5)
        ax_nao.set_ylabel('NAO Correlation', color='red')
        ax_nao.tick_params(axis='y', colors='red')

    ax_ts.set_xlim(min(years), max(years))
    ax_ts.set_ylabel('Frequency (%)', color='blue')
    ax_ts.tick_params(axis='y', colors='blue')
    ax_ts.grid(True, linestyle='--', alpha=0.3)

    # Add panel label
    ax_ts.text(0.02, 0.9, f"(b)", transform=ax_ts.transAxes,
               fontsize=14, fontweight='bold')

    # 3. Add table with trend statistics
    ax_table = fig.add_subplot(gs[2])
    ax_table.axis('off')  # Hide axis

    # Create table data
    # Calculate trend in different periods
    periods = []
    freq_trends = []
    p_values = []

    # Group the data by decades
    decades = np.arange(np.floor(min(years) / 10) * 10, np.ceil(max(years) / 10) * 10 + 1, 10)
    for i in range(len(decades) - 1):
        period_start = int(decades[i])
        period_end = int(decades[i + 1])

        mask = (years >= period_start) & (years < period_end)
        if sum(mask) >= 5:  # Only calculate if at least 5 points
            period_years = years[mask]
            period_freq = freq[mask]

            slope, intercept, r_value, p_value, std_err = stats.linregress(period_years, period_freq)
            trend = slope * 10  # Trend per decade

            periods.append(f"{period_start}-{period_end - 1}")
            freq_trends.append(f"{trend:.2f}%/decade")
            p_values.append(f"{p_value:.3f}")

    # Add overall trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, freq)
    trend = slope * 10  # Trend per decade
    periods.append(f"Overall ({min(years)}-{max(years)})")
    freq_trends.append(f"{trend:.2f}%/decade")
    p_values.append(f"{p_value:.3f}")

    # Create DataFrame for table
    table_data = pd.DataFrame({
        'Period': periods,
        'Frequency Trend': freq_trends,
        'P-value': p_values,
        'Significant': ['*' if float(p.replace('0.', '')) < 0.05 else '' for p in p_values]
    })

    # Create the table
    table = ax_table.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Highlight significant trends
    for i, p_val in enumerate(p_values):
        if float(p_val.replace('0.', '')) < 0.05:
            cell = table[(i + 1, 2)]
            cell.set_facecolor('#d8f3dc')  # Light green background

    # Add panel label
    ax_table.text(0.02, 0.95, f"(c)", transform=ax_table.transAxes,
                  fontsize=14, fontweight='bold')

    # Add a caption for the table
    plt.figtext(0.5, 0.08, f"Table: Trend analysis for Cluster {cluster_index} frequency",
                ha="center", fontsize=10, fontstyle='italic')

    plt.tight_layout()
    plt.subplots_adjust(right=0.9, bottom=0.15, hspace=0.3)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    return fig


#######################################################################################################################
#######################################################################################################################

def create_combined_cluster_correlation_analysis(correlation_results, cluster_info, output_file=None):
    """
    Create a multi-panel figure showing correlation analyses for all clusters

    Parameters:
    -----------
    correlation_results : dict
        Dictionary of correlation results for each cluster
    cluster_info : dict
        Dictionary containing metadata about each cluster
    output_file : str, optional
        If provided, save the figure to this file
    """
    # Filter correlation results to only include frequency indices (not dipole indices)
    freq_results = {k: v for k, v in correlation_results.items() if '_freq_std' in k}

    n_clusters = len(freq_results)

    # Limit to maximum of 4 clusters for 2×2 grid
    if n_clusters > 4:
        print(f"Warning: {n_clusters} clusters found, limiting to first 4 for 2×2 grid")
        freq_results = dict(list(freq_results.items())[:4])
        n_clusters = 4

    if n_clusters == 0:
        print("No frequency correlation results found")
        return None

    # Create figure with subplot grid
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2)

    # Get global min/max for consistent color scaling
    all_corrs = []
    for cluster in freq_results:
        corr_map = freq_results[cluster]['corr_map']
        all_corrs.extend(corr_map.flatten())

    vmax = min(0.8, np.nanmax(np.abs(all_corrs)))

    # Create a land feature for all maps
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)

    # Plot each cluster's correlation map (limit to 4)
    for i, cluster_key in enumerate(sorted(list(freq_results.keys())[:4])):
        row, col = divmod(i, 2)
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())

        corr_map = freq_results[cluster_key]['corr_map']
        p_map = freq_results[cluster_key]['p_map']
        lons = freq_results[cluster_key]['lons']
        lats = freq_results[cluster_key]['lats']

        masked_corr = np.ma.masked_where(p_map > 0.05, corr_map)

        # Plot correlation
        cf = ax.contourf(lons, lats, masked_corr,
                         levels=np.linspace(-vmax, vmax, 17),
                         cmap='RdBu_r', transform=ccrs.PlateCarree(),
                         extend='both')

        # Add contour lines
        cs = ax.contour(lons, lats, corr_map,
                        levels=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
                        colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.1f')

        # Add stippling for significance with reduced density
        significance_mask = p_map < 0.05
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
        significant_lons = lons_mesh[significance_mask]
        significant_lats = lats_mesh[significance_mask]

        # Use a larger stride to reduce stippling density
        stride = 5
        ax.scatter(significant_lons[::stride], significant_lats[::stride],
                   color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

        # Add coastlines and land
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Extract cluster number from key name
        cluster_num = cluster_key.split('_')[1]

        # Get cluster metadata
        cluster_name = cluster_info.get(int(cluster_num), {}).get('name', f"Cluster {cluster_num}")
        physical_pattern = cluster_info.get(int(cluster_num), {}).get('physical_pattern', '')

        # Add title with physical interpretation
        title = f"Cluster {cluster_num} Frequency"
        if physical_pattern:
            title += f"\n({physical_pattern})"
        ax.set_title(title)

        # Add panel label
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

        # Find max correlation location
        max_idx = np.unravel_index(np.nanargmax(np.abs(corr_map)), corr_map.shape)
        max_lat, max_lon = lats[max_idx[0]], lons[max_idx[1]]
        max_corr = corr_map[max_idx]

        # Add text box with key statistics
        stats_text = f"Max corr: {max_corr:.2f}\nLat: {max_lat:.1f}°, Lon: {max_lon:.1f}°"
        ax.text(0.02, 0.08, stats_text, transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Correlation Coefficient (p < 0.05)')

    # Add a common title
    fig.suptitle('Cluster Frequency Correlations with Z500', fontsize=16, y=0.98)

    # Add annotation about physical interpretation
    annotation = (
        "Panel (a): Cluster 1 pattern with correlation patterns\n"
        "Panel (b): Cluster 2 pattern with correlation patterns\n"
        "Panel (c): Cluster 3 pattern with correlation patterns\n"
        "Panel (d): Cluster 4 pattern with correlation patterns"
    )

    plt.figtext(0.5, 0.01, annotation, ha='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    return fig


#######################################################################################################################
#######################################################################################################################

def create_z500_correlation_timeseries_plot(cluster_index, time_series_data, z500_file, output_dir):
    """
    Creates a more comprehensive analysis of Z500 correlations with time series information

    Parameters:
    -----------
    cluster_index : int
        The cluster number to analyze
    time_series_data : pandas DataFrame
        Data containing years and cluster frequencies over time
    z500_file : str
        Path to the Z500 data file
    output_dir : str
        Directory to save output files

    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # This is a template for the function you would implement in your code
    # It would load Z500 data, calculate correlations for different time periods
    # and create visualizations with tables

    # Example implementation outline:

    # 1. Load Z500 data
    # ds_z500 = xr.open_dataset(z500_file)

    # 2. Set up time periods for sliding window analysis
    # window_size = 20  # years
    # step = 5  # years
    # start_years = range(1950, 2000, step)

    # 3. Calculate correlation maps for each time window
    # time_window_correlations = {}
    # for start_year in start_years:
    #     end_year = start_year + window_size
    #     # Select data for this window
    #     # Calculate correlation map
    #     # Store results

    # 4. Create visualization showing how correlation patterns evolved
    # fig = create_correlation_evolution_plot(time_window_correlations)

    # 5. Create table with correlation statistics for each period
    # table_data = create_correlation_statistics_table(time_window_correlations)

    # 6. Save results
    # output_file = os.path.join(output_dir, f'cluster_{cluster_index}_z500_correlation_evolution.png')
    # fig.savefig(output_file, dpi=300, bbox_inches='tight')

    print(f"Created comprehensive Z500 correlation analysis for cluster {cluster_index}")
    return {"status": "success"}


#######################################################################################################################
#######################################################################################################################

def create_som_pattern_2d_projection(som, clustering_results, output_dir):
    """
    Create a standalone static 2D projection plot of SOM patterns colored by cluster

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    clustering_results : dict
        Results from cluster_som_patterns
    output_dir : str
        Directory to save the output file
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get pattern to cluster mapping and distance matrix
    pattern_clusters = clustering_results["pattern_clusters"]
    distance_matrix = clustering_results["distance_matrix"]

    # Get SOM dimensions
    weights = som.get_weights()
    n_rows, n_cols = weights.shape[:2]

    # Get unique clusters
    unique_clusters = sorted(set(pattern_clusters.values()))

    # Create 2D projection using MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pattern_positions = mds.fit_transform(distance_matrix)

    # Define specific colors for clusters 1-4 (not 0-3)
    specific_colors = {
        1: 'blue',  # Cluster 1
        2: 'orange',  # Cluster 2
        3: 'green',  # Cluster 3
        4: 'red'  # Cluster 4
    }

    # Create figure
    plt.figure(figsize=(12, 10))

    # Plot each pattern
    for pattern_idx in range(1, n_rows * n_cols + 1):
        if pattern_idx in pattern_clusters:
            cluster = pattern_clusters[pattern_idx]
            color = specific_colors.get(cluster, 'gray')

            plt.scatter(
                pattern_positions[pattern_idx - 1, 0],
                pattern_positions[pattern_idx - 1, 1],
                c=color,
                s=100,
                edgecolors='black',
                zorder=10
            )

            # Add pattern number label
            plt.text(
                pattern_positions[pattern_idx - 1, 0],
                pattern_positions[pattern_idx - 1, 1],
                str(pattern_idx),
                fontsize=10,
                ha='center',
                va='center',
                fontweight='bold',
                color='black',
                zorder=20
            )

    # Add legend with 1-indexed clusters (not 0-indexed)
    for cluster in unique_clusters:
        plt.scatter([], [], c=specific_colors.get(cluster, 'gray'),
                    label=f'Cluster {cluster}', s=100, edgecolors='black')

    plt.legend(title="Clusters", loc='upper right', fontsize=12)
    plt.title('2D Projection of SOM Patterns', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure as PNG
    output_file = os.path.join(output_dir, 'pattern_2d_projection.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"2D projection plot saved to: {output_file}")

    return output_file


#######################################################################################################################
#######################################################################################################################

def plot_cluster3_actual_z500(cluster_occurrences, z500_data, output_dir):
    """
    Plot the actual Z500 fields for the periods when Cluster 3 was present

    Parameters:
    -----------
    cluster_occurrences : dict
        Results from analyze_cluster_occurrences containing time series info
    z500_data : xarray.Dataset
        Z500 data covering the time period including 2022-2024, uses valid_time dimension
    output_dir : str
        Directory to save output plots
    """
    # Extract dates when Cluster 3 occurred
    cluster_id = 3
    cluster_times = []

    for i, cluster in enumerate(cluster_occurrences['cluster_sequence']):
        if cluster == cluster_id:
            cluster_times.append(cluster_occurrences['times'][i])

    print(f"Found {len(cluster_times)} occurrences of Cluster 3")
    print(f"Dates: {cluster_times}")

    # Create directory for output
    cluster3_dir = os.path.join(output_dir, 'cluster3_z500_analysis')
    os.makedirs(cluster3_dir, exist_ok=True)

    # Check Z500 data dimensions
    print(f"Z500 data dimensions: {z500_data.z.dims}")

    # Select 500 hPa level if needed
    if 'pressure_level' in z500_data.dims:
        print("Selecting 500 hPa level from Z500 data")
        # Find the index closest to 500 hPa
        if hasattr(z500_data.pressure_level, 'values'):
            pressure_levels = z500_data.pressure_level.values
            idx_500 = np.abs(pressure_levels - 500).argmin()
            z500_level = pressure_levels[idx_500]
            print(f"Using pressure level: {z500_level} hPa")
            z500_data = z500_data.sel(pressure_level=z500_level)
        else:
            print("Warning: pressure_level dimension exists but values not accessible")

    # Find nearest valid_time points in the Z500 dataset
    z500_subset = []
    for time_point in cluster_times:
        # Find nearest valid_time to the cluster time
        time_diff = abs(pd.to_datetime(z500_data.valid_time.values) - time_point)
        nearest_idx = time_diff.argmin()
        nearest_time = z500_data.valid_time.values[nearest_idx]

        # Select Z500 data for this time point
        z500_time_slice = z500_data.sel(valid_time=nearest_time)
        z500_subset.append(z500_time_slice)

        print(f"Selected time {nearest_time} for cluster time {time_point}")
        print(f"Z500 shape for this time: {z500_time_slice.z.shape}")

    # Combine the time slices
    if z500_subset:
        # Create a new dataset with these time points
        z500_cluster3 = xr.concat(z500_subset, dim='valid_time')

        # Calculate climatology (mean Z500 from entire dataset)
        z500_climatology = z500_data.z.mean('valid_time')

        # Calculate anomalies
        z500_anomalies = z500_cluster3.z - z500_climatology

        # Calculate composite (average of all occurrences)
        z500_composite = z500_cluster3.z.mean('valid_time')
        z500_anomaly_composite = z500_anomalies.mean('valid_time')

        # Create land feature
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='black',
                                            facecolor='lightgray')

        # 1. Plot the absolute Z500 composite
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        # Plot Z500 composite
        levels = np.arange(49000, 58000, 1000)
        cf = ax.contourf(z500_composite.longitude, z500_composite.latitude, z500_composite,
                         levels=levels, cmap='viridis', transform=ccrs.PlateCarree())

        # Add contour lines
        cs = ax.contour(z500_composite.longitude, z500_composite.latitude, z500_composite,
                        levels=levels, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.0f')

        # Add coastlines and land
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Add title and colorbar
        ax.set_title(f'Z500 Composite for Cluster 3 (n={len(cluster_times)} occurrences)')
        plt.colorbar(cf, ax=ax, label='Z500 (m)')

        # Save figure
        plt.savefig(os.path.join(cluster3_dir, 'cluster3_z500_composite.png'),
                    dpi=300, bbox_inches='tight')

        # 2. Plot the Z500 anomaly composite
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        # Find appropriate levels for anomaly
        anom_max = max(abs(float(z500_anomaly_composite.min())), abs(float(z500_anomaly_composite.max())))
        anom_levels = np.linspace(-anom_max, anom_max, 21)

        # Plot Z500 anomaly
        cf = ax.contourf(z500_anomaly_composite.longitude, z500_anomaly_composite.latitude,
                         z500_anomaly_composite, levels=anom_levels, cmap='RdBu_r',
                         transform=ccrs.PlateCarree(), extend='both')

        # Add contour lines
        cs = ax.contour(z500_anomaly_composite.longitude, z500_anomaly_composite.latitude,
                        z500_anomaly_composite, levels=anom_levels[::2], colors='black',
                        linewidths=0.5, transform=ccrs.PlateCarree())
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.0f')

        # Add coastlines and land
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Add title and colorbar
        ax.set_title(f'Z500 Anomaly for Cluster 3 (n={len(cluster_times)} occurrences)')
        plt.colorbar(cf, ax=ax, label='Z500 Anomaly (m)')

        # Save figure
        plt.savefig(os.path.join(cluster3_dir, 'cluster3_z500_anomaly.png'),
                    dpi=300, bbox_inches='tight')

        # 3. Plot individual maps for each occurrence
        for i, (time, z500_slice) in enumerate(zip(cluster_times, z500_subset)):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

            # Get anomaly for this date
            z500_anomaly = z500_slice.z - z500_climatology

            # Plot anomaly
            cf = ax.contourf(z500_anomaly.longitude, z500_anomaly.latitude,
                             z500_anomaly, levels=anom_levels, cmap='RdBu_r',
                             transform=ccrs.PlateCarree(), extend='both')

            # Add contour lines
            cs = ax.contour(z500_anomaly.longitude, z500_anomaly.latitude,
                            z500_anomaly, levels=anom_levels[::2], colors='black',
                            linewidths=0.5, transform=ccrs.PlateCarree())

            # Add coastlines and land
            ax.add_feature(land)
            ax.coastlines(resolution='50m')

            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            # Add title and colorbar
            time_str = pd.to_datetime(time).strftime("%Y-%m-%d")
            ax.set_title(f'Z500 Anomaly for Cluster 3 - {time_str}')
            plt.colorbar(cf, ax=ax, label='Z500 Anomaly (m)')

            # Save figure
            plt.savefig(os.path.join(cluster3_dir, f'cluster3_z500_{pd.to_datetime(time).strftime("%Y%m%d")}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

        return {
            'composite': z500_composite,
            'anomaly': z500_anomaly_composite,
            'dates': cluster_times
        }
    else:
        print("No Z500 data found for Cluster 3 occurrences")
        return None


#######################################################################################################################
#######################################################################################################################

def create_z500_anomaly_with_ttest(cluster_years, z500_yearly, z500_climatology, clustering_results, output_dir):
    """
    Create a 2×2 panel figure showing Z500 anomalies for each cluster with t-test significance stippling

    Parameters:
    -----------
    cluster_years : dict
        Dictionary mapping cluster numbers to years when each cluster was dominant
    z500_yearly : xarray.Dataset
        Yearly Z500 data with a 'year' coordinate
    z500_climatology : xarray.DataArray
        Mean Z500 field (climatology)
    clustering_results : dict
        Results from cluster_som_patterns function
    output_dir : str
        Directory to save output

    Returns:
    --------
    str
        Path to the saved figure
    """

    # Create output directory
    ttest_dir = os.path.join(output_dir, 'z500_anomalies_ttest')
    os.makedirs(ttest_dir, exist_ok=True)

    # Set up the figure for 2×2 panel
    fig = plt.figure(figsize=(16, 14))

    # Create land feature
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)

    # Get number of clusters
    pattern_clusters = clustering_results['pattern_clusters']
    unique_clusters = sorted(set(pattern_clusters.values()))

    # Calculate anomalies and t-test for each cluster
    all_anomalies = []
    p_values_all = []

    for cluster in unique_clusters:
        # Skip if not enough years
        if len(cluster_years.get(cluster, [])) < 3:
            print(
                f"Warning: Cluster {cluster} has < 3 occurrences ({len(cluster_years.get(cluster, []))}). T-test may not be reliable.")

        # Get years for this cluster
        years = cluster_years.get(cluster, [])

        if len(years) > 0:
            # Calculate composite Z500 for this cluster
            z500_composite = z500_yearly.sel(year=years).z.mean('year')

            # Calculate anomaly from climatology
            z500_anomaly = z500_composite - z500_climatology

            # Collect individual year anomalies for t-test
            all_year_anomalies = []
            for year in years:
                year_anomaly = z500_yearly.sel(year=year).z - z500_climatology
                all_year_anomalies.append(year_anomaly)

            # Perform t-test if we have enough years
            if len(years) >= 3:  # Minimum needed for t-test
                # Stack individual anomalies
                stacked_anomalies = np.stack([anom.values for anom in all_year_anomalies])

                # Calculate t-statistic and p-value
                t_stat = np.mean(stacked_anomalies, axis=0) / (
                        np.std(stacked_anomalies, axis=0, ddof=1) / np.sqrt(len(years)))
                # Two-tailed t-test
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(years) - 1))
            else:
                # Not enough samples for proper t-test
                p_values = np.ones_like(z500_anomaly.values)

            all_anomalies.append(z500_anomaly)
            p_values_all.append(p_values)
        else:
            # Create empty placeholders if no data
            all_anomalies.append(None)
            p_values_all.append(None)

    # Find global min/max for consistent color scale
    valid_anomalies = [anom for anom in all_anomalies if anom is not None]
    if valid_anomalies:
        anom_min = min(anom.min().values for anom in valid_anomalies)
        anom_max = max(anom.max().values for anom in valid_anomalies)
        abs_max = max(abs(anom_min), abs(anom_max))
    else:
        abs_max = 100  # Default fallback

    # Round to nearest 100
    abs_max = int(np.ceil(abs_max / 100) * 100)

    # Plot each cluster in 2×2 grid
    for i, (anomaly, p_values) in enumerate(zip(all_anomalies, p_values_all)):
        if anomaly is None:
            continue

        # Create subplot in 2×2 grid
        #row, col = divmod(i, 2)
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

        # Create contour levels
        levels = np.linspace(-abs_max, abs_max, 21)

        # Plot anomaly
        cf = ax.contourf(anomaly.longitude, anomaly.latitude, anomaly,
                         levels=levels, cmap='RdBu_r',
                         transform=ccrs.PlateCarree(),
                         extend='both')

        # Add contour lines
        cs = ax.contour(anomaly.longitude, anomaly.latitude, anomaly,
                        colors='k', linewidths=0.5,
                        transform=ccrs.PlateCarree())

        # Add contour labels (only on some contours to avoid cluttering)
        ax.clabel(cs, cs.levels[::4], inline=True, fontsize=8, fmt='%d')

        # Add stippling for significance
        if p_values is not None:
            significance_mask = p_values < 0.05
            lons_mesh, lats_mesh = np.meshgrid(anomaly.longitude, anomaly.latitude)
            significant_lons = lons_mesh[significance_mask]
            significant_lats = lats_mesh[significance_mask]

            # Add stippling with reasonable density
            # Adjust stride value to control stippling density
            stride = max(1, int(np.sqrt(len(significant_lons)) / 15))

            ax.scatter(significant_lons[::stride], significant_lats[::stride],
                       color='black', s=1.5, alpha=0.7, transform=ccrs.PlateCarree())

        # Add coastlines and land
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Get number of occurrences for title
        cluster_num = unique_clusters[i] if i < len(unique_clusters) else i + 1
        n_occurrences = len(cluster_years.get(cluster_num, []))

        # Add title with physical interpretation
        pattern_names = {
            1: "Gulf Stream Displacement",
            2: "Negative AMO-like Pattern",
            3: "Tripole Pattern",
            4: "Subtropical Warming"
        }
        pattern_name = pattern_names.get(i, f"Cluster {i}")

        ax.set_title(f'Z500 Anomaly for Cluster {i} (n={n_occurrences} occurrences)\n{pattern_name}', fontsize=12)

        # Add panel label
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Z500 Anomaly (m)')

    # Add note about significance testing
    plt.figtext(0.5, 0.01,
                "Note: Stippling indicates anomalies significant at p < 0.05 level (two-tailed t-test)",
                ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.02, 0.9, 0.98])

    # Save figure
    fig_path = os.path.join(ttest_dir, 'z500_anomalies_with_ttest.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Z500 anomaly figure with t-test saved to: {fig_path}")
    return fig_path


#######################################################################################################################
#######################################################################################################################

def run_z500_ttest_analysis(som, X, ds, clustering_results, z500_file, output_dir):
    """
    Wrapper function to load Z500 data and run the t-test analysis
    with proper handling of 1-indexed clusters
    """
    print("\nGenerating Z500 anomaly plots with t-test significance...")

    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers (1-based)
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]

    # Map patterns to clusters
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    years = times.year

    # Get the actual unique clusters that exist in the data
    unique_clusters = sorted(set(pattern_clusters.values()))
    print(f"Working with {len(unique_clusters)} unique clusters: {unique_clusters}")

    # Create a mapping of year to dominant cluster
    year_cluster = {}
    for year in np.unique(years):
        # Find dominant cluster for this year
        year_mask = (years == year)
        if np.any(year_mask):
            year_clusters_data = cluster_sequence[year_mask]  # Fixed variable name here

            # Create a proper bincount for the clusters
            max_cluster = max(unique_clusters) if unique_clusters else 1
            cluster_counts = np.zeros(max_cluster + 1)
            for cluster in year_clusters_data:
                if cluster <= max_cluster:
                    cluster_counts[cluster] += 1

            # Find dominant cluster (skip index 0 since clusters start from 1)
            if len(cluster_counts) > 1:
                dominant_cluster = np.argmax(cluster_counts[1:]) + 1
            else:
                dominant_cluster = unique_clusters[0] if unique_clusters else 1

            year_cluster[year] = dominant_cluster

    # Create cluster_years dictionary for the function
    cluster_years = {}  # This is what was missing!
    for cluster in unique_clusters:
        cluster_years[cluster] = [year for year, clus in year_cluster.items() if clus == cluster]
        print(f"Cluster {cluster}: {len(cluster_years[cluster])} years")

    # Load Z500 data
    print("Loading Z500 data...")
    ds_z500 = xr.open_dataset(z500_file)

    # Select 500 hPa level if necessary
    if 'pressure_level' in ds_z500.dims:
        ds_z500 = ds_z500.sel(pressure_level=500.0)

    # Select DJF months
    z500_times = pd.to_datetime(ds_z500.valid_time.values)
    is_djf = (z500_times.month == 12) | (z500_times.month == 1) | (z500_times.month == 2)
    ds_z500_djf = ds_z500.isel(valid_time=is_djf)

    # Create year coordinate for seasonal averages
    year_coords = [t.year if t.month > 2 else t.year - 1
                   for t in pd.to_datetime(ds_z500_djf.valid_time.values)]
    ds_z500_djf = ds_z500_djf.assign_coords(year=('valid_time', year_coords))

    # Calculate seasonal means
    z500_yearly = ds_z500_djf.groupby('year').mean('valid_time')

    # Calculate climatology
    z500_climatology = z500_yearly.z.mean('year')

    # Call the t-test function with the unique clusters that actually exist
    fig_path = create_z500_anomaly_with_ttest(
        cluster_years, z500_yearly, z500_climatology, clustering_results, output_dir
    )

    print(f"Z500 anomaly analysis with t-test completed. Figure saved to: {fig_path}")
    return fig_path



#######################################################################################################################
#######################################################################################################################

def create_simple_cluster_indices(som, X, ds, clustering_results, output_dir):
    print("\nCreating cluster indices...")

    # Create output directory
    indices_dir = os.path.join(output_dir, 'cluster_indices')
    os.makedirs(indices_dir, exist_ok=True)

    # Get cluster sequence
    bmu_indices = np.array([som.winner(x) for x in X])
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    pattern_clusters = clustering_results["pattern_clusters"]

    # Verify clusters are 1-indexed
    unique_clusters = sorted(set(pattern_clusters.values()))
    min_cluster = min(unique_clusters) if unique_clusters else 1

    if min_cluster == 0:
        print("ERROR: Pattern clusters are 0-indexed! Fixing...")
        pattern_clusters = {k: v + 1 for k, v in pattern_clusters.items()}
        unique_clusters = sorted(set(pattern_clusters.values()))
        print(f"Fixed: Clusters are now 1-indexed: {unique_clusters}")
    else:
        print(f"Confirmed: Clusters are properly 1-indexed: {unique_clusters}")

    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time information
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Create DataFrame
    df = pd.DataFrame({
        'time': times,
        'year': years,
        'month': months,
        'cluster': cluster_sequence
    })

    # Filter to DJF and create season years
    is_djf = (df['month'] == 12) | (df['month'] == 1) | (df['month'] == 2)
    df_djf = df[is_djf].copy()
    df_djf['season_year'] = df_djf.apply(
        lambda row: row['year'] + 1 if row['month'] == 12 else row['year'], axis=1
    )

    # Calculate annual cluster frequencies
    annual_freq = df_djf.groupby(['season_year', 'cluster']).size().unstack(fill_value=0)
    annual_freq_norm = annual_freq.div(annual_freq.sum(axis=1), axis=0)

    # Create standardized indices
    cluster_indices = {}

    for cluster in unique_clusters:
        if cluster in annual_freq_norm.columns:
            freq_series = annual_freq_norm[cluster]
            # Standardize to mean=0, std=1
            if freq_series.std() > 0:
                std_series = (freq_series - freq_series.mean()) / freq_series.std()
            else:
                std_series = freq_series - freq_series.mean()

            cluster_indices[f'cluster_{cluster}_index'] = std_series

    # Save to CSV
    indices_df = pd.DataFrame(cluster_indices)
    indices_df.to_csv(os.path.join(indices_dir, 'cluster_indices.csv'))

    # Save annual frequencies
    annual_freq_norm.to_csv(os.path.join(indices_dir, 'cluster_frequencies.csv'))

    print(f"Cluster indices saved to: {indices_dir}")
    print(f"Created indices for clusters: {list(cluster_indices.keys())}")

    return indices_df

#######################################################################################################################
#######################################################################################################################

def apply_indices_to_europe_multivar(som, X, ds, clustering_results, data_file, output_dir,
                                     variables=['z', 'q', 'crwc', 'ciwc']):
    """
    Apply cluster indices derived from North Atlantic SST patterns to European meteorological data

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array
    ds : xarray.Dataset
        Original SST dataset with coordinates
    clustering_results : dict
        Results from cluster_som_patterns function
    data_file : str
        Path to the data file for Europe/Mediterranean region
    output_dir : str
        Directory to save analysis results
    variables : list
        List of variables to analyze (default: ['z', 'q', 'crwc', 'ciwc'])

    Returns:
    --------
    dict
        Dictionary containing correlation maps and analysis results for all variables
    """
    print(f"\nApplying cluster indices to European/Mediterranean data for variables: {', '.join(variables)}...")

    # Create output directory
    europe_dir = os.path.join(output_dir, 'europe_multivar_analysis')
    os.makedirs(europe_dir, exist_ok=True)

    # Create subdirectories for each variable
    for var in variables:
        os.makedirs(os.path.join(europe_dir, var), exist_ok=True)

    # Load data for European domain
    print(f"Loading data from: {data_file}")
    ds_euro = xr.open_dataset(data_file)

    # Select 500 hPa level if needed and not already selected
    if 'pressure_level' in ds_euro.dims and len(ds_euro.pressure_level) > 1:
        ds_euro = ds_euro.sel(pressure_level=500.0)
        print("Selected 500 hPa pressure level")

    # Check available variables
    available_vars = [var for var in variables if var in ds_euro]
    if len(available_vars) < len(variables):
        missing = set(variables) - set(available_vars)
        print(f"Warning: Variables not found in dataset: {', '.join(missing)}")

    print(f"Processing variables: {', '.join(available_vars)}")

    # Calculate seasonal climatology and anomalies for each variable
    print("Calculating anomalies for all variables...")
    anomalies = {}

    for var in available_vars:
        try:
            # Calculate monthly climatology and anomalies
            monthly_clim = ds_euro[var].groupby('valid_time.month').mean('valid_time')
            anomalies[var] = ds_euro[var].groupby('valid_time.month') - monthly_clim
            print(f"  Calculated anomalies for {var}")
        except Exception as e:
            print(f"  Error calculating anomalies for {var}: {str(e)}")

    # Get BMUs for all data points from the original dataset
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers (1-based)
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Map patterns to clusters
    pattern_clusters = clustering_results['pattern_clusters']
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates from original dataset
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Get unique clusters that actually exist
    unique_clusters = sorted(set(pattern_clusters.values()))
    n_clusters = len(unique_clusters)  # Define n_clusters here

    # Create DataFrame with original data time series
    df_orig = pd.DataFrame({
        'time': times,
        'year': years,
        'month': months,
        'cluster': cluster_sequence
    })

    # Filter to DJF months
    is_djf = (df_orig['month'] == 12) | (df_orig['month'] == 1) | (df_orig['month'] == 2)
    df_djf = df_orig[is_djf].copy()

    # Create "season year" column (December belongs to next year's DJF)
    def get_season_year(row):
        if row['month'] == 12:
            return row['year'] + 1
        return row['year']

    df_djf['season_year'] = df_djf.apply(get_season_year, axis=1)

    # Calculate cluster frequency by season
    print("Calculating seasonal cluster frequencies...")
    seasonal_counts = df_djf.groupby(['season_year', 'cluster']).size().unstack(fill_value=0)
    seasonal_freq = seasonal_counts.div(seasonal_counts.sum(axis=1), axis=0)

    # Save the cluster indices
    indices_df = pd.DataFrame(seasonal_freq)
    indices_df.to_csv(os.path.join(europe_dir, 'cluster_indices.csv'))
    print(f"Cluster indices saved to: {os.path.join(europe_dir, 'cluster_indices.csv')}")

    # Create indices dictionary
    indices = {}

    # Simple indices - frequency of each cluster that actually exists
    for cluster in unique_clusters:
        if cluster in seasonal_freq.columns:
            indices[f'cluster_{cluster}_freq'] = seasonal_freq[cluster]
            # Create standardized version
            if seasonal_freq[cluster].std() > 0:
                std_index = (seasonal_freq[cluster] - seasonal_freq[cluster].mean()) / seasonal_freq[cluster].std()
            else:
                std_index = seasonal_freq[cluster] - seasonal_freq[cluster].mean()
            indices[f'cluster_{cluster}_freq_std'] = std_index

    # Get Euro data time info and find common times
    euro_times = pd.to_datetime(ds_euro.valid_time.values)
    euro_years = pd.DatetimeIndex(euro_times).year
    euro_months = pd.DatetimeIndex(euro_times).month

    # Filter to DJF months in Euro dataset
    is_euro_djf = (euro_months == 12) | (euro_months == 1) | (euro_months == 2)
    euro_djf_times = euro_times[is_euro_djf]

    # Create season year for Euro data
    euro_season_years = []
    for t in euro_djf_times:
        if t.month == 12:
            euro_season_years.append(t.year + 1)
        else:
            euro_season_years.append(t.year)

    # Common season years between indices and Euro data
    common_years = sorted(list(set(indices_df.index).intersection(set(euro_season_years))))

    if len(common_years) < 3:
        print(f"Warning: Only {len(common_years)} common years between indices and Euro data.")
        print("Need at least 3 years for meaningful analysis.")
        return None

    print(f"Found {len(common_years)} common years for analysis: {common_years[:5]}...")

    # For each variable, analyze correlations with cluster indices
    all_results = {}

    for var in anomalies:
        var_dir = os.path.join(europe_dir, var)
        print(f"\nProcessing correlations for {var}...")

        # Store correlation results for this variable
        var_corr_results = {}

        # For each cluster index, calculate correlation maps
        for cluster in unique_clusters:
            cluster_key = f'cluster_{cluster}_freq_std'

            if cluster_key not in indices:
                print(f"  Warning: {cluster_key} not found in indices")
                continue

            print(f"  Calculating correlations for {cluster_key}...")

            # For each year, get the index value and the corresponding anomaly field
            index_values = []
            anomaly_fields = []

            for year in common_years:
                # Get index value for this year
                if year in indices_df.index:
                    index_val = indices[cluster_key].loc[year]

                    # Get corresponding anomaly fields
                    year_mask = np.array(euro_season_years) == year
                    if np.any(year_mask):
                        year_times = euro_djf_times[year_mask]
                        year_anomalies = anomalies[var].sel(valid_time=year_times).mean('valid_time')

                        index_values.append(index_val)
                        anomaly_fields.append(year_anomalies)

            if not index_values:
                print(f"  No data found for {cluster_key}")
                continue

            # Convert to arrays
            index_array = np.array(index_values)
            anomaly_stack = xr.concat(anomaly_fields, dim='year')

            # Calculate correlation at each grid point
            correlation_map = np.zeros((len(anomaly_stack.latitude), len(anomaly_stack.longitude)))
            p_value_map = np.ones_like(correlation_map)

            for i in range(len(anomaly_stack.latitude)):
                for j in range(len(anomaly_stack.longitude)):
                    if hasattr(anomaly_stack, 'pressure_level') and len(anomaly_stack.pressure_level) > 0:
                        data_series = anomaly_stack.isel(pressure_level=0)[:, i, j].values
                    else:
                        data_series = anomaly_stack[:, i, j].values

                    # Check for constant input and NaN values
                    if not np.isnan(data_series).any() and np.std(data_series) > 1e-10 and np.std(index_array) > 1e-10:
                        try:
                            r, p = stats.pearsonr(index_array, data_series)
                            correlation_map[i, j] = r
                            p_value_map[i, j] = p
                        except:
                            # Handle any correlation calculation errors
                            correlation_map[i, j] = np.nan
                            p_value_map[i, j] = np.nan
                    else:
                        correlation_map[i, j] = np.nan
                        p_value_map[i, j] = np.nan

            # Store results
            var_corr_results[cluster_key] = {
                'corr_map': correlation_map,
                'p_map': p_value_map,
                'lats': anomaly_stack.latitude.values,
                'lons': anomaly_stack.longitude.values
            }

            # Create visualization
            try:
                create_correlation_map(
                    correlation_map=correlation_map,
                    p_value_map=p_value_map,
                    lats=anomaly_stack.latitude.values,
                    lons=anomaly_stack.longitude.values,
                    cluster_key=cluster_key,
                    var_name=var,
                    output_file=os.path.join(var_dir, f'corr_{var}_{cluster_key}.png')
                )
            except Exception as e:
                print(f"  Error creating correlation map: {str(e)}")

        # Create combined visualization for this variable
        try:
            create_combined_correlation_map(
                corr_results=var_corr_results,
                var_name=var,
                output_file=os.path.join(var_dir, f'combined_{var}_correlations.png')
            )
        except Exception as e:
            print(f"  Error creating combined correlation map: {str(e)}")

        # Create composite analysis
        try:
            cluster_keys = [f'cluster_{i}_freq_std' for i in unique_clusters if f'cluster_{i}_freq_std' in indices]
            composites = create_variable_composites(
                indices=indices,
                anomalies=anomalies[var],
                euro_times=euro_times,
                euro_season_years=euro_season_years,
                common_years=common_years,
                var_name=var,
                cluster_keys=cluster_keys,
                output_dir=var_dir
            )
        except Exception as e:
            print(f"  Error creating composites: {str(e)}")
            composites = {}

        # Store all results for this variable
        all_results[var] = {
            'correlation_results': var_corr_results,
            'composites': composites
        }

    # Save all results
    try:
        with open(os.path.join(europe_dir, 'europe_multivar_results.pkl'), 'wb') as f:
            pickle.dump(all_results, f)
    except Exception as e:
        print(f"Error saving results: {str(e)}")

    print(f"Analysis complete. Results saved to: {europe_dir}")
    return all_results

def create_correlation_map(correlation_map, p_value_map, lats, lons, cluster_key, var_name, output_file=None):
    """Create a correlation map for a single cluster index and variable"""
    # Define variable names for plotting
    var_names = {
        'z': 'Geopotential Height',
        'q': 'Specific Humidity',
        'crwc': 'Rain Water Content',
        'ciwc': 'Cloud Ice Water Content'
    }

    full_var_name = var_names.get(var_name, var_name.upper())

    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Determine color scale
    vmax = min(0.8, np.nanmax(np.abs(correlation_map)))

    # Mask non-significant correlations
    masked_corr = np.ma.masked_where(p_value_map > 0.05, correlation_map)

    # Plot correlation
    cf = ax.contourf(lons, lats, masked_corr,
                     levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', transform=ccrs.PlateCarree(),
                     extend='both')

    # Add contour lines
    cs = ax.contour(lons, lats, correlation_map,
                    levels=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
                    colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

    ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.1f')

    # Add stippling for significance
    significance_mask = p_value_map < 0.05
    lons_mesh, lats_mesh = np.meshgrid(lons, lats)
    significant_lons = lons_mesh[significance_mask]
    significant_lats = lats_mesh[significance_mask]
    stride = 5  # Adjust for stippling density
    ax.scatter(significant_lons[::stride], significant_lats[::stride],
               color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

    # Add coastlines and borders
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)
    ax.add_feature(land)
    ax.coastlines(resolution='50m')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Get cluster number from key name
    cluster_num = cluster_key.split('_')[1]

    # Add title
    ax.set_title(f'Correlation between {full_var_name} and Cluster {cluster_num} Index')

    # Add colorbar
    plt.colorbar(cf, label='Correlation Coefficient (p < 0.05)')

    # Save figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig


def create_combined_correlation_map(corr_results, var_name, output_file=None):
    """Create a combined 2×2 panel figure showing correlations for all clusters"""
    # Define variable names for plotting
    var_names = {
        'z': 'Geopotential Height',
        'q': 'Specific Humidity',
        'crwc': 'Rain Water Content',
        'ciwc': 'Cloud Ice Water Content'
    }

    full_var_name = var_names.get(var_name, var_name.upper())

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Find global min/max for consistent color scaling
    all_corrs = []
    for results in corr_results.values():
        all_corrs.extend(results['corr_map'].flatten())

    vmax = min(0.8, np.nanmax(np.abs(all_corrs)))

    # Create land feature
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)

    # Physical interpretations for each cluster
    interpretations = {
        '0': "NAO+ Pattern",
        '1': "NAO- Pattern",
        '2': "Blocking Pattern",
        '3': "Subtropical High"
    }

    # Plot each cluster's correlation map
    for i, cluster_key in enumerate(sorted([k for k in corr_results.keys()])):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

        corr_map = corr_results[cluster_key]['corr_map']
        p_map = corr_results[cluster_key]['p_map']
        lons = corr_results[cluster_key]['lons']
        lats = corr_results[cluster_key]['lats']

        # Mask non-significant correlations
        masked_corr = np.ma.masked_where(p_map > 0.05, corr_map)

        # Plot correlation
        cf = ax.contourf(lons, lats, masked_corr,
                         levels=np.linspace(-vmax, vmax, 17),
                         cmap='RdBu_r', transform=ccrs.PlateCarree(),
                         extend='both')

        # Add contour lines
        cs = ax.contour(lons, lats, corr_map,
                        levels=[-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6],
                        colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%.1f')

        # Add stippling for significance
        significance_mask = p_map < 0.05
        lons_mesh, lats_mesh = np.meshgrid(lons, lats)
        significant_lons = lons_mesh[significance_mask]
        significant_lats = lats_mesh[significance_mask]
        stride = 5  # Adjust for density
        ax.scatter(significant_lons[::stride], significant_lats[::stride],
                   color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

        # Add coastlines and land
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Get cluster number from key name
        cluster_num = cluster_key.split('_')[1]

        # Add title with physical interpretation if available
        title = f"Cluster {cluster_num} vs {full_var_name}"
        if cluster_num in interpretations:
            title += f"\n({interpretations[cluster_num]})"
        ax.set_title(title)

        # Add panel label
        ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top')

        # Add text box with max correlation info
        max_idx = np.unravel_index(np.nanargmax(np.abs(corr_map)), corr_map.shape)
        max_val = corr_map[max_idx]
        max_lat, max_lon = lats[max_idx[0]], lons[max_idx[1]]

        ax.text(0.02, 0.06, f"Max corr: {max_val:.2f}\nLocation: {max_lat:.1f}°N, {max_lon:.1f}°E",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(cf, cax=cbar_ax)
    cbar.set_label('Correlation Coefficient (p < 0.05)')

    # Add overall title
    plt.suptitle(f'Correlation between {full_var_name} and Cluster Indices', fontsize=16, y=0.98)

    # Save figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig

#############################################################################################################

#############################################################################################################


def create_variable_composites(indices, anomalies, euro_times, euro_season_years, common_years,
                               var_name, cluster_keys, output_dir):
    """Create composites for high/low index values"""
    # Define variable names for plotting
    var_names = {
        'z': 'Geopotential Height',
        'q': 'Specific Humidity',
        'crwc': 'Rain Water Content',
        'ciwc': 'Cloud Ice Water Content'
    }

    full_var_name = var_names.get(var_name, var_name.upper())

    # Store composites
    composites = {}

    # Create land feature
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray',
                                        alpha=0.3)

    for cluster_key in cluster_keys:
        # Get index values
        if cluster_key not in indices:
            continue

        index_values = indices[cluster_key].loc[common_years]

        # Define thresholds for high/low composites
        high_threshold = np.percentile(index_values, 80)
        low_threshold = np.percentile(index_values, 20)

        # Get years with high/low index values
        high_years = index_values[index_values > high_threshold].index.values
        low_years = index_values[index_values < low_threshold].index.values

        # Collect anomalies for high/low years
        high_anomalies = []
        low_anomalies = []

        for year in high_years:
            year_mask = np.array(euro_season_years) == year
            if np.any(year_mask):
                year_times = euro_times[year_mask]
                high_anomalies.append(anomalies.sel(valid_time=year_times).mean('valid_time'))

        for year in low_years:
            year_mask = np.array(euro_season_years) == year
            if np.any(year_mask):
                year_times = euro_times[year_mask]
                low_anomalies.append(anomalies.sel(valid_time=year_times).mean('valid_time'))

        # Calculate composites
        if high_anomalies and low_anomalies:
            high_composite = xr.concat(high_anomalies, dim='year').mean('year')
            low_composite = xr.concat(low_anomalies, dim='year').mean('year')

            # Apply t-test for significance
            high_stack = xr.concat(high_anomalies, dim='year')
            low_stack = xr.concat(low_anomalies, dim='year')

            # Calculate t-stats and p-values for high composite
            high_t_stat = high_composite / (high_stack.std('year') / np.sqrt(len(high_years)))
            high_p_val = 2 * (1 - stats.t.cdf(np.abs(high_t_stat), len(high_years) - 1))

            # Calculate t-stats and p-values for low composite
            low_t_stat = low_composite / (low_stack.std('year') / np.sqrt(len(low_years)))
            low_p_val = 2 * (1 - stats.t.cdf(np.abs(low_t_stat), len(low_years) - 1))

            # Store composites
            composites[f"{cluster_key}_high"] = {
                'composite': high_composite,
                'p_values': high_p_val,
                'years': high_years
            }

            composites[f"{cluster_key}_low"] = {
                'composite': low_composite,
                'p_values': low_p_val,
                'years': low_years
            }

            # Create visualization
            fig = plt.figure(figsize=(15, 10))

            # Plot high composite
            ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())

            # Determine appropriate scaling for different variables
            if var_name == 'z':
                vmax = max(100, np.nanmax(np.abs(high_composite.values)))
                levels = np.linspace(-vmax, vmax, 21)
                label = 'Geopotential Height Anomaly (m)'
            elif var_name == 'q':
                vmax = max(0.0002, np.nanmax(np.abs(high_composite.values)))
                levels = np.linspace(-vmax, vmax, 21)
                label = 'Specific Humidity Anomaly (kg/kg)'
            elif var_name in ['crwc', 'ciwc']:
                vmax = max(0.00001, np.nanmax(np.abs(high_composite.values)))
                levels = np.linspace(-vmax, vmax, 21)
                label = 'Water Content Anomaly (kg/kg)'
            else:
                vmax = np.nanmax(np.abs(high_composite.values))
                levels = np.linspace(-vmax, vmax, 21)
                label = f'{var_name.upper()} Anomaly'

            # Plot high composite
            cf1 = ax1.contourf(high_composite.longitude, high_composite.latitude, high_composite,
                               levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                               extend='both')

            # Add contour lines
            cs1 = ax1.contour(high_composite.longitude, high_composite.latitude, high_composite,
                              colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

            ax1.clabel(cs1, cs1.levels[::2], inline=True, fontsize=8)

            # Add stippling for significance
            pvals = high_p_val.values if isinstance(high_p_val, xr.DataArray) else high_p_val
            significance_mask = pvals < 0.05
            lons_mesh, lats_mesh = np.meshgrid(high_composite.longitude, high_composite.latitude)
            significant_lons = lons_mesh[significance_mask]
            significant_lats = lats_mesh[significance_mask]
            stride = 5
            ax1.scatter(significant_lons[::stride], significant_lats[::stride],
                        color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

            # Add coastlines
            ax1.add_feature(land)
            ax1.coastlines(resolution='50m')

            # Add gridlines
            gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl1.top_labels = False
            gl1.right_labels = False

            # Get cluster number
            cluster_num = cluster_key.split('_')[1]

            # Title
            ax1.set_title(f'{full_var_name} Anomaly for High Cluster {cluster_num} Index\n(n={len(high_years)} years)')

            # Plot low composite
            ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())

            cf2 = ax2.contourf(low_composite.longitude, low_composite.latitude, low_composite,
                               levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                               extend='both')

            # Add contour lines
            cs2 = ax2.contour(low_composite.longitude, low_composite.latitude, low_composite,
                              colors='k', linewidths=0.5, transform=ccrs.PlateCarree())

            ax2.clabel(cs2, cs2.levels[::2], inline=True, fontsize=8)

            # Add stippling for significance
            pvals = low_p_val.values if isinstance(low_p_val, xr.DataArray) else low_p_val
            significance_mask = pvals < 0.05
            significant_lons = lons_mesh[significance_mask]
            significant_lats = lats_mesh[significance_mask]
            ax2.scatter(significant_lons[::stride], significant_lats[::stride],
                        color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())

            # Add coastlines
            ax2.add_feature(land)
            ax2.coastlines(resolution='50m')

            # Add gridlines
            gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl2.top_labels = False
            gl2.right_labels = False

            # Title
            ax2.set_title(f'{full_var_name} Anomaly for Low Cluster {cluster_num} Index\n(n={len(low_years)} years)')

            # Add colorbar
            plt.colorbar(cf1, ax=[ax1, ax2], orientation='horizontal', pad=0.05, label=label)

            # Save figure
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'composite_{var_name}_{cluster_key}.png'), dpi=300,
                            bbox_inches='tight')
            plt.close(fig)

    return composites


#######################################################################################################################
#######################################################################################################################

def project_clusters_on_wider_domain(som, X, ds, clustering_results, z500_file, output_dir,
                                     lon_range=(-85, 45), lat_range=(0, 75)):
    """
    Project SST-derived clusters onto Z500 data with a wider spatial domain

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array
    ds : xarray.Dataset
        Original SST dataset
    clustering_results : dict
        Results from cluster_som_patterns
    z500_file : str
        Path to Z500 data file
    output_dir : str
        Directory to save output
    lon_range : tuple
        Longitude range (min, max)
    lat_range : tuple
        Latitude range (min, max)
    """
    # Create output directory
    projection_dir = os.path.join(output_dir, 'wider_domain_projection')
    os.makedirs(projection_dir, exist_ok=True)

    print(f"\nProjecting clusters onto wider domain: Lon {lon_range}, Lat {lat_range}...")

    # Get time and cluster information
    bmu_indices = np.array([som.winner(x) for x in X])
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])
    pattern_clusters = clustering_results["pattern_clusters"]

    min_cluster = min(pattern_clusters.values()) if pattern_clusters else 0
    if min_cluster == 0:
        print("Warning: Converting 0-indexed clusters to 1-indexed...")
        pattern_clusters = {k: v + 1 for k, v in pattern_clusters.items()}

    unique_clusters = sorted(set(pattern_clusters.values()))
    print(f"Working with {len(unique_clusters)} clusters: {unique_clusters}")

    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time info
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Create a DataFrame for easier filtering
    df = pd.DataFrame({
        'time': times,
        'year': years,
        'month': months,
        'cluster': cluster_sequence
    })

    # Load Z500 data with error checking
    print(f"Loading Z500 data from: {z500_file}")
    try:
        ds_z500 = xr.open_dataset(z500_file)
        print(f"Z500 dataset dimensions: {list(ds_z500.dims)}")
        print(f"Z500 coordinates: {list(ds_z500.coords)}")

        # Check available dimensions before selection
        if 'latitude' not in ds_z500.dims and 'lat' in ds_z500.dims:
            print("Renaming 'lat' to 'latitude'")
            ds_z500 = ds_z500.rename({'lat': 'latitude'})

        if 'longitude' not in ds_z500.dims and 'lon' in ds_z500.dims:
            print("Renaming 'lon' to 'longitude'")
            ds_z500 = ds_z500.rename({'lon': 'longitude'})

        # Check if dimensions exist
        if 'latitude' not in ds_z500.dims or 'longitude' not in ds_z500.dims:
            print(f"Error: Required dimensions missing. Available dimensions: {list(ds_z500.dims)}")
            return None

        # Check the actual range of coordinates
        print(f"Longitude range in data: {ds_z500.longitude.min().values} to {ds_z500.longitude.max().values}")
        print(f"Latitude range in data: {ds_z500.latitude.min().values} to {ds_z500.latitude.max().values}")

        # Adjust lat/lon range to match available data if needed
        actual_lat_range = (
            max(lat_range[0], float(ds_z500.latitude.min().values)),
            min(lat_range[1], float(ds_z500.latitude.max().values))
        )

        actual_lon_range = (
            max(lon_range[0], float(ds_z500.longitude.min().values)),
            min(lon_range[1], float(ds_z500.longitude.max().values))
        )

        print(f"Adjusted lat range: {actual_lat_range}")
        print(f"Adjusted lon range: {actual_lon_range}")

        # Select pressure level if needed
        if 'pressure_level' in ds_z500.dims:
            ds_z500 = ds_z500.sel(pressure_level=500.0)
            print("Selected 500 hPa pressure level")

        # Check if latitude is ascending or descending
        lat_values = ds_z500.latitude.values
        lat_is_ascending = lat_values[0] < lat_values[-1]
        print(f"Latitude is {'ascending' if lat_is_ascending else 'descending'}")

        # Select domain based on latitude ordering
        if lat_is_ascending:
            # Ascending order (e.g., 0 to 90)
            ds_z500_selected = ds_z500.sel(
                longitude=slice(actual_lon_range[0], actual_lon_range[1]),
                latitude=slice(actual_lat_range[0], actual_lat_range[1])
            )
        else:
            # Descending order (e.g., 90 to 0)
            ds_z500_selected = ds_z500.sel(
                longitude=slice(actual_lon_range[0], actual_lon_range[1]),
                latitude=slice(actual_lat_range[1], actual_lat_range[0])
            )

        # Check the selection worked
        print(f"Selected domain shape: {ds_z500_selected.z.shape}")

        if ds_z500_selected.z.size == 0:
            print("ERROR: Selected domain has zero size! Trying index-based selection instead.")

            # Try index-based selection as a fallback
            lat_indices = np.where((ds_z500.latitude >= min(actual_lat_range)) &
                                   (ds_z500.latitude <= max(actual_lat_range)))[0]
            lon_indices = np.where((ds_z500.longitude >= min(actual_lon_range)) &
                                   (ds_z500.longitude <= max(actual_lon_range)))[0]

            if len(lat_indices) > 0 and len(lon_indices) > 0:
                ds_z500_selected = ds_z500.isel(
                    latitude=slice(min(lat_indices), max(lat_indices) + 1),
                    longitude=slice(min(lon_indices), max(lon_indices) + 1)
                )
                print(f"Index-based selection shape: {ds_z500_selected.z.shape}")
            else:
                print("Index-based selection failed. No valid data available.")
                return None

        # Create land mask for maps
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='black',
                                            facecolor='lightgray',
                                            alpha=0.3)

        # Get unique clusters that actually exist
        unique_clusters = sorted(set(pattern_clusters.values()))

        # For each cluster, create composites for each month
        for cluster in unique_clusters:
            # Create directory for this cluster
            cluster_dir = os.path.join(projection_dir, f'cluster_{cluster}')
            os.makedirs(cluster_dir, exist_ok=True)

            print(f"Processing Cluster {cluster}...")

            # Process each month separately
            for month in [12, 1, 2]:
                # Get dates for this cluster and month
                cluster_month_mask = (df['cluster'] == cluster) & (df['month'] == month)
                if not np.any(cluster_month_mask):
                    print(f"  No data for Cluster {cluster}, Month {month}")
                    continue

                cluster_times = df.loc[cluster_month_mask, 'time'].values

                # Find closest times in Z500 dataset
                z500_times = pd.to_datetime(
                    ds_z500_selected.time.values) if 'time' in ds_z500_selected.dims else pd.to_datetime(
                    ds_z500_selected.valid_time.values)
                matched_times = []

                for t in cluster_times:
                    time_diffs = abs(z500_times - pd.to_datetime(t))
                    if len(time_diffs) > 0:  # Ensure there are times to match against
                        closest_idx = np.argmin(time_diffs)
                        # Use more lenient matching - accept within 30 days
                        if time_diffs[closest_idx].days <= 30:
                            matched_times.append(z500_times[closest_idx])

                if not matched_times:
                    print(f"  No matching Z500 data for Cluster {cluster}, Month {month}")
                    continue

                # Create composite
                time_dim = 'time' if 'time' in ds_z500_selected.dims else 'valid_time'
                try:
                    z500_composite = ds_z500_selected.z.sel({time_dim: matched_times}).mean(time_dim)

                    # Calculate climatology for this month
                    month_mask = pd.to_datetime(ds_z500_selected[time_dim].values).month == month
                    if np.any(month_mask):
                        z500_climatology = ds_z500_selected.z.sel(
                            {time_dim: ds_z500_selected[time_dim].values[month_mask]}).mean(time_dim)

                        # Calculate anomaly
                        z500_anomaly = z500_composite - z500_climatology

                        # Create figure
                        fig = plt.figure(figsize=(12, 10))
                        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

                        # Find appropriate scale
                        max_val = np.nanmax(np.abs(z500_anomaly.values))
                        levels = np.linspace(-max_val, max_val, 21)

                        cf = ax.contourf(
                            z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                            levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                            extend='both'
                        )

                        # Add contour lines
                        cs = ax.contour(
                            z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                            levels=np.arange(-100, 101, 20), colors='black', linewidths=0.5,
                            transform=ccrs.PlateCarree()
                        )

                        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%d')

                        # Add coastlines
                        ax.add_feature(land)
                        ax.coastlines(resolution='50m')

                        # Add gridlines
                        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
                        gl.top_labels = False
                        gl.right_labels = False

                        # Set extent to the selected domain
                        ax.set_extent([min(ds_z500_selected.longitude), max(ds_z500_selected.longitude),
                                       min(ds_z500_selected.latitude), max(ds_z500_selected.latitude)],
                                      crs=ccrs.PlateCarree())

                        # Add title
                        month_name = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                                      'July', 'August', 'September', 'October', 'November', 'December'][month]
                        ax.set_title(f'Cluster {cluster} Z500 Anomaly - {month_name}\n(n={len(matched_times)} months)')

                        # Add colorbar
                        plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, label='Z500 Anomaly (m)')

                        # Save figure
                        plt.savefig(os.path.join(cluster_dir, f'cluster_{cluster}_month_{month:02d}.png'),
                                    dpi=300, bbox_inches='tight')
                        plt.close(fig)

                        print(f"  Created composite for Cluster {cluster}, Month {month} ({len(matched_times)} dates)")
                    else:
                        print(f"  No climatology data for Month {month}")

                except Exception as e:
                    print(f"  Error creating composite: {str(e)}")

            # Create annual composite for this cluster
            try:
                # Get all dates for this cluster
                cluster_mask = (df['cluster'] == cluster)
                cluster_times = df.loc[cluster_mask, 'time'].values

                # Find corresponding times in Z500 data
                time_dim = 'time' if 'time' in ds_z500_selected.dims else 'valid_time'
                z500_times = pd.to_datetime(ds_z500_selected[time_dim].values)
                matched_times = []

                for t in cluster_times:
                    time_diffs = abs(z500_times - pd.to_datetime(t))
                    if len(time_diffs) > 0:
                        closest_idx = np.argmin(time_diffs)
                        if time_diffs[closest_idx].days <= 30:
                            matched_times.append(z500_times[closest_idx])

                if matched_times:
                    # Create composite
                    z500_composite = ds_z500_selected.z.sel({time_dim: matched_times}).mean(time_dim)
                    z500_climatology = ds_z500_selected.z.mean(time_dim)
                    z500_anomaly = z500_composite - z500_climatology

                    # Create figure
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

                    max_val = np.nanmax(np.abs(z500_anomaly.values))
                    levels = np.linspace(-max_val, max_val, 21)

                    cf = ax.contourf(
                        z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                        levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                        extend='both'
                    )

                    cs = ax.contour(
                        z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                        levels=np.arange(-100, 101, 20), colors='black', linewidths=0.5,
                        transform=ccrs.PlateCarree()
                    )

                    ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%d')

                    ax.add_feature(land)
                    ax.coastlines(resolution='50m')

                    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
                    gl.top_labels = False
                    gl.right_labels = False

                    ax.set_extent([min(ds_z500_selected.longitude), max(ds_z500_selected.longitude),
                                   min(ds_z500_selected.latitude), max(ds_z500_selected.latitude)],
                                  crs=ccrs.PlateCarree())

                    ax.set_title(f'Cluster {cluster} Z500 Anomaly - Annual\n(n={len(matched_times)} months)')

                    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, label='Z500 Anomaly (m)')

                    plt.savefig(os.path.join(projection_dir, f'cluster_{cluster}_annual.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close(fig)

                    print(f"Created annual composite for Cluster {cluster} ({len(matched_times)} dates)")

            except Exception as e:
                print(f"Error creating annual composite for Cluster {cluster}: {str(e)}")

        return projection_dir

    except Exception as e:
        print(f"Error during Z500 data processing: {str(e)}")
        traceback.print_exc()
        return None

#######################################################################################################################
#######################################################################################################################

def enhanced_project_clusters_on_wider_domain(som, X, ds, clustering_results, z500_file, output_dir,
                                              lon_range=(-85, 45), lat_range=(0, 75)):
    """
    Project SST-derived clusters onto Z500 data with a wider spatial domain with t-test significance
    Similar to the first screenshot visualization

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array
    ds : xarray.Dataset
        Original SST dataset
    clustering_results : dict
        Results from cluster_som_patterns
    z500_file : str
        Path to Z500 data file
    output_dir : str
        Directory to save output
    lon_range : tuple
        Longitude range (min, max)
    lat_range : tuple
        Latitude range (min, max)
    """

    projection_dir = os.path.join(output_dir, 'wider_domain_projection_enhanced')
    os.makedirs(projection_dir, exist_ok=True)

    print(f"\nProjecting clusters onto wider domain with t-test: Lon {lon_range}, Lat {lat_range}...")

    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers (1-based)
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]

    # Ensure 1-indexing
    min_cluster = min(pattern_clusters.values()) if pattern_clusters else 0
    if min_cluster == 0:
        print("Converting 0-indexed clusters to 1-indexed...")
        pattern_clusters = {k: v + 1 for k, v in pattern_clusters.items()}

    # Map patterns to clusters
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Get the actual unique clusters that exist in the data
    unique_clusters = sorted(set(pattern_clusters.values()))
    print(f"Working with {len(unique_clusters)} unique clusters: {unique_clusters}")

    # Create a DataFrame for easier filtering
    df = pd.DataFrame({
        'time': times,
        'year': years,
        'month': months,
        'cluster': cluster_sequence
    })

    # Load Z500 data
    print(f"Loading Z500 data from: {z500_file}")
    try:
        ds_z500 = xr.open_dataset(z500_file)

        # Check available dimensions and rename if needed
        if 'latitude' not in ds_z500.dims and 'lat' in ds_z500.dims:
            print("Renaming 'lat' to 'latitude'")
            ds_z500 = ds_z500.rename({'lat': 'latitude'})

        if 'longitude' not in ds_z500.dims and 'lon' in ds_z500.dims:
            print("Renaming 'lon' to 'longitude'")
            ds_z500 = ds_z500.rename({'lon': 'longitude'})

        # Check if dimensions exist
        if 'latitude' not in ds_z500.dims or 'longitude' not in ds_z500.dims:
            print(f"Error: Required dimensions missing. Available dimensions: {list(ds_z500.dims)}")
            return None

        # Check the actual range of coordinates
        print(f"Longitude range in data: {ds_z500.longitude.min().values} to {ds_z500.longitude.max().values}")
        print(f"Latitude range in data: {ds_z500.latitude.min().values} to {ds_z500.latitude.max().values}")

        # Adjust lat/lon range to match available data if needed
        actual_lat_range = (
            max(lat_range[0], float(ds_z500.latitude.min().values)),
            min(lat_range[1], float(ds_z500.latitude.max().values))
        )

        actual_lon_range = (
            max(lon_range[0], float(ds_z500.longitude.min().values)),
            min(lon_range[1], float(ds_z500.longitude.max().values))
        )

        print(f"Adjusted lat range: {actual_lat_range}")
        print(f"Adjusted lon range: {actual_lon_range}")

        # Select pressure level if needed
        if 'pressure_level' in ds_z500.dims:
            ds_z500 = ds_z500.sel(pressure_level=500.0)
            print("Selected 500 hPa pressure level")

        # Check if latitude is ascending or descending
        lat_values = ds_z500.latitude.values
        lat_is_ascending = lat_values[0] < lat_values[-1]
        print(f"Latitude is {'ascending' if lat_is_ascending else 'descending'}")

        # Select domain based on latitude ordering
        if lat_is_ascending:
            # Ascending order (e.g., 0 to 90)
            ds_z500_selected = ds_z500.sel(
                longitude=slice(actual_lon_range[0], actual_lon_range[1]),
                latitude=slice(actual_lat_range[0], actual_lat_range[1])
            )
        else:
            # Descending order (e.g., 90 to 0)
            ds_z500_selected = ds_z500.sel(
                longitude=slice(actual_lon_range[0], actual_lon_range[1]),
                latitude=slice(actual_lat_range[1], actual_lat_range[0])
            )

        # Check the selection worked
        print(f"Selected domain shape: {ds_z500_selected.z.shape}")

        if ds_z500_selected.z.size == 0:
            print("ERROR: Selected domain has zero size! Trying index-based selection instead.")

            # Try index-based selection as a fallback
            lat_indices = np.where((ds_z500.latitude >= min(actual_lat_range)) &
                                   (ds_z500.latitude <= max(actual_lat_range)))[0]
            lon_indices = np.where((ds_z500.longitude >= min(actual_lon_range)) &
                                   (ds_z500.longitude <= max(actual_lon_range)))[0]

            if len(lat_indices) > 0 and len(lon_indices) > 0:
                ds_z500_selected = ds_z500.isel(
                    latitude=slice(min(lat_indices), max(lat_indices) + 1),
                    longitude=slice(min(lon_indices), max(lon_indices) + 1)
                )
                print(f"Index-based selection shape: {ds_z500_selected.z.shape}")
            else:
                print("Index-based selection failed. No valid data available.")
                return None

        # Create a mapping of year to dominant cluster
        year_cluster = {}
        for year in np.unique(years):
            # Find dominant cluster for this year
            year_mask = (years == year)
            if np.any(year_mask):
                year_clusters = cluster_sequence[year_mask]
                # Use proper bincount with minlength to avoid index errors
                max_cluster = max(unique_clusters) if unique_clusters else 0
                cluster_counts = np.bincount(year_clusters, minlength=max_cluster + 1)
                dominant_cluster = np.argmax(cluster_counts)
                year_cluster[year] = dominant_cluster

        # Create cluster_years dictionary for the t-test
        cluster_years = {}
        for cluster in unique_clusters:
            cluster_years[cluster] = [year for year, clus in year_cluster.items() if clus == cluster]
            print(f"Cluster {cluster}: {len(cluster_years[cluster])} years")

        year_cluster_df = pd.DataFrame({
            'Year': list(year_cluster.keys()),
            'Dominant_Cluster': list(year_cluster.values())
        })

        # Sort by year for easier reading
        year_cluster_df = year_cluster_df.sort_values('Year')

        # Save to CSV
        year_cluster_df.to_csv(os.path.join(projection_dir, 'year_to_cluster_mapping.csv'), index=False)
        print(f"Year-to-cluster mapping saved to: {os.path.join(projection_dir, 'year_to_cluster_mapping.csv')}")

        # Calculate cluster frequencies by year
        years = sorted(year_cluster.keys())
        cluster_freq = {}

        for year in years:
            # Get all clusters in this year
            year_mask = (df['year'] == year)
            if np.any(year_mask):
                year_clusters = cluster_sequence[year_mask]
                # Count frequency of each cluster
                unique_clusters = sorted(set(pattern_clusters.values()))
                for cluster in unique_clusters:
                    if cluster not in cluster_freq:
                        cluster_freq[cluster] = {}
                    cluster_freq[cluster][year] = np.mean(year_clusters == cluster)

        # Convert to DataFrame
        freq_data = []
        for cluster, freq_by_year in cluster_freq.items():
            for year, freq in freq_by_year.items():
                freq_data.append({
                    'Year': year,
                    'Cluster': cluster,
                    'Frequency': freq
                })

        freq_df = pd.DataFrame(freq_data)
        freq_df.to_csv(os.path.join(projection_dir, 'cluster_frequencies.csv'), index=False)
        print(f"Cluster frequencies saved to: {os.path.join(projection_dir, 'cluster_frequencies.csv')}")

        # Create standardized indices (useful for correlation analysis)
        indices_df = pd.DataFrame(index=years)

        # Add a column for each cluster
        for cluster in unique_clusters:
            # Create standardized index for this cluster
            cluster_index = np.array([cluster_freq.get(cluster, {}).get(year, 0) for year in years])

            # Standardize (mean=0, std=1)
            if np.std(cluster_index) > 0:
                std_index = (cluster_index - np.mean(cluster_index)) / np.std(cluster_index)
            else:
                std_index = np.zeros_like(cluster_index)

            indices_df[f'Cluster_{cluster}_Index'] = std_index

        # Save standardized indices
        indices_df.to_csv(os.path.join(projection_dir, 'cluster_standardized_indices.csv'))
        print(
            f"Standardized cluster indices saved to: {os.path.join(projection_dir, 'cluster_standardized_indices.csv')}")

        # Calculate seasonal means for all years
        time_dim = 'time' if 'time' in ds_z500_selected.dims else 'valid_time'
        z500_times = pd.to_datetime(ds_z500_selected[time_dim].values)

        # Create season year for each time point
        # Create season year for each time point (e.g., Dec 2020 belongs to winter 2021)
        z500_season_years = []
        for t in z500_times:
            if t.month == 12:  # December belongs to the next year's winter
                z500_season_years.append(t.year + 1)
            else:
                z500_season_years.append(t.year)

        # Add as coordinate
        z500_season_years = np.array(z500_season_years)
        ds_z500_selected = ds_z500_selected.assign_coords(season_year=('valid_time', z500_season_years))

        # Calculate season means
        z500_yearly = ds_z500_selected.groupby('season_year').mean(time_dim)

        # Calculate climatology
        z500_climatology = z500_yearly.z.mean('season_year')

        # Create land feature for maps
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='black',
                                            facecolor='lightgray',
                                            alpha=0.3)

        # Create 2x2 panel figure for all clusters (similar to your first screenshot)
        fig = plt.figure(figsize=(16, 14))

        # Find global min/max for consistent color scaling
        all_anomalies = []

        for cluster in unique_clusters:
            if len(cluster_years[cluster]) < 3:
                print(f"Warning: Cluster {cluster} has < 3 occurrences. T-test may not be reliable.")
                continue

            # Years when this cluster was dominant
            years = cluster_years[cluster]

            # Calculate composite
            z500_composite = z500_yearly.sel(season_year=years).z.mean('season_year')

            # Calculate anomaly
            z500_anomaly = z500_composite - z500_climatology
            all_anomalies.append(z500_anomaly)

        # Find global min/max for consistent color scale
        if all_anomalies:
            anom_min = min(anom.min().item() for anom in all_anomalies)
            anom_max = max(anom.max().item() for anom in all_anomalies)
            abs_max = max(abs(anom_min), abs(anom_max))
            # Round to nearest 100
            abs_max = int(np.ceil(abs_max / 100) * 100)
        else:
            abs_max = 100  # Default fallback

        # Plot each cluster with statistical significance
        for i, cluster in enumerate(unique_clusters):
            if i >= 4:  # Only create a 2x2 panel
                print(f"Skipping cluster {cluster} to maintain 2x2 layout")
                continue

            # Skip if not enough years
            if len(cluster_years[cluster]) < 3:
                print(f"Skipping cluster {cluster}: not enough years")
                continue

            # Calculate row and column for 2x2 grid
            row, col = divmod(i, 2)
            ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())

            # Calculate composite and anomaly
            years = cluster_years[cluster]
            z500_composite = z500_yearly.sel(season_year=years).z.mean('season_year')
            z500_anomaly = z500_composite - z500_climatology

            # Collect individual year anomalies for t-test
            all_year_anomalies = []
            for year in years:
                year_anomaly = z500_yearly.sel(season_year=year).z - z500_climatology
                all_year_anomalies.append(year_anomaly)

            # Perform t-test
            if len(years) >= 3:  # Minimum needed for t-test
                # Stack individual anomalies
                stacked_anomalies = np.stack([anom.values for anom in all_year_anomalies])

                # Calculate t-statistic and p-value
                t_stat = np.mean(stacked_anomalies, axis=0) / (
                        np.std(stacked_anomalies, axis=0, ddof=1) / np.sqrt(len(years)))
                # Two-tailed t-test
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(years) - 1))
            else:
                # Not enough samples for proper t-test
                p_values = np.ones_like(z500_anomaly.values)

            # Create contour levels
            levels = np.linspace(-abs_max, abs_max, 21)

            # Plot anomaly
            cf = ax.contourf(
                z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                levels=levels, cmap='RdBu_r', transform=ccrs.PlateCarree(),
                extend='both'
            )

            # Add contour lines
            cs = ax.contour(
                z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                levels=np.arange(-100, 101, 20), colors='black', linewidths=0.5,
                transform=ccrs.PlateCarree()
            )

            # Add contour labels (only on some contours to avoid cluttering)
            ax.clabel(cs, cs.levels[::4], inline=True, fontsize=8, fmt='%d')

            # Add stippling for significance
            significance_mask = p_values < 0.05
            lons_mesh, lats_mesh = np.meshgrid(z500_anomaly.longitude, z500_anomaly.latitude)
            significant_lons = lons_mesh[significance_mask]
            significant_lats = lats_mesh[significance_mask]

            # Add stippling with reasonable density
            # Adjust stride value to control stippling density
            stride = max(1, int(np.sqrt(len(significant_lons)) / 20))
            ax.scatter(significant_lons[::stride], significant_lats[::stride],
                       color='black', s=1.5, alpha=0.7, transform=ccrs.PlateCarree())

            # Add coastlines and land
            ax.add_feature(land)
            ax.coastlines(resolution='50m')

            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            # Set extent to the selected domain
            ax.set_extent([min(z500_anomaly.longitude), max(z500_anomaly.longitude),
                           min(z500_anomaly.latitude), max(z500_anomaly.latitude)],
                          crs=ccrs.PlateCarree())

            # Physical pattern names
            pattern_names = {
                1: "Gulf Stream Displacement",
                2: "Negative AMO-like Pattern",
                3: "Tripole Pattern",
                4: "Subtropical Warming"
            }
            pattern_name = pattern_names.get(cluster, f"Cluster {cluster}")

            # Add title with physical interpretation
            ax.set_title(f'Z500 Anomaly for Cluster {cluster}\n{pattern_name} (n={len(years)} years)')

            # Add panel label
            ax.text(0.02, 0.98, f"({chr(97 + i)})", transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')

        # Add a single colorbar for all subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = plt.colorbar(cf, cax=cbar_ax)
        cbar.set_label('Z500 Anomaly (m)')

        # Add note about significance testing
        plt.figtext(0.5, 0.01,
                    "Note: Stippling indicates anomalies significant at p < 0.05 level (two-tailed t-test)",
                    ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

        # Adjust layout
        plt.tight_layout(rect=[0, 0.02, 0.9, 0.98])

        # Save figure
        plt.savefig(os.path.join(projection_dir, 'z500_anomalies_wider_domain.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Also create individual figures for each cluster
        for cluster in unique_clusters:
            if len(cluster_years[cluster]) < 3:
                continue

            years = cluster_years[cluster]

            # Create figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

            # Calculate composite and anomaly
            z500_composite = z500_yearly.sel(season_year=years).z.mean('season_year')
            z500_anomaly = z500_composite - z500_climatology

            # Collect individual year anomalies for t-test
            all_year_anomalies = []
            for year in years:
                year_anomaly = z500_yearly.sel(season_year=year).z - z500_climatology
                all_year_anomalies.append(year_anomaly)

            # Perform t-test
            stacked_anomalies = np.stack([anom.values for anom in all_year_anomalies])
            t_stat = np.mean(stacked_anomalies, axis=0) / (
                    np.std(stacked_anomalies, axis=0, ddof=1) / np.sqrt(len(years)))
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(years) - 1))

            # Plot anomaly
            cf = ax.contourf(
                z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                levels=np.linspace(-abs_max, abs_max, 21), cmap='RdBu_r',
                transform=ccrs.PlateCarree(), extend='both'
            )

            # Add contour lines
            cs = ax.contour(
                z500_anomaly.longitude, z500_anomaly.latitude, z500_anomaly,
                levels=np.arange(-100, 101, 20), colors='black', linewidths=0.5,
                transform=ccrs.PlateCarree()
            )

            ax.clabel(cs, cs.levels[::2], inline=True, fontsize=8, fmt='%d')

            # Add stippling for significance
            significance_mask = p_values < 0.05
            lons_mesh, lats_mesh = np.meshgrid(z500_anomaly.longitude, z500_anomaly.latitude)
            significant_lons = lons_mesh[significance_mask]
            significant_lats = lats_mesh[significance_mask]

            stride = max(1, int(np.sqrt(len(significant_lons)) / 25))
            ax.scatter(significant_lons[::stride], significant_lats[::stride],
                       color='black', s=1.5, alpha=0.7, transform=ccrs.PlateCarree())

            # Add coastlines and land
            ax.add_feature(land)
            ax.coastlines(resolution='50m')

            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            # Set extent
            ax.set_extent([min(z500_anomaly.longitude), max(z500_anomaly.longitude),
                           min(z500_anomaly.latitude), max(z500_anomaly.latitude)],
                          crs=ccrs.PlateCarree())

            # Get pattern name
            pattern_name = pattern_names.get(cluster, f"Cluster {cluster}")

            # Add title
            ax.set_title(f'Z500 Anomaly for Cluster {cluster}\n{pattern_name} (n={len(years)} years)')

            # Add colorbar
            plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, label='Z500 Anomaly (m)')

            # Add note about significance
            plt.figtext(0.5, 0.01, "Stippling indicates p < 0.05", ha='center')

            # Save figure
            plt.savefig(os.path.join(projection_dir, f'cluster_{cluster}_z500_anomaly.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"Z500 anomaly plots with t-test saved to: {projection_dir}")
        return projection_dir

    except Exception as e:
        print(f"Error during Z500 data processing: {str(e)}")
        traceback.print_exc()
        return None


#######################################################################################################################
#######################################################################################################################

def verify_cluster_indexing(clustering_results):
    """
    Verify that clusters are properly 1-indexed
    """
    pattern_clusters = clustering_results["pattern_clusters"]
    unique_clusters = sorted(set(pattern_clusters.values()))

    print(f"Cluster verification:")
    print(f"  Unique clusters: {unique_clusters}")
    print(f"  Min cluster: {min(unique_clusters) if unique_clusters else 'None'}")
    print(f"  Max cluster: {max(unique_clusters) if unique_clusters else 'None'}")

    if unique_clusters and min(unique_clusters) == 1:
        print("  ✓ Clusters are properly 1-indexed")
        return True
    else:
        print("  ✗ Clusters are NOT properly 1-indexed")
        return False


#######################################################################################################################
#######################################################################################################################

def analyze_hadisst_patterns(file_path):
    print(f"\nStarting analysis of HadISST data from: {file_path}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'hadisst_som_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load and preprocess data
        ds = load_and_preprocess_sst(file_path)
        X = prepare_som_input(ds)

        # Test different iteration counts
        print("\nTesting different iteration counts...")
        iteration_results = test_som_iterations(X)
        best_iterations = min(iteration_results.items(), key=lambda x: x[1])[0]
        print(f"\nUsing best iteration count: {best_iterations}")

        # Train final SOM
        som = train_som(X, n_iterations=best_iterations)

        # Calculate pattern occurrences - store in variable for reuse
        pattern_results = analyze_pattern_occurrences(som, X, ds)
        print_pattern_summary(pattern_results)
        save_pattern_summary(pattern_results, output_dir)

        # Calculate improved variance metrics
        variance_results = calculate_corrected_variance(som, X)

        print_variance_summary(variance_results)

        validate_variance_results(variance_results)

        # Print variance analysis results
        print("\nVariance Analysis Results:")
        print(f"Total Explained Variance: {variance_results['total_explained_variance_percent']:.2f}%")
        print(f"Quantization Error: {variance_results['quality_metrics']['quantization_error']:.4f}")
        print(f"Topographic Error: {variance_results['quality_metrics']['topographic_error']:.4f}")
        print(f"Neighborhood Preservation: {variance_results['quality_metrics']['neighborhood_preservation']:.4f}")

        # Calculate statistics
        stats = analyze_pattern_statistics(som, X, ds.sst.shape)

        # Generate visualizations
        print("\nGenerating visualizations...")
        fig_patterns = plot_som_patterns(som, ds.sst.shape, ds, output_dir, pattern_results)
        plt.close(fig_patterns)

        # Interactive patterns and scree plot
        interactive_fig, scree_fig = plot_interactive_patterns(
            som=som,
            original_shape=ds.sst.shape,
            ds=ds,
            stats=stats,
            X=X,
            iteration_results=iteration_results,
            variance_results=variance_results
        )

        # Save interactive visualizations
        if interactive_fig is not None:
            interactive_fig.write_html(os.path.join(output_dir, 'interactive_patterns.html'))
        if scree_fig is not None:
            scree_fig.write_html(os.path.join(output_dir, 'scree_diagram.html'))

        # Save patterns for projection
        save_patterns_for_projection(
            som=som,
            ds=ds,
            variance_results=variance_results,
            pattern_results=pattern_results,
            output_dir=output_dir
        )

        transition_results = analyze_pattern_transitions(som, X, ds)

        transition_fig = plot_pattern_transitions(transition_results, som, output_dir)
        if transition_fig is not None:
            transition_fig.write_html(os.path.join(output_dir, 'pattern_transitions_combined.html'))

        # Save transition results
        with open(os.path.join(output_dir, 'pattern_transitions.pkl'), 'wb') as f:
            pickle.dump(transition_results, f)

        # Save metadata
        with open(os.path.join(output_dir, 'analysis_metadata.txt'), 'w') as f:
            f.write(f"HadISST Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Region: 0-60N, 80W-5W\n")
            f.write(f"Season: DJF\n")
            f.write(f"Time period: 1950-present\n")
            f.write(f"Data: Detrended anomalies\n")
            f.write(f"Final data shape: {ds.sst.shape}\n")
            f.write(f"SOM input shape: {X.shape}\n")
            f.write(f"SOM grid size: 6x5\n")
            f.write(f"Best number of iterations: {best_iterations}\n\n")
            f.write("Quality Metrics:\n")
            f.write(f"Quantization Error: {variance_results['quality_metrics']['quantization_error']:.4f}\n")
            f.write(f"Topographic Error: {variance_results['quality_metrics']['topographic_error']:.4f}\n")
            f.write(f"Total Explained Variance: {variance_results['total_explained_variance_percent']:.2f}%\n")
            f.write(
                f"Neighborhood Preservation: {variance_results['quality_metrics']['neighborhood_preservation']:.4f}\n")

        print("\nGenerating top patterns visualization...")
        top_patterns_fig = plot_top_patterns_with_distribution(
            som=som,
            original_shape=ds.sst.shape,
            ds=ds,
            pattern_results=pattern_results,
            variance_results=variance_results,
            output_dir=output_dir
        )

        if top_patterns_fig is not None:
            top_patterns_fig.write_html(os.path.join(output_dir, 'top_patterns_distribution.html'))
            print("Top patterns visualization saved successfully")

        # =================== CLUSTERING ANALYSIS ===================== #
        print("\n =================== CLUSTERING ANALYSIS ===================== ")

        # After training SOM and analyzing patterns
        print("\nClustering similar SOM patterns...")
        clustering_results = apply_clustering_to_som(
            som=som,
            original_shape=ds.sst.shape,
            ds=ds,
            method='kmeans',
            n_clusters=4,
            output_dir=output_dir
        )

        verify_cluster_indexing(clustering_results)

        # After training SOM and analyzing patterns:
        print("\nPlotting Som Patterns Grid...")
        plot_som_patterns_grid(
            som=som,
            original_shape=ds.sst.shape,
            ds=ds,
            output_dir=output_dir,
            pattern_results=pattern_results
        )

        print("\nAnalyzing cluster occurrences over time...")
        cluster_occurrences = analyze_cluster_occurrences(som, X, ds, clustering_results)

        plot_results = plot_cluster_frequency_over_time(cluster_occurrences, output_dir)
        print(f"Cluster frequency plots saved to: {output_dir}/cluster_analysis/")

        print("\nGenerating NAO index data for correlation analysis...")
        nao_index_data = load_nao_data_from_file('nao_index.csv')

        # Correlate clusters with NAO index
        if len(nao_index_data) == len(cluster_occurrences['cluster_sequence']):
            print("\nCorrelating clusters with NAO index...")
            nao_correlations = correlate_clusters_with_indices(
                cluster_occurrences,
                {'NAO': nao_index_data}
            )

        print("\nCreating cluster composite maps...")
        composites_dir = create_cluster_composites(clustering_results, som, ds, output_dir)
        print(f"Cluster composites saved to: {composites_dir}")

        # Create network visualization of pattern relationships
        print("\nCreating network visualization of pattern relationships...")
        network_dir = create_cluster_network(clustering_results, output_dir)
        print(f"Network visualization saved to: {network_dir}")

        # Create physical interpretation summary
        print("\nCreating physical interpretation templates...")
        interpretation_dir = create_physical_interpretation_summary(clustering_results, output_dir)
        print(f"Interpretation templates saved to: {interpretation_dir}")

        # Create climate mode correlation analysis
        print("\nCreating climate mode correlation analysis...")
        correlations_dir = create_climate_mode_correlations(clustering_results, output_dir)
        print(f"Climate mode correlations saved to: {correlations_dir}")

        # You can also save the clustering results
        with open(os.path.join(output_dir, 'pattern_clustering_results.pkl'), 'wb') as f:
            pickle.dump(clustering_results, f)

        with open(os.path.join(output_dir, 'cluster_occurrences.pkl'), 'wb') as f:
            pickle.dump(cluster_occurrences, f)

        print("\nAnalyzing pattern time series evolution...")
        time_series_results = analyze_pattern_time_series(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            output_dir=output_dir
        )

        # Print summary of significant trends
        print("\nSignificant Pattern Trends:")
        for pattern, stats in sorted(time_series_results['trends']['pattern_trends'].items()):
            if stats['p_value'] < 0.05:  # Only show significant trends
                print(
                    f"  Pattern {pattern}: {stats['trend_per_decade']:.4f} change per decade (p={stats['p_value']:.4f})")

        # =================== analyze_z500_by_pattern ===================== #
        # print("\n =================== Analyze z500 by Pattern ===================== ")

        # After training SOM and analyzing patterns
        # pattern_z500_associations = analyze_z500_by_pattern(
        #    som=som,
        #    X=X,
        #    ds=ds,
        #    z500_file='mon_data_u_v_t_z_q_1000_500.nc',
        #    output_dir=output_dir
        # )

        # =================== ANALYZE Z500 BY CLUSTER ===================== #

        # print("\n =================== ANALYZE Z500 BY CLUSTER ===================== ")
        # cluster_z500_associations = analyze_z500_by_cluster(
        #    som=som,
        #    X=X,
        #    ds=ds,
        #    clustering_results=clustering_results,
        #    z500_file='mon_data_u_v_t_z_q_1000_500.nc',
        #    output_dir=output_dir
        # )

        # =================== CLUSTER Combine ===================== #

        # After applying clustering and creating individual visualizations
        print("\n =================== applying clustering and creating individual visualizations========== ")
        print("\nCreating combined Z500 visualization...")
        combined_z500_dir = create_combined_z500_view(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            z500_file='mon_data_u_v_t_z_q_1000_500.nc',
            output_dir=output_dir
        )
        print(f"Combined Z500 visualizations saved to: {combined_z500_dir}")

        print("\n =================== CLUSTER INDICES AND Z500 CORRELATIONS ===================== ")
        cluster_indices_results = create_cluster_indices_for_z500_correlation(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            z500_file='mon_data_u_v_t_z_q_1000_500.nc',
            output_dir=output_dir
        )

        # Define indices_dir correctly
        indices_dir = os.path.join(output_dir, 'cluster_indices_z500')

        # Load the correlation results that were saved by create_cluster_indices_for_z500_correlation
        try:
            with open(os.path.join(indices_dir, 'z500_correlations.pkl'), 'rb') as f:
                correlation_results = pickle.load(f)

            # After creating individual correlation maps
            print("\nCreating grid correlation visualizations...")
            grid_plots_dir = create_grid_correlation_plots(
                correlation_results=correlation_results,
                output_dir=indices_dir
            )
            print(f"Grid correlation plots saved to: {grid_plots_dir}")
        except Exception as e:
            print(f"Warning: Could not create grid correlation plots: {str(e)}")

        # After clustering and adavanced visualizations
        print("\n =================== CLUSTER combined Z500 visualization ===================== ")

        # After running the individual cluster analysis
        print("\nCreating combined Z500 visualization...")
        combined_z500_dir = create_combined_z500_view(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            z500_file='mon_data_u_v_t_z_q_1000_500.nc',
            output_dir=output_dir
        )
        print(f"Combined Z500 visualizations saved to: {combined_z500_dir}")

        # After creating individual correlation maps
        print("\nCreating grid correlation visualizations...")
        grid_plots_dir = create_grid_correlation_plots(
            correlation_results=correlation_results,  # This should be the dictionary with all correlation maps
            output_dir=indices_dir  # Same directory where individual plots are saved
        )
        print(f"Grid correlation plots saved to: {grid_plots_dir}")

        # After creating individual correlation maps
        print("\nCreating improved correlation visualizations...")
        improved_corr_dir = create_improved_correlation_plots(
            correlation_results=correlation_results,
            clustering_results=clustering_results,
            som=som,
            ds=ds,
            output_dir=output_dir
        )
        print(f"Improved correlation plots saved to: {improved_corr_dir}")

        print(f"\n Creating sst cluster composite figure ...")
        improved_sst_cluster_composite_figures_dir = create_sst_cluster_composite_figure(
            clustering_results=clustering_results,
            som=som,
            ds=ds,
            output_dir=output_dir
        )
        print(f"Improved sst cluster composite figures saved to: {improved_sst_cluster_composite_figures_dir}")

        #######################################################################################################################
        #######################################################################################################################

        #
        # After running your create_cluster_indices_for_z500_correlation function:
        correlation_results = create_cluster_indices_for_z500_correlation(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            z500_file='mon_data_u_v_t_z_q_1000_500.nc',
            output_dir=output_dir
        )

        # Define your cluster information
        cluster_info = {
            0: {'name': 'NAO+', 'physical_pattern': 'NAO+ Pattern'},
            1: {'name': 'NAO-', 'physical_pattern': 'NAO- Pattern'},
            2: {'name': 'Block', 'physical_pattern': 'Blocking Pattern'},
            3: {'name': 'SubHi', 'physical_pattern': 'Subtropical High'}
        }

        #
        # Add this before attempting to create enhanced visualizations
        print("\nExamining correlation_results structure...")
        print(f"Type of correlation_results: {type(correlation_results)}")
        print(
            f"Keys in correlation_results: {list(correlation_results.keys()) if hasattr(correlation_results, 'keys') else 'No keys'}")

        # If correlation_results is a dictionary, print the keys of the first item
        if isinstance(correlation_results, dict) and len(correlation_results) > 0:
            first_key = list(correlation_results.keys())[0]
            print(
                f"Structure of first item ({first_key}): {correlation_results[first_key].keys() if hasattr(correlation_results[first_key], 'keys') else type(correlation_results[first_key])}")

        # Skip the enhanced visualization rather than modifying it
        print("Skipping enhanced visualizations due to structure mismatch")

        # Create the enhanced visualizations
        try:
            # Check if we have the expected correlation structure
            if isinstance(correlation_results, dict):
                # Load the actual correlation results from the pickle file
                indices_dir = os.path.join(output_dir, 'cluster_indices_z500')
                correlation_file = os.path.join(indices_dir, 'z500_correlations.pkl')

                if os.path.exists(correlation_file):
                    print("Loading correlation results from pickle file...")
                    with open(correlation_file, 'rb') as f:
                        z500_correlations = pickle.load(f)

                    # Define cluster information
                    cluster_info = {
                        1: {'name': 'Cluster1', 'physical_pattern': 'Gulf Stream Displacement'},
                        2: {'name': 'Cluster2', 'physical_pattern': 'Negative AMO-like Pattern'},
                        3: {'name': 'Cluster3', 'physical_pattern': 'Tripole Pattern'},
                        4: {'name': 'Cluster4', 'physical_pattern': 'Subtropical Warming'}
                    }

                    # Check the structure of z500_correlations
                    print(f"Z500 correlations structure: {type(z500_correlations)}")
                    if isinstance(z500_correlations, dict) and len(z500_correlations) > 0:
                        first_key = list(z500_correlations.keys())[0]
                        print(f"First key structure: {type(z500_correlations[first_key])}")

                        # Only proceed if we have the correct structure
                        if isinstance(z500_correlations[first_key], dict) and 'corr_map' in z500_correlations[
                            first_key]:
                            # Generate combined overview visualization
                            combined_file = os.path.join(output_dir, 'cluster_indices_z500',
                                                         'combined_z500_correlations.png')
                            create_combined_cluster_correlation_analysis(
                                correlation_results=z500_correlations,
                                cluster_info=cluster_info,
                                output_file=combined_file
                            )
                            print(f"Created combined correlation visualization: {combined_file}")
                        else:
                            print(
                                "Z500 correlations don't have expected structure with 'corr_map' - skipping enhanced visualizations")
                    else:
                        print("Z500 correlations file is empty or invalid - skipping enhanced visualizations")
                else:
                    print("Z500 correlations pickle file not found - skipping enhanced visualizations")
            else:
                print("Correlation results are not in expected format - skipping enhanced visualizations")

        except Exception as e:
            print(f"Error creating enhanced visualizations: {str(e)}")
            # Don't let this stop the rest of the analysis
            traceback.print_exc()

        # After your SOM clustering and other analyses have been completed
        print("\nCreating Z500 analysis for Cluster 3...")
        print("\nLoading Z500 data for Cluster 3 analysis...")
        try:
            # Load Z500 data - adjust path as needed
            z500_file = 'mon_data_u_v_t_z_q_1000_500.nc'  # Your Z500 file path
            z500_data = xr.open_dataset(z500_file)

            # Now call the function with the loaded data
            cluster3_results = plot_cluster3_actual_z500(
                cluster_occurrences=cluster_occurrences,
                z500_data=z500_data,
                output_dir=output_dir
            )
            print(f"Cluster 3 Z500 analysis complete. {len(cluster3_results['dates'])} occurrences found.")

        except Exception as e:
            print(f"Error analyzing Cluster 3 Z500: {str(e)}")

        print("\nGenerating Z500 anomaly plots with t-test significance...")
        z500_ttest_fig = run_z500_ttest_analysis(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            z500_file='mon_data_u_v_t_z_q_1000_500.nc',
            output_dir=output_dir
        )

        # After clustering is complete
        wider_domain_projection = project_clusters_on_wider_domain(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            z500_file='combined_geopotential_500hPa_monthly.nc',
            output_dir=output_dir,
            lon_range=(-85, 45),  # Wide Atlantic-European domain
            lat_range=(0, 85)  # From equator to Arctic
        )

        enhanced_project_clusters_on_wider_domain(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            z500_file='combined_geopotential_500hPa_monthly.nc',  # Path to your Z500 data
            output_dir=output_dir,
            lon_range=(-85, 45),  # Adjust as needed for your wider domain
            lat_range=(0, 75)  # Adjust as needed for your wider domain
        )

        # After running the clustering
        print("\nCreating 2D projection of SOM patterns...")
        projection_file = create_som_pattern_2d_projection(
            som=som,
            clustering_results=clustering_results,
            output_dir=output_dir
        )

        #
        # Add this after your other analyses in analyze_hadisst_patterns function
        # For example, after the Z500 t-test analysis

        print("\nCreating cluster indices for applications...")
        cluster_indices = create_cluster_indices_for_applications(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            output_dir=output_dir,
            start_year=1950,
            end_year=2024
        )

        print("\nAnalyze cluster z500 lead lag with indices...")
        cluster_indices_file = os.path.join(output_dir, 'cluster_indices_for_applications',
                                            'cluster_indices_for_applications.csv')
        lag_output_dir = os.path.join(output_dir, 'lag_analysis_results')
        lag_results = analyze_cluster_z500_lead_lag_with_indices(
            cluster_indices_file=cluster_indices_file,
            z500_file='combined_geopotential_500hPa_monthly.nc',
            output_dir=lag_output_dir,
            max_lag_days=180,
            lag_step=5
        )

        print("\nApplying cluster indices to European meteorological data...")
        europe_multivar_results = apply_indices_to_europe_multivar(
            som=som,
            X=X,
            ds=ds,
            clustering_results=clustering_results,
            data_file='data_z500_ci_z_q_cr_1980_2024.nc',  # Update with your actual file path
            output_dir=output_dir,
            variables=['z', 'q', 'crwc', 'ciwc']  # Include all variables you want to analyze
        )
        print(f"European multivariate analysis completed. Results available in {output_dir}/europe_multivar_analysis/")

        print(f"Analysis complete. Results saved in: {output_dir}")
        return ds, som, output_dir, stats, variance_results

        # After your other analyses


    except Exception as e:
        print(f"Detailed error during analysis:")
        traceback.print_exc()
        raise


#######################################################################################################################
#######################################################################################################################

if __name__ == "__main__":
    file_path = "HadISST_sst.nc"
    try:
        ds, som, output_dir, stats, variance_results = analyze_hadisst_patterns(file_path)
        print(f"Analysis completed successfully. Results are in: {output_dir}")
        print("\nQuality Metrics Summary:")
        print(f"Weighted Explained Variance: {variance_results['total_explained_variance_ratio'] * 100:.2f}%")
        print(f"Quantization Error: {variance_results['quality_metrics']['quantization_error']:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")