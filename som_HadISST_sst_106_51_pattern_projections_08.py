#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project SOM Patterns: A standalone program to project pre-trained SOM patterns
and clusters onto HadISST sea surface temperature data.
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import traceback
from scipy import spatial


def load_som_model(model_path):
    """Load a trained SOM model from disk"""
    print(f"Loading SOM model from: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            som = pickle.load(f)
        print(f"SOM model loaded. Grid shape: {som.get_weights().shape[:2]}")
        return som
    except Exception as e:
        print(f"Error loading SOM model: {str(e)}")
        raise


def load_clustering_results(clustering_path):
    """Load clustering results if available"""
    if os.path.exists(clustering_path):
        try:
            with open(clustering_path, 'rb') as f:
                clustering = pickle.load(f)
            print(f"Loaded clustering with {clustering.get('n_clusters', 0)} clusters")
            return clustering
        except Exception as e:
            print(f"Error loading clustering results: {str(e)}")
            return None
    else:
        print(f"No clustering results found at {clustering_path}")
        return None


def project_patterns_on_sst(som, sst_file, output_dir, clustering=None):
    """
    Project SOM patterns onto all months of SST data

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    sst_file : str
        Path to SST data file (NetCDF)
    output_dir : str
        Directory to save results
    clustering : dict, optional
        Clustering results to also map clusters
    """
    print("\n" + "=" * 30 + " PROJECTING SOM PATTERNS " + "=" * 30)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load SST data
    print(f"Loading SST data from: {sst_file}")
    ds = xr.open_dataset(sst_file)

    # Print dataset info
    print(f"Dataset variables: {list(ds.data_vars)}")
    print(f"Dataset dimensions: {list(ds.dims)}")
    print(f"Dataset coordinates: {list(ds.coords)}")
    print(f"SST variable shape: {ds.sst.shape}")

    # Get SOM input size
    som_weights = som.get_weights()
    som_rows, som_cols, n_features = som_weights.shape
    print(f"SOM weights shape: {som_weights.shape}")

    # Select time range from 1950 onwards
    ds = ds.sel(time=slice('1950-01-01', None))
    print(f"Time points after selection: {len(ds.time)}")

    # Initialize pattern to cluster mapping
    pattern_to_cluster = {}
    if clustering and 'pattern_clusters' in clustering:
        pattern_to_cluster = clustering['pattern_clusters']
        print(f"Using clustering with {len(set(pattern_to_cluster.values()))} clusters")

    # Select exactly the same spatial domain
    try:
        # Find indices corresponding to the lat/lon ranges
        lat_indices = np.where((ds.latitude >= 0) & (ds.latitude <= 60))[0]

        # Adjust longitude range based on your dataset's convention
        lon_indices = np.where((ds.longitude >= -80) & (ds.longitude <= -5))[0]
        # If longitude is 0 to 360, adjust:
        if len(lon_indices) == 0 and ds.longitude.max() > 180:
            lon_indices = np.where(
                ((ds.longitude >= 280) & (ds.longitude <= 360)) |
                ((ds.longitude >= 0) & (ds.longitude <= 5))
            )[0]

        # Select using indices instead of values
        ds = ds.isel(
            latitude=slice(min(lat_indices), max(lat_indices) + 1),
            longitude=slice(min(lon_indices), max(lon_indices) + 1)
        )

        print(f"Selected region shape: {ds.sst.shape}")
        print(f"Selected latitude range: {ds.latitude.values.min()} to {ds.latitude.values.max()}")
        print(f"Selected longitude range: {ds.longitude.values.min()} to {ds.longitude.values.max()}")

        # Verify the expected size matches the SOM features
        expected_features = ds.sst.isel(time=0).size
        print(f"Expected features: {expected_features}, SOM features: {n_features}")

        if expected_features != n_features:
            print(f"WARNING: Size mismatch! Reshaping may be required.")

        time_coords = pd.to_datetime(ds.time.values)
        all_months = np.unique([t.month for t in time_coords])

        print(f"Processing data for months: {sorted(all_months)}")

        # Initialize results storage
        all_results = []
        monthly_stats = {}

        # Process each month individually
        for month in sorted(all_months):
            print(f"\nProcessing month {month}...")

            # Select data for this month
            month_mask = np.array([t.month == month for t in time_coords])
            month_times = ds.time.values[month_mask]

            if len(month_times) == 0:
                print(f"No data for month {month}, skipping.")
                continue

            print(f"Found {len(month_times)} time points for month {month}")

            # Process each time point
            month_results = []

            for t_idx, t in enumerate(month_times):
                # Get SST data for this time point
                try:
                    sst_data = ds.sst.sel(time=t).values

                    # Check for all NaNs
                    if np.all(np.isnan(sst_data)):
                        print(f"Time point {t} contains all NaNs, skipping.")
                        continue

                    # Flatten and prepare data for SOM
                    sst_flat = sst_data.flatten()
                    nan_mask = np.isnan(sst_flat)

                    if np.all(nan_mask):
                        print(f"Time point {t} contains all NaNs after flattening, skipping.")
                        continue

                    # Replace NaNs with mean of valid data
                    valid_data = sst_flat[~nan_mask]
                    if len(valid_data) > 0:
                        data_mean = np.mean(valid_data)
                        sst_clean = np.copy(sst_flat)
                        sst_clean[nan_mask] = data_mean
                    else:
                        continue

                    # Find best matching pattern
                    bmu = som.winner(sst_clean)
                    pattern_num = bmu[0] * som_cols + bmu[1] + 1  # 1-based pattern index

                    # Get cluster if available (should be 1-indexed already)
                    cluster = pattern_to_cluster.get(pattern_num, -1) if pattern_to_cluster else -1

                    # Calculate quantization error
                    error = np.linalg.norm(sst_clean - som.get_weights()[bmu[0], bmu[1]])

                    # Calculate similarity distribution to all patterns
                    similarity_distribution = calculate_pattern_similarity_distribution(som, sst_clean)

                    # Calculate cluster similarity distribution
                    cluster_similarity_distribution = calculate_cluster_similarity_distribution(
                        similarity_distribution, pattern_to_cluster)

                    # Store result
                    result = {
                        'time': pd.to_datetime(t),
                        'year': pd.to_datetime(t).year,
                        'month': month,
                        'pattern': pattern_num,
                        'cluster': cluster,
                        'error': error,
                        'similarity_distribution': similarity_distribution,
                        'cluster_similarity_distribution': cluster_similarity_distribution
                    }
                    month_results.append(result)

                    # Progress indicator
                    if (t_idx + 1) % 20 == 0:
                        print(f"  Processed {t_idx + 1}/{len(month_times)} time points")

                except Exception as e:
                    print(f"Error processing time point {t}: {str(e)}")
                    continue

            # Skip month if no valid results
            if not month_results:
                print(f"No valid results for month {month}, skipping.")
                continue

            # Convert to DataFrame
            month_df = pd.DataFrame([{k: v for k, v in result.items() if k != 'similarity_distribution'}
                                     for result in month_results])

            # Save month results
            month_file = os.path.join(output_dir, f'month_{month}_projection.csv')
            month_df.to_csv(month_file, index=False)

            # Save similarity distributions
            dist_rows = []
            for result in month_results:
                time_val = result['time']
                year_val = result['year']
                month_val = result['month']
                for pattern, similarity in result['similarity_distribution'].items():
                    dist_rows.append({
                        'time': time_val,
                        'year': year_val,
                        'month': month_val,
                        'pattern': pattern,
                        'similarity_percentage': similarity
                    })

            dist_df = pd.DataFrame(dist_rows)
            dist_file = os.path.join(output_dir, f'month_{month}_similarity_distribution.csv')
            dist_df.to_csv(dist_file, index=False)

            print(f"Saved {len(dist_df)} similarity scores for month {month} to {dist_file}")

            # Save cluster similarity distributions to a separate file
            cluster_dist_rows = []
            for result in month_results:
                time_val = result['time']
                year_val = result['year']
                month_val = result['month']
                for cluster, similarity in result['cluster_similarity_distribution'].items():
                    cluster_dist_rows.append({
                        'time': time_val,
                        'year': year_val,
                        'month': month_val,
                        'cluster': cluster,
                        'similarity_percentage': similarity
                    })

            cluster_dist_df = pd.DataFrame(cluster_dist_rows)
            cluster_dist_file = os.path.join(output_dir, f'month_{month}_cluster_similarity_distribution.csv')
            cluster_dist_df.to_csv(cluster_dist_file, index=False)
            print(f"Saved {len(cluster_dist_df)} cluster similarity scores for month {month} to {cluster_dist_file}")

            print(f"Saved {len(month_df)} projections for month {month} to {month_file}")

            # Calculate pattern frequencies
            pattern_counts = month_df['pattern'].value_counts().to_dict()
            pattern_freqs = {int(p): count / len(month_df) for p, count in pattern_counts.items()}

            # Calculate cluster frequencies if available
            cluster_freqs = {}
            if 'cluster' in month_df.columns and month_df['cluster'].nunique() > 1:
                cluster_counts = month_df['cluster'].value_counts().to_dict()
                cluster_freqs = {int(c): count / len(month_df) for c, count in cluster_counts.items()}

            # Store monthly stats
            monthly_stats[month] = {
                'sample_count': len(month_df),
                'mean_error': float(month_df['error'].mean()),
                'median_error': float(month_df['error'].median()),
                'pattern_frequencies': pattern_freqs,
                'cluster_frequencies': cluster_freqs
            }

            # Append to all results
            all_results.append(month_df)

        # Combine all results if any
        if all_results:
            all_df = pd.concat(all_results)
            all_file = os.path.join(output_dir, 'all_projections.csv')
            all_df.to_csv(all_file, index=False)
            print(f"\nSaved combined results ({len(all_df)} projections) to {all_file}")

            if pattern_to_cluster:
                import json
                mapping_file = os.path.join(output_dir, 'cluster_pattern_mapping.json')
                with open(mapping_file, 'w') as f:
                    json.dump({str(k): v for k, v in pattern_to_cluster.items()}, f)

            # Create summary visualizations
            create_projection_visualizations(all_df, monthly_stats, output_dir, som_rows, som_cols)

            return all_df, monthly_stats
        else:
            print("No valid projection results found.")
            return None, None
    except Exception as e:
        print(f"Error selecting spatial domain: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_monthly_similarity_distributions(output_dir):
    """
    Analyze monthly similarity distributions across all time points

    Parameters:
    -----------
    output_dir : str
        Directory containing similarity distribution files
    """
    # Create output directory
    analysis_dir = os.path.join(output_dir, 'similarity_distribution_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    # Read all distribution files
    all_distributions = []
    for month in range(1, 13):
        dist_file = os.path.join(output_dir, f'month_{month}_similarity_distribution.csv')
        if os.path.exists(dist_file):
            month_dist = pd.read_csv(dist_file)
            all_distributions.append(month_dist)

    if not all_distributions:
        print("No similarity distribution files found.")
        return

    # Combine all distributions
    combined_dist = pd.concat(all_distributions)

    # Calculate average similarity percentage by month and pattern
    monthly_avg_dist = combined_dist.groupby(['month', 'pattern'])['similarity_percentage'].mean().reset_index()

    # Save to CSV
    monthly_avg_dist.to_csv(os.path.join(analysis_dir, 'monthly_avg_similarity_distribution.csv'), index=False)

    # Create visualization
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Get unique patterns
    patterns = sorted(monthly_avg_dist['pattern'].unique())

    # Create a heatmap of pattern similarity by month
    pivot_data = monthly_avg_dist.pivot(index='month', columns='pattern', values='similarity_percentage')

    plt.figure(figsize=(14, 8))
    plt.imshow(pivot_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Similarity Percentage')
    plt.xlabel('Pattern')
    plt.ylabel('Month')
    plt.title('Average Pattern Similarity Distribution by Month')
    plt.xticks(np.arange(len(patterns)), patterns)
    plt.yticks(np.arange(12), month_names)
    plt.savefig(os.path.join(analysis_dir, 'monthly_pattern_similarity_heatmap.png'), dpi=300)
    plt.close()

    print(f"Monthly similarity distribution analysis saved to: {analysis_dir}")

    return monthly_avg_dist


def calculate_pattern_similarity_distribution(som, sst_clean):
    """
    Calculate similarity scores to all patterns for a single time point
    """
    weights = som.get_weights()
    som_rows, som_cols = weights.shape[:2]

    # Calculate distance to all neurons/patterns
    distances = {}
    for i in range(som_rows):
        for j in range(som_cols):
            pattern_idx = i * som_cols + j + 1  # 1-based pattern index
            pattern_weights = weights[i, j]
            # Calculate Euclidean distance
            dist = np.linalg.norm(sst_clean - pattern_weights)
            distances[pattern_idx] = dist

    # Convert distances to similarities (inverse of distance)
    max_dist = max(distances.values())
    similarities = {p: 1 - (d / max_dist) for p, d in distances.items()}

    # Normalize to get percentages
    total_similarity = sum(similarities.values())
    similarity_percentages = {p: (s / total_similarity) * 100 for p, s in similarities.items()}

    return similarity_percentages

def calculate_cluster_similarity_distribution(similarity_distribution, pattern_to_cluster):
    """
    Calculate similarity scores for clusters based on pattern similarity distribution
    """
    # Initialize cluster similarities
    cluster_similarities = {}

    # Sum pattern similarities by cluster
    for pattern, similarity in similarity_distribution.items():
        cluster = pattern_to_cluster.get(pattern, -1)  # -1 for unassigned patterns
        if cluster not in cluster_similarities:
            cluster_similarities[cluster] = 0
        cluster_similarities[cluster] += similarity

    return cluster_similarities


def visualize_similarity_timeseries(output_dir):
    """
    Create time series visualizations of pattern and cluster similarities

    Parameters:
    -----------
    output_dir : str
        Directory containing similarity distribution files
    """
    # Create output directory for time series
    timeseries_dir = os.path.join(output_dir, 'similarity_timeseries')
    os.makedirs(timeseries_dir, exist_ok=True)

    # Load all similarity distribution files
    all_distributions = []
    for month in range(1, 13):
        dist_file = os.path.join(output_dir, f'month_{month}_similarity_distribution.csv')
        if os.path.exists(dist_file):
            month_dist = pd.read_csv(dist_file)
            all_distributions.append(month_dist)

    if not all_distributions:
        print("No similarity distribution files found.")
        return

    # Combine all distributions
    combined_dist = pd.concat(all_distributions)

    # Get unique years and patterns
    years = sorted(combined_dist['year'].unique())
    patterns = sorted(combined_dist['pattern'].unique())

    # Plot similarity time series for top patterns
    # First, identify top patterns by average similarity
    avg_by_pattern = combined_dist.groupby('pattern')['similarity_percentage'].mean()
    top_patterns = avg_by_pattern.nlargest(5).index

    # Create time series for each pattern
    for pattern in top_patterns:
        # Get data for this pattern across all time points
        pattern_data = combined_dist[combined_dist['pattern'] == pattern]

        # Calculate yearly averages
        yearly_avg = pattern_data.groupby('year')['similarity_percentage'].mean()

        # Plot time series
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
        plt.title(f'Pattern {pattern} Similarity Percentage (1950-present)')
        plt.xlabel('Year')
        plt.ylabel('Average Similarity (%)')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(timeseries_dir, f'pattern_{pattern}_timeseries.png'), dpi=300)
        plt.close()

    # Load cluster mapping if available
    cluster_mapping_file = os.path.join(output_dir, 'cluster_pattern_mapping.json')
    try:
        import json
        with open(cluster_mapping_file, 'r') as f:
            pattern_to_cluster = json.load(f)

        # Convert keys from strings to integers if needed
        pattern_to_cluster = {int(k): v for k, v in pattern_to_cluster.items()}

        # Add cluster information to the distribution data
        combined_dist['cluster'] = combined_dist['pattern'].map(
            lambda p: pattern_to_cluster.get(p, -1))

        # Calculate cluster similarity by summing pattern similarities within each cluster
        cluster_dist = combined_dist.groupby(['year', 'month', 'cluster'])['similarity_percentage'].sum().reset_index()

        # Get unique clusters
        clusters = sorted(cluster_dist['cluster'].unique())

        # Plot cluster similarity time series
        for cluster in clusters:
            if cluster < 0:  # Skip unassigned patterns
                continue

            # Get data for this cluster across all time points
            cluster_data = cluster_dist[cluster_dist['cluster'] == cluster]

            # Calculate yearly averages
            yearly_avg = cluster_data.groupby('year')['similarity_percentage'].mean()

            # Plot time series
            plt.figure(figsize=(12, 6))
            plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
            plt.title(f'Cluster {cluster} Similarity Percentage (1950-present)')
            plt.xlabel('Year')
            plt.ylabel('Average Similarity (%)')
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(timeseries_dir, f'cluster_{cluster}_timeseries.png'), dpi=300)
            plt.close()

        # Create a seasonal cycle visualization for each cluster
        for cluster in clusters:
            if cluster < 0:  # Skip unassigned patterns
                continue

            # Get data for this cluster
            cluster_data = cluster_dist[cluster_dist['cluster'] == cluster]

            # Calculate monthly averages across all years
            monthly_avg = cluster_data.groupby('month')['similarity_percentage'].mean()

            # Plot seasonal cycle
            plt.figure(figsize=(10, 6))
            plt.bar(monthly_avg.index, monthly_avg.values)
            plt.title(f'Cluster {cluster} Seasonal Cycle')
            plt.xlabel('Month')
            plt.ylabel('Average Similarity (%)')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(timeseries_dir, f'cluster_{cluster}_seasonal_cycle.png'), dpi=300)
            plt.close()

    except (FileNotFoundError, json.JSONDecodeError):
        print("Cluster mapping not available, skipping cluster visualizations")

    # Create seasonal evolution visualization
    # Calculate monthly averages for top patterns
    monthly_pattern_avg = combined_dist[combined_dist['pattern'].isin(top_patterns)].groupby(
        ['month', 'pattern'])['similarity_percentage'].mean().reset_index()

    # Plot seasonal evolution
    plt.figure(figsize=(12, 8))
    for pattern in top_patterns:
        pattern_data = monthly_pattern_avg[monthly_pattern_avg['pattern'] == pattern]
        plt.plot(pattern_data['month'], pattern_data['similarity_percentage'],
                 marker='o', linewidth=2, label=f'Pattern {pattern}')

    plt.title('Seasonal Evolution of Top Pattern Similarities')
    plt.xlabel('Month')
    plt.ylabel('Average Similarity (%)')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(timeseries_dir, 'seasonal_pattern_evolution.png'), dpi=300)
    plt.close()

    # Save pattern-to-cluster mapping if not already saved
    if 'pattern_to_cluster' in locals() and not os.path.exists(cluster_mapping_file):
        with open(cluster_mapping_file, 'w') as f:
            json.dump(pattern_to_cluster, f)

    print(f"Time series visualizations saved to: {timeseries_dir}")

def create_projection_visualizations(all_df, monthly_stats, output_dir, som_rows, som_cols):
    """Create visualizations of the projection results"""
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    try:
        # 1. Pattern frequency by month
        plt.figure(figsize=(12, 8))

        # Get top 5 patterns across all months
        top_patterns = all_df['pattern'].value_counts().nlargest(5).index.tolist()

        # Prepare data for plotting
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = sorted(monthly_stats.keys())

        # Create a separate line for each top pattern
        for pattern in top_patterns:
            pattern_freqs = []
            for month in months:
                freq = monthly_stats[month]['pattern_frequencies'].get(int(pattern), 0)
                pattern_freqs.append(freq)

            plt.plot(months, pattern_freqs, marker='o', linewidth=2, label=f'Pattern {pattern}')

        plt.xticks(months, [month_names[m - 1] for m in months])
        plt.xlabel('Month')
        plt.ylabel('Relative Frequency')
        plt.title('Top 5 Pattern Frequencies by Month')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'top_patterns_by_month.png'), dpi=300)
        plt.close()

        print(f"Created visualization plots in {viz_dir}")

        # 2. Heatmap of all pattern frequencies by month
        n_patterns = som_rows * som_cols
        pattern_freq_matrix = np.zeros((len(months), n_patterns))

        for i, month in enumerate(months):
            for pattern in range(1, n_patterns + 1):
                pattern_freq_matrix[i, pattern - 1] = monthly_stats[month]['pattern_frequencies'].get(pattern, 0)

        plt.figure(figsize=(14, 8))
        plt.imshow(pattern_freq_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='Relative Frequency')
        plt.xlabel('Pattern')
        plt.ylabel('Month')
        plt.xticks(np.arange(n_patterns), np.arange(1, n_patterns + 1))
        plt.yticks(np.arange(len(months)), [month_names[m - 1] for m in months])
        plt.title('Pattern Frequency Heatmap by Month')

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'pattern_frequency_heatmap.png'), dpi=300)
        plt.close()

        # 3. Projection error by month
        mean_errors = [monthly_stats[m]['mean_error'] for m in months]
        median_errors = [monthly_stats[m]['median_error'] for m in months]

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(months)) - 0.2, mean_errors, width=0.4, label='Mean Error')
        plt.bar(np.arange(len(months)) + 0.2, median_errors, width=0.4, label='Median Error')

        plt.xticks(np.arange(len(months)), [month_names[m - 1] for m in months])
        plt.xlabel('Month')
        plt.ylabel('Projection Error')
        plt.title('SOM Projection Error by Month')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'projection_error_by_month.png'), dpi=300)
        plt.close()

        # 4. Pattern frequency time series (yearly)
        if 'year' in all_df.columns:
            # Calculate yearly pattern frequencies
            years = sorted(all_df['year'].unique())

            # Group by year and pattern
            year_pattern_counts = all_df.groupby(['year', 'pattern']).size().unstack(fill_value=0)

            # Calculate relative frequencies
            year_pattern_freq = year_pattern_counts.div(year_pattern_counts.sum(axis=1), axis=0)

            # Plot top 5 patterns
            plt.figure(figsize=(12, 6))

            for pattern in top_patterns:
                if pattern in year_pattern_freq.columns:
                    plt.plot(years, year_pattern_freq[pattern],
                             marker='o', linewidth=2, label=f'Pattern {pattern}')

            plt.xlabel('Year')
            plt.ylabel('Relative Frequency')
            plt.title('Top Pattern Frequencies (1950-present)')
            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'pattern_frequency_time_series.png'), dpi=300)
            plt.close()

        print(f"Created visualization plots in {viz_dir}")

    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        traceback.print_exc()


def create_pattern_similarity_indices(all_df, output_dir, clustering=None):
    """
    Create time series indices based on pattern projections

    Parameters:
    -----------
    all_df : pandas.DataFrame
        DataFrame containing all projection results
    output_dir : str
        Directory to save indices
    clustering : dict, optional
        Clustering results for cluster-based indices
    """
    indices_dir = os.path.join(output_dir, 'indices')
    os.makedirs(indices_dir, exist_ok=True)

    print("\nCreating pattern similarity indices...")

    # Ensure we have all required columns
    required_columns = ['time', 'year', 'month', 'pattern', 'error']
    if not all(col in all_df.columns for col in required_columns):
        print("Error: Missing required columns in results dataframe")
        return

    # Sort by time for consistent time series
    all_df = all_df.sort_values('time')

    # Calculate yearly pattern frequencies (Pattern Frequency Index)
    yearly_pattern_freq = all_df.pivot_table(
        index='year',
        columns='pattern',
        values='time',
        aggfunc='count',
        fill_value=0
    )

    # Normalize to get frequencies
    yearly_pattern_freq = yearly_pattern_freq.div(yearly_pattern_freq.sum(axis=1), axis=0)

    # Create monthly pattern frequencies (Seasonal Pattern Index)
    monthly_pattern_freq = all_df.pivot_table(
        index=['year', 'month'],
        columns='pattern',
        values='time',
        aggfunc='count',
        fill_value=0
    )

    # Normalize to get frequencies
    monthly_pattern_freq = monthly_pattern_freq.div(monthly_pattern_freq.sum(axis=1), axis=0)

    # Calculate monthly average error (Pattern Similarity Index)
    monthly_error = all_df.groupby(['year', 'month'])['error'].mean().to_frame('similarity_index')
    # Convert to similarity (higher is more similar)
    max_error = monthly_error['similarity_index'].max()
    monthly_error['similarity_index'] = 1 - (monthly_error['similarity_index'] / max_error)

    # If clustering available, calculate cluster indices
    if 'cluster' in all_df.columns and clustering:
        # Create yearly cluster frequencies
        yearly_cluster_freq = all_df.pivot_table(
            index='year',
            columns='cluster',
            values='time',
            aggfunc='count',
            fill_value=0
        )

        # Normalize to get frequencies
        yearly_cluster_freq = yearly_cluster_freq.div(yearly_cluster_freq.sum(axis=1), axis=0)

        # Create monthly cluster frequencies
        monthly_cluster_freq = all_df.pivot_table(
            index=['year', 'month'],
            columns='cluster',
            values='time',
            aggfunc='count',
            fill_value=0
        )

        # Normalize to get frequencies
        monthly_cluster_freq = monthly_cluster_freq.div(monthly_cluster_freq.sum(axis=1), axis=0)

        # Save cluster indices
        yearly_cluster_freq.to_csv(os.path.join(indices_dir, 'yearly_cluster_frequency_index.csv'))
        monthly_cluster_freq.to_csv(os.path.join(indices_dir, 'monthly_cluster_frequency_index.csv'))

    # Calculate additional indices

    # 1. Winter Pattern Index: how winter-like each month is based on DJF pattern frequencies
    # First identify winter pattern distribution
    winter_mask = all_df['month'].isin([12, 1, 2])
    winter_pattern_dist = all_df[winter_mask].groupby('pattern').size() / winter_mask.sum()

    # Create winter similarity index for each month
    monthly_winter_similarity = []
    for (year, month), group in all_df.groupby(['year', 'month']):
        # Get pattern distribution for this month
        month_pattern_dist = group.groupby('pattern').size() / len(group)

        # Calculate cosine similarity with winter pattern
        common_patterns = list(set(winter_pattern_dist.index) & set(month_pattern_dist.index))
        if common_patterns:
            winter_vec = np.array([winter_pattern_dist.get(p, 0) for p in common_patterns])
            month_vec = np.array([month_pattern_dist.get(p, 0) for p in common_patterns])
            similarity = 1 - spatial.distance.cosine(winter_vec, month_vec)
        else:
            similarity = 0

        monthly_winter_similarity.append({
            'year': year,
            'month': month,
            'winter_similarity': similarity
        })

    winter_similarity_df = pd.DataFrame(monthly_winter_similarity)

    # Save indices
    yearly_pattern_freq.to_csv(os.path.join(indices_dir, 'yearly_pattern_frequency_index.csv'))
    monthly_pattern_freq.to_csv(os.path.join(indices_dir, 'monthly_pattern_frequency_index.csv'))
    monthly_error.to_csv(os.path.join(indices_dir, 'monthly_similarity_index.csv'))
    winter_similarity_df.to_csv(os.path.join(indices_dir, 'monthly_winter_similarity_index.csv'))

    # Create visualizations of indices
    visualize_pattern_indices(
        yearly_pattern_freq,
        monthly_pattern_freq,
        monthly_error,
        winter_similarity_df,
        indices_dir
    )

    print(f"Pattern indices saved to: {indices_dir}")
    return {
        'yearly_pattern_freq': yearly_pattern_freq,
        'monthly_pattern_freq': monthly_pattern_freq,
        'similarity_index': monthly_error,
        'winter_similarity': winter_similarity_df
    }


def visualize_pattern_indices(yearly_pattern_freq, monthly_pattern_freq, similarity_index, winter_similarity,
                              output_dir):
    """Create visualizations of the pattern indices"""
    # Get top patterns
    top_patterns = yearly_pattern_freq.mean().nlargest(5).index

    # 1. Yearly pattern frequency time series
    plt.figure(figsize=(12, 6))
    for pattern in top_patterns:
        plt.plot(yearly_pattern_freq.index, yearly_pattern_freq[pattern],
                 marker='o', linewidth=2, label=f'Pattern {pattern}')

    plt.xlabel('Year')
    plt.ylabel('Relative Frequency')
    plt.title('Top Pattern Frequencies Over Time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'yearly_pattern_frequency.png'), dpi=300)
    plt.close()

    # 2. Monthly pattern similarity index (error)
    # Reshape to have months as columns
    similarity_pivot = similarity_index.reset_index().pivot(
        index='year', columns='month', values='similarity_index')

    # Calculate mean by month
    monthly_avg_similarity = similarity_pivot.mean()

    plt.figure(figsize=(10, 6))
    plt.bar(monthly_avg_similarity.index, monthly_avg_similarity.values)
    plt.xlabel('Month')
    plt.ylabel('Pattern Similarity Index')
    plt.title('Average Pattern Similarity by Month')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'monthly_pattern_similarity.png'), dpi=300)
    plt.close()

    # 3. Winter pattern similarity by month
    winter_pivot = winter_similarity.pivot_table(
        index='year', columns='month', values='winter_similarity')

    monthly_avg_winter_sim = winter_pivot.mean()

    plt.figure(figsize=(10, 6))
    plt.bar(monthly_avg_winter_sim.index, monthly_avg_winter_sim.values)
    plt.xlabel('Month')
    plt.ylabel('Similarity to Winter Pattern Distribution')
    plt.title('Average Similarity to Winter Pattern Distribution by Month')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'monthly_winter_similarity.png'), dpi=300)
    plt.close()

    # 4. Pattern frequency heatmap by month
    # Get average monthly pattern frequency across years
    monthly_avg_pattern = monthly_pattern_freq.groupby('month').mean()

    plt.figure(figsize=(14, 6))
    plt.imshow(monthly_avg_pattern.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Relative Frequency')
    plt.xlabel('Month')
    plt.ylabel('Pattern')
    plt.title('Average Pattern Frequency by Month')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.yticks(range(len(monthly_avg_pattern.columns)),
               [f'Pattern {p}' for p in monthly_avg_pattern.columns])
    plt.savefig(os.path.join(output_dir, 'monthly_pattern_heatmap.png'), dpi=300)
    plt.close()



def transform_cluster_similarity_to_spreadsheet(similarity_dir, cluster_id, output_file):
    """
    Transform cluster similarity distribution data to a spreadsheet format
    with years as rows and months as columns

    Parameters:
    -----------
    similarity_dir : str
        Directory containing cluster similarity distribution files
    cluster_id : int
        The cluster ID to extract data for
    output_file : str
        Path to save the CSV output file
    """

    print(f"Transforming data for cluster {cluster_id}...")

    # Load all monthly similarity files
    all_data = []
    for month in range(1, 13):
        file_path = os.path.join(similarity_dir, f'month_{month}_cluster_similarity_distribution.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Filter to just the specified cluster
            df = df[df['cluster'] == cluster_id]
            all_data.append(df)
        else:
            print(f"Warning: File not found: {file_path}")

    if not all_data:
        print("No data found for any month!")
        return

    # Combine all data
    combined_df = pd.concat(all_data)

    # Pivot to get years as rows and months as columns
    pivot_df = combined_df.pivot_table(
        index='year',
        columns='month',
        values='similarity_percentage',
        aggfunc='mean'  # In case there are multiple entries per year-month
    )

    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_df.columns = [month_names[i - 1] for i in pivot_df.columns]

    # Reset index to make 'year' a regular column
    pivot_df = pivot_df.reset_index()

    # Save to CSV
    pivot_df.to_csv(output_file, index=False)
    print(f"Saved spreadsheet format to: {output_file}")

    # Also create an Excel file if openpyxl is available
    try:
        excel_file = output_file.replace('.csv', '.xlsx')
        pivot_df.to_excel(excel_file, index=False)
        print(f"Saved Excel format to: {excel_file}")
    except:
        print("Could not create Excel file - openpyxl may not be installed")

    return pivot_df


def process_all_clusters(similarity_dir, output_dir):
    """Process all clusters and save as spreadsheets"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of available clusters
    # Read first file to identify clusters
    first_file = os.path.join(similarity_dir, 'month_1_cluster_similarity_distribution.csv')
    if not os.path.exists(first_file):
        print("No cluster data found!")
        return

    df = pd.read_csv(first_file)
    clusters = df['cluster'].unique()

    print(f"Found {len(clusters)} clusters: {clusters}")

    # Process each cluster
    for cluster in clusters:
        output_file = os.path.join(output_dir, f'cluster_{cluster}_monthly_similarity.csv')
        transform_cluster_similarity_to_spreadsheet(similarity_dir, cluster, output_file)

    print("All clusters processed!")

def calculate_cluster_month_distances(output_dir, clustering):
    """
    Calculate distances between clusters and monthly patterns using pattern assignments

    Parameters:
    -----------
    output_dir : str
        Directory containing projection results
    clustering : dict
        Clustering results containing pattern clusters
    """
    print("\nCalculating distances between clusters and monthly patterns...")

    # Create output directory for distance analysis
    distance_dir = os.path.join(output_dir, 'cluster_month_distances')
    os.makedirs(distance_dir, exist_ok=True)

    # Get pattern_clusters from clustering results
    if 'pattern_clusters' not in clustering:
        print("Error: No pattern_clusters in clustering results")
        return

    pattern_to_cluster = clustering['pattern_clusters']
    clusters = sorted(set(pattern_to_cluster.values()))

    print(f"Found {len(clusters)} clusters: {clusters}")

    # Load monthly pattern distributions
    monthly_patterns = {}

    for month in range(1, 13):
        dist_file = os.path.join(output_dir, f'month_{month}_similarity_distribution.csv')
        if os.path.exists(dist_file):
            dist_df = pd.read_csv(dist_file)
            # Group by pattern and calculate mean similarity
            avg_pattern = dist_df.groupby('pattern')['similarity_percentage'].mean().reset_index()
            monthly_patterns[month] = avg_pattern

    if not monthly_patterns:
        print("No monthly pattern data found.")
        return

    # Calculate distances
    distances = []

    for month, month_patterns in monthly_patterns.items():
        # Create cluster similarity vectors for this month
        cluster_similarities = {cluster: 0.0 for cluster in clusters}

        # Sum up similarities by cluster
        for _, row in month_patterns.iterrows():
            pattern = int(row['pattern'])
            similarity = row['similarity_percentage']

            # Get cluster for this pattern
            cluster = pattern_to_cluster.get(str(pattern), -1)
            if cluster >= 0:  # Only include patterns that are assigned to clusters
                if cluster not in cluster_similarities:
                    cluster_similarities[cluster] = 0
                cluster_similarities[cluster] += similarity

        # Calculate total similarity for normalization
        total_similarity = sum(cluster_similarities.values())
        if total_similarity > 0:
            # Normalize
            for cluster in cluster_similarities:
                cluster_similarities[cluster] /= total_similarity

        # Calculate distance as 1 - similarity (higher similarity = lower distance)
        for cluster, similarity in cluster_similarities.items():
            distance = 1.0 - similarity
            distances.append({
                'month': month,
                'cluster': cluster,
                'euclidean_distance': distance,
                'similarity': similarity
            })

    # Convert to DataFrame and save
    distances_df = pd.DataFrame(distances)
    distances_file = os.path.join(distance_dir, 'cluster_month_distances.csv')
    distances_df.to_csv(distances_file, index=False)

    # Create index values
    mean_dist = distances_df['euclidean_distance'].mean()
    std_dist = distances_df['euclidean_distance'].std()
    distances_df['index_value'] = (mean_dist - distances_df['euclidean_distance']) / std_dist

    # Create a heatmap visualization
    distances_pivot = distances_df.pivot(index='month', columns='cluster', values='euclidean_distance')

    plt.figure(figsize=(10, 8))
    plt.imshow(distances_pivot, cmap='viridis_r')  # viridis_r makes smaller distances darker
    plt.colorbar(label='Distance (1 - Similarity)')
    plt.title('Distances Between Clusters and Monthly Patterns')
    plt.xlabel('Cluster')
    plt.ylabel('Month')
    plt.xticks(range(len(clusters)), clusters)
    plt.yticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    plt.savefig(os.path.join(distance_dir, 'cluster_month_distances_heatmap.png'), dpi=300)
    plt.close()

    print(f"Cluster-month distances saved to {distances_file}")

    return distances_df


def plot_cluster_indices_timeseries(output_dir, distances_df):
    """
    Plot time series of cluster indices for all available years

    Parameters:
    -----------
    output_dir : str
        Directory to save output plots
    distances_df : pandas.DataFrame
        DataFrame containing cluster-month distances and indices
    """
    print("\nCreating time series plots of cluster indices...")

    # Create output directory
    timeseries_dir = os.path.join(output_dir, 'cluster_indices_timeseries')
    os.makedirs(timeseries_dir, exist_ok=True)

    # Load the all_projections.csv to get the full time range
    projections_file = os.path.join(output_dir, 'all_projections.csv')
    if not os.path.exists(projections_file):
        print(f"Error: Cannot find projections file at {projections_file}")
        return

    all_data = pd.read_csv(projections_file)

    # Get unique years and clusters
    years = sorted(all_data['year'].unique())
    clusters = sorted(distances_df['cluster'].unique())

    # Calculate indices for each year-month
    # First, combine the distances with the all_data
    month_indices = []

    for year in years:
        for month in range(1, 13):
            year_month_data = all_data[(all_data['year'] == year) & (all_data['month'] == month)]

            if len(year_month_data) == 0:
                continue

            # For each cluster, get the index value for this month
            for cluster in clusters:
                cluster_month_idx = distances_df[
                    (distances_df['month'] == month) &
                    (distances_df['cluster'] == cluster)
                    ]

                if len(cluster_month_idx) > 0:
                    # Get the index value (either euclidean_distance or index_value if available)
                    if 'index_value' in cluster_month_idx.columns:
                        index_val = cluster_month_idx['index_value'].values[0]
                    else:
                        # Use inverse of distance as a simple index
                        index_val = 1.0 / (cluster_month_idx['euclidean_distance'].values[0] + 0.0001)

                    month_indices.append({
                        'year': year,
                        'month': month,
                        'cluster': cluster,
                        'index_value': index_val
                    })

    # Convert to DataFrame
    indices_df = pd.DataFrame(month_indices)

    # Save the complete indices
    indices_file = os.path.join(timeseries_dir, 'year_month_cluster_indices.csv')
    indices_df.to_csv(indices_file, index=False)

    # Create annual averages
    annual_indices = indices_df.groupby(['year', 'cluster'])['index_value'].mean().reset_index()

    # Plot time series with all clusters
    plt.figure(figsize=(15, 8))
    for cluster in clusters:
        cluster_data = annual_indices[annual_indices['cluster'] == cluster]
        plt.plot(cluster_data['year'], cluster_data['index_value'],
                 marker='o', linewidth=2, label=f'Cluster {cluster}')

    plt.title('Annual Average Cluster Indices (1950-present)')
    plt.xlabel('Year')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(timeseries_dir, 'annual_cluster_indices.png'), dpi=300)
    plt.close()

    # Also create individual plots for each cluster
    for cluster in clusters:
        cluster_data = indices_df[indices_df['cluster'] == cluster]

        # Plot annual cycle
        plt.figure(figsize=(10, 6))
        monthly_avg = cluster_data.groupby('month')['index_value'].mean()
        plt.bar(monthly_avg.index, monthly_avg.values)
        plt.title(f'Cluster {cluster} Annual Cycle')
        plt.xlabel('Month')
        plt.ylabel('Average Index Value')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(timeseries_dir, f'cluster_{cluster}_annual_cycle.png'), dpi=300)
        plt.close()

        # Plot time series by season
        seasons = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }

        plt.figure(figsize=(15, 8))
        for season, months in seasons.items():
            season_data = cluster_data[cluster_data['month'].isin(months)]
            seasonal_avg = season_data.groupby('year')['index_value'].mean().reset_index()
            plt.plot(seasonal_avg['year'], seasonal_avg['index_value'],
                     marker='o', linewidth=2, label=season)

        plt.title(f'Cluster {cluster} Seasonal Indices (1950-present)')
        plt.xlabel('Year')
        plt.ylabel('Index Value')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(timeseries_dir, f'cluster_{cluster}_seasonal_indices.png'), dpi=300)
        plt.close()

    print(f"Time series plots saved to: {timeseries_dir}")
    return indices_df

########################################################################################################################
########################################################################################################################


def calculate_cluster_month_index(output_dir, clustering):
    """
    Calculate a normalized index of cluster-month similarity centered around zero

    Parameters:
    -----------
    output_dir : str
        Directory containing projection results
    clustering : dict
        Clustering results containing pattern clusters

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing monthly index values for each cluster
    """
    print("\nCalculating cluster-month indices...")

    # Create output directory
    index_dir = os.path.join(output_dir, 'cluster_indices')
    os.makedirs(index_dir, exist_ok=True)

    # Get pattern-to-cluster mapping
    if 'pattern_clusters' not in clustering:
        print("Error: No pattern_clusters in clustering results")
        return None

    pattern_to_cluster = clustering['pattern_clusters']
    clusters = sorted(set(pattern_to_cluster.values()))

    print(f"Found {len(clusters)} clusters: {clusters}")

    # Load monthly similarity distributions
    all_distributions = []
    for month in range(1, 13):
        dist_file = os.path.join(output_dir, f'month_{month}_cluster_similarity_distribution.csv')
        if os.path.exists(dist_file):
            df = pd.read_csv(dist_file)
            all_distributions.append(df)

    if not all_distributions:
        print("No distribution files found.")
        return None

    # Combine all distributions
    combined_df = pd.concat(all_distributions)

    # Calculate average similarity by month and cluster
    monthly_avg = combined_df.groupby(['month', 'cluster'])['similarity_percentage'].mean().reset_index()

    # Calculate overall average similarity for each cluster (across all months)
    cluster_avg = monthly_avg.groupby('cluster')['similarity_percentage'].mean().reset_index()
    cluster_avg = {row['cluster']: row['similarity_percentage'] for _, row in cluster_avg.iterrows()}

    # Calculate standard deviation for each cluster (across all months)
    cluster_std = monthly_avg.groupby('cluster')['similarity_percentage'].std().reset_index()
    cluster_std = {row['cluster']: row['similarity_percentage'] for _, row in cluster_std.iterrows()}

    # Calculate normalized index (centered around zero)
    monthly_avg['avg_similarity'] = monthly_avg.apply(
        lambda row: cluster_avg[row['cluster']], axis=1)
    monthly_avg['std_similarity'] = monthly_avg.apply(
        lambda row: cluster_std[row['cluster']], axis=1)

    # Z-score calculation: (value - mean) / std_dev
    monthly_avg['index_value'] = (monthly_avg['similarity_percentage'] - monthly_avg['avg_similarity']) / monthly_avg[
        'std_similarity']

    # Save to CSV
    monthly_avg.to_csv(os.path.join(index_dir, 'cluster_month_index.csv'), index=False)

    # Create heatmap visualization
    pivot_data = monthly_avg.pivot(index='month', columns='cluster', values='index_value')

    plt.figure(figsize=(10, 8))
    plt.imshow(pivot_data, aspect='auto', cmap='RdBu_r')  # Red-Blue diverging colormap
    plt.colorbar(label='Normalized Index (z-score)')
    plt.title('Cluster-Month Index Values')
    plt.xlabel('Cluster')
    plt.ylabel('Month')
    plt.xticks(range(len(clusters)), clusters)
    plt.yticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.savefig(os.path.join(index_dir, 'cluster_month_index_heatmap.png'), dpi=300)
    plt.close()

    # Create bar chart showing each cluster's monthly index
    plt.figure(figsize=(14, 10))

    # Create subplot for each cluster
    n_clusters = len(clusters)
    n_rows = (n_clusters + 1) // 2  # Ceiling division for number of rows

    for i, cluster in enumerate(clusters):
        plt.subplot(n_rows, 2, i + 1)

        cluster_data = monthly_avg[monthly_avg['cluster'] == cluster]
        colors = ['red' if x < 0 else 'blue' for x in cluster_data['index_value']]

        plt.bar(range(1, 13), cluster_data['index_value'], color=colors)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Cluster {cluster} Monthly Index')
        plt.xlabel('Month')
        plt.ylabel('Index Value')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(index_dir, 'cluster_monthly_indices.png'), dpi=300)
    plt.close()

    print(f"Cluster-month indices saved to: {index_dir}")

    return monthly_avg


########################################################################################################################
########################################################################################################################

def create_index_timeseries(output_dir, monthly_avg):
    """
    Create time series of the cluster-month index for all years

    Parameters:
    -----------
    output_dir : str
        Directory containing projection results
    monthly_avg : pandas.DataFrame
        DataFrame with monthly cluster indices
    """
    print("\nCreating index time series...")

    # Create output directory
    timeseries_dir = os.path.join(output_dir, 'index_timeseries')
    os.makedirs(timeseries_dir, exist_ok=True)

    # Load all projections to get years
    all_proj_file = os.path.join(output_dir, 'all_projections.csv')
    if not os.path.exists(all_proj_file):
        print(f"Error: Cannot find projections at {all_proj_file}")
        return

    all_df = pd.read_csv(all_proj_file)
    years = sorted(all_df['year'].unique())

    # Load all cluster similarity distributions
    all_distributions = []
    for month in range(1, 13):
        dist_file = os.path.join(output_dir, f'month_{month}_cluster_similarity_distribution.csv')
        if os.path.exists(dist_file):
            df = pd.read_csv(dist_file)
            all_distributions.append(df)

    if not all_distributions:
        print("No distribution files found.")
        return

    # Combine all distributions
    combined_df = pd.concat(all_distributions)

    # Get the clusters
    clusters = sorted(monthly_avg['cluster'].unique())

    # Group by year, month, cluster
    yearly_data = combined_df.groupby(['year', 'month', 'cluster'])['similarity_percentage'].mean().reset_index()

    # Add index value to each year-month-cluster combination
    indexed_data = []

    for _, row in yearly_data.iterrows():
        # Find the index value for this month-cluster
        index_row = monthly_avg[
            (monthly_avg['month'] == row['month']) &
            (monthly_avg['cluster'] == row['cluster'])
            ]

        if not index_row.empty:
            avg_similarity = index_row['avg_similarity'].values[0]
            std_similarity = index_row['std_similarity'].values[0]

            # Calculate index value for this specific year-month-cluster
            index_value = (row['similarity_percentage'] - avg_similarity) / std_similarity

            indexed_data.append({
                'year': row['year'],
                'month': row['month'],
                'cluster': row['cluster'],
                'similarity': row['similarity_percentage'],
                'index_value': index_value
            })

    # Convert to DataFrame
    index_df = pd.DataFrame(indexed_data)

    # Save full dataset
    index_df.to_csv(os.path.join(timeseries_dir, 'full_index_timeseries.csv'), index=False)

    # Plot time series for each cluster
    for cluster in clusters:
        cluster_data = index_df[index_df['cluster'] == cluster]

        # Annual average
        annual_avg = cluster_data.groupby('year')['index_value'].mean().reset_index()

        plt.figure(figsize=(14, 6))
        plt.plot(annual_avg['year'], annual_avg['index_value'], marker='o', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Cluster {cluster} Annual Index (1950-present)')
        plt.xlabel('Year')
        plt.ylabel('Index Value')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(timeseries_dir, f'cluster_{cluster}_annual_index.png'), dpi=300)
        plt.close()

        # Seasonal time series
        seasons = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }

        plt.figure(figsize=(14, 6))
        for season, months in seasons.items():
            season_data = cluster_data[cluster_data['month'].isin(months)]
            seasonal_avg = season_data.groupby('year')['index_value'].mean().reset_index()
            plt.plot(seasonal_avg['year'], seasonal_avg['index_value'], marker='o', linewidth=2, label=season)

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Cluster {cluster} Seasonal Index (1950-present)')
        plt.xlabel('Year')
        plt.ylabel('Index Value')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(timeseries_dir, f'cluster_{cluster}_seasonal_index.png'), dpi=300)
        plt.close()

    # Plot all clusters together
    # Annual average for all clusters
    plt.figure(figsize=(14, 6))
    for cluster in clusters:
        cluster_data = index_df[index_df['cluster'] == cluster]
        annual_avg = cluster_data.groupby('year')['index_value'].mean().reset_index()
        plt.plot(annual_avg['year'], annual_avg['index_value'], marker='o', linewidth=2, label=f'Cluster {cluster+1}')

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('All Clusters Annual Index (1950-present)')
    plt.xlabel('Year')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(timeseries_dir, 'all_clusters_annual_index.png'), dpi=300)
    plt.close()

    print(f"Index time series saved to: {timeseries_dir}")

    return index_df


########################################################################################################################
########################################################################################################################

def analyze_winter_indices(output_dir, index_df):
    """
    Analyze just the winter (DJF) indices from the cluster index data

    Parameters:
    -----------
    output_dir : str
        Directory to save output files
    index_df : pandas.DataFrame
        DataFrame containing the full index data
    """
    print("\nAnalyzing winter (DJF) indices...")

    # Create output directory
    winter_dir = os.path.join(output_dir, 'winter_indices')
    os.makedirs(winter_dir, exist_ok=True)

    # Select only winter months (December, January, February)
    winter_df = index_df[index_df['month'].isin([12, 1, 2])].copy()

    # Handle the December issue (December belongs to previous year's winter)
    for idx, row in winter_df.iterrows():
        if row['month'] == 12:
            winter_df.at[idx, 'winter_year'] = row['year'] + 1
        else:
            winter_df.at[idx, 'winter_year'] = row['year']

    # Calculate winter average for each cluster and year
    winter_avg = winter_df.groupby(['winter_year', 'cluster'])['index_value'].mean().reset_index()

    # Save the winter averages
    winter_avg.to_csv(os.path.join(winter_dir, 'winter_indices.csv'), index=False)

    # Plot time series for all clusters
    plt.figure(figsize=(15, 8))
    clusters = sorted(winter_avg['cluster'].unique())

    for cluster in clusters:
        cluster_data = winter_avg[winter_avg['cluster'] == cluster]
        plt.plot(cluster_data['winter_year'], cluster_data['index_value'],
                 marker='o', linewidth=2, label=f'Cluster {cluster}')

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Winter (DJF) Cluster Indices (1950-present)')
    plt.xlabel('Winter Year')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(winter_dir, 'all_clusters_winter_index.png'), dpi=300)
    plt.close()

    # Create individual plots for each cluster
    for cluster in clusters:
        cluster_data = winter_avg[winter_avg['cluster'] == cluster]

        plt.figure(figsize=(15, 8))
        plt.plot(cluster_data['winter_year'], cluster_data['index_value'],
                 marker='o', linewidth=2, color='blue')

        # Add trend line
        if len(cluster_data) > 1:
            z = np.polyfit(cluster_data['winter_year'], cluster_data['index_value'], 1)
            p = np.poly1d(z)
            plt.plot(cluster_data['winter_year'], p(cluster_data['winter_year']),
                     'r--', linewidth=1, label=f'Trend: {z[0]:.4f} per year')

        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Cluster {cluster} Winter (DJF) Index (1950-present)')
        plt.xlabel('Winter Year')
        plt.ylabel('Index Value')
        if len(cluster_data) > 1:
            plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(winter_dir, f'cluster_{cluster+1}_winter_index.png'), dpi=300)
        plt.close()

    # Create a composite plot with 5-year running mean
    plt.figure(figsize=(15, 8))

    for cluster in clusters:
        cluster_data = winter_avg[winter_avg['cluster'] == cluster]

        # Calculate 5-year running mean if enough data
        if len(cluster_data) >= 5:
            cluster_data = cluster_data.sort_values('winter_year')
            cluster_data['running_mean'] = cluster_data['index_value'].rolling(window=5, center=True).mean()

            plt.plot(cluster_data['winter_year'], cluster_data['running_mean'],
                     linewidth=3, label=f'Cluster {cluster} (5-yr mean)')

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Winter (DJF) Cluster Indices - 5-Year Running Mean (1950-present)')
    plt.xlabel('Winter Year')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(winter_dir, 'all_clusters_winter_index_running_mean.png'), dpi=300)
    plt.close()

    # Calculate monthly breakdown within winter
    monthly_winter = winter_df.groupby(['cluster', 'month'])['index_value'].mean().reset_index()

    # Create bar chart of monthly winter values
    plt.figure(figsize=(12, 8))

    month_labels = {12: 'Dec', 1: 'Jan', 2: 'Feb'}
    width = 0.2
    x = np.arange(3)

    for i, cluster in enumerate(clusters):
        cluster_data = monthly_winter[monthly_winter['cluster'] == cluster]
        # Sort by actual winter order: Dec, Jan, Feb
        cluster_data = cluster_data.set_index('month').reindex([12, 1, 2]).reset_index()

        plt.bar(x + (i - len(clusters) / 2 + 0.5) * width,
                cluster_data['index_value'],
                width,
                label=f'Cluster {cluster}')

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Average Index Value by Winter Month')
    plt.xlabel('Month')
    plt.xticks(x, ['Dec', 'Jan', 'Feb'])
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(winter_dir, 'winter_monthly_breakdown.png'), dpi=300)
    plt.close()

    # Calculate stats and correlations
    correlation_matrix = winter_avg.pivot(index='winter_year', columns='cluster', values='index_value').corr()
    correlation_matrix.to_csv(os.path.join(winter_dir, 'winter_cluster_correlations.csv'))

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation Between Cluster Winter Indices')
    plt.xticks(range(len(clusters)), [f'Cluster {c}' for c in clusters])
    plt.yticks(range(len(clusters)), [f'Cluster {c}' for c in clusters])

    # Add correlation values to the cells
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                     ha='center', va='center',
                     color='white' if abs(correlation_matrix.iloc[i, j]) > 0.5 else 'black')

    plt.savefig(os.path.join(winter_dir, 'winter_cluster_correlations.png'), dpi=300)
    plt.close()

    print(f"Winter index analysis saved to: {winter_dir}")

    return winter_avg



########################################################################################################################
########################################################################################################################


if __name__ == "__main__":
    # Define file paths
    sst_file = 'HadISST_sst_v2.nc'
    som_model_path = 'hadisst_som_analysis_20250317_144516/trained_som_model.pkl'
    clustering_path = 'hadisst_som_analysis_20250317_144516/pattern_clustering_results.pkl'

    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'som_projection_results_{timestamp}'

    print(f"Results will be saved to: {output_dir}")

    # Load SOM model
    som = load_som_model(som_model_path)

    # Load clustering results
    clustering = load_clustering_results(clustering_path)

    # Project patterns onto SST data and capture returned values
    all_df, monthly_stats = project_patterns_on_sst(som, sst_file, output_dir, clustering)

    monthly_indices = calculate_cluster_month_index(output_dir, clustering)

    # Generate time series of the index
    if monthly_indices is not None:
        index_timeseries = create_index_timeseries(output_dir, monthly_indices)


    # Now check if all_df is valid before proceeding
    if all_df is not None:
        # Import the spatial module needed for cosine distance
        indices = create_pattern_similarity_indices(all_df, output_dir, clustering)
        monthly_avg_dist = analyze_monthly_similarity_distributions(output_dir)
        visualize_similarity_timeseries(output_dir)

        cluster_month_distances = calculate_cluster_month_distances(output_dir, clustering)

        monthly_indices = calculate_cluster_month_index(output_dir, clustering)

        if monthly_indices is not None:
            index_timeseries = create_index_timeseries(output_dir, monthly_indices)

            # Add this line to analyze just the winter indices
            if index_timeseries is not None:
                winter_indices = analyze_winter_indices(output_dir, index_timeseries)


        spreadsheet_output_dir = os.path.join(output_dir, 'spreadsheets')
        process_all_clusters(output_dir, spreadsheet_output_dir)


