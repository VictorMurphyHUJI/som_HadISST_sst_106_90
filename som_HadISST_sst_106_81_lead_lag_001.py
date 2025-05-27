import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import traceback


def load_cluster_occurrences(file_path):
    """
    Load cluster occurrences from saved file
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def analyze_cluster_z500_lead_lag(cluster_occurrences, z500_file, output_dir, max_lag_days=180, lag_step=5):
    """
    Analyze lagged correlations between SST clusters and Z500 fields using daily Z500 data

    Parameters:
    -----------
    cluster_occurrences : dict
        Results from analyze_cluster_occurrences
    z500_file : str
        Path to daily Z500 data file
    output_dir : str
        Directory to save output
    max_lag_days : int
        Maximum lag to consider (in days)
    lag_step : int
        Step size for lags (in days)
    """
    print(f"Loading daily Z500 data from {z500_file} for lag analysis...")

    # Load Z500 data
    ds_z500 = xr.open_dataset(z500_file)

    # Extract time coordinates from cluster occurrences
    times = pd.to_datetime(cluster_occurrences['times'])
    cluster_sequence = cluster_occurrences['cluster_sequence']

    # Create DataFrame for clusters
    cluster_df = pd.DataFrame({
        'time': times,
        'year': times.dt.year,
        'month': times.dt.month,
        'day': times.dt.day,  # Add day information
        'cluster': cluster_sequence
    })

    # Filter to DJF months
    cluster_df = cluster_df[cluster_df['month'].isin([12, 1, 2])]

    # Set winter year (Dec belongs to next year's winter)
    cluster_df['winter_year'] = cluster_df.apply(
        lambda x: x['year'] + 1 if x['month'] == 12 else x['year'], axis=1)

    # Get unique clusters
    clusters = sorted(np.unique(cluster_sequence))
    n_clusters = len(clusters)

    # Prepare Z500 data
    # Select time dimension based on what's available
    time_dim = 'valid_time'  # Use the correct dimension name from your dataset
    z500_times = pd.to_datetime(ds_z500[time_dim].values)
    z500_df = pd.DataFrame({'time': z500_times})
    z500_df['year'] = z500_df['time'].dt.year
    z500_df['month'] = z500_df['time'].dt.month
    z500_df['day'] = z500_df['time'].dt.day

    # Filter to DJF
    z500_djf = ds_z500.sel({time_dim: ds_z500[time_dim].values[z500_df['month'].isin([12, 1, 2])]})

    # Create winter years for Z500
    z500_times_djf = pd.to_datetime(z500_djf[time_dim].values)
    z500_df_djf = pd.DataFrame({'time': z500_times_djf})
    z500_df_djf['year'] = z500_df_djf['time'].dt.year
    z500_df_djf['month'] = z500_df_djf['time'].dt.month
    z500_df_djf['day'] = z500_df_djf['time'].dt.day
    z500_df_djf['winter_year'] = z500_df_djf.apply(
        lambda x: x['year'] + 1 if x['month'] == 12 else x['year'], axis=1)

    # Update the Z500 dataset with winter_year
    z500_djf = z500_djf.assign_coords(winter_year=('valid_time', z500_df_djf['winter_year'].values))

    # Define regions for Z500 analysis
    # Check longitude range first
    lon_min, lon_max = float(ds_z500.longitude.min()), float(ds_z500.longitude.max())
    print(f"Z500 longitude range: {lon_min} to {lon_max}")

    # Define regions based on longitude range
    if lon_min >= 0 and lon_max <= 360:  # 0 to 360 range
        regions = {
            'north_atlantic': {'lat': slice(60, 30), 'lon': slice(280, 360)},
            'europe': {'lat': slice(65, 35), 'lon': slice(350, 40)},
            'arctic': {'lat': slice(90, 65), 'lon': slice(0, 360)},
            'mediterranean': {'lat': slice(45, 30), 'lon': slice(350, 40)}
        }
    else:  # -180 to 180 range
        regions = {
            'north_atlantic': {'lat': slice(60, 30), 'lon': slice(-80, 0)},
            'europe': {'lat': slice(65, 35), 'lon': slice(-10, 40)},
            'arctic': {'lat': slice(90, 65), 'lon': slice(-180, 180)},
            'mediterranean': {'lat': slice(45, 30), 'lon': slice(-10, 40)}
        }

    # Calculate regional Z500 indices
    z500_indices = {}
    for region_name, region_coords in regions.items():
        try:
            # Create safer selection by checking coordinates
            if 'latitude' in z500_djf.dims and 'longitude' in z500_djf.dims:
                # Calculate regional average
                regional_z500 = z500_djf.z.sel(
                    latitude=region_coords['lat'],
                    longitude=region_coords['lon']
                ).mean(dim=['latitude', 'longitude'])

                # Convert to DataFrame
                z500_values_df = pd.DataFrame({
                    'time': z500_times_djf,
                    'z500': regional_z500.values.flatten(),  # Flatten in case of remaining dimensions
                    'winter_year': z500_df_djf['winter_year'].values
                })

                # Create a clean time index
                z500_values_df = z500_values_df.set_index('time')
                z500_indices[region_name] = z500_values_df

                print(f"Created Z500 index for {region_name} with {len(z500_values_df)} data points")

            else:
                print(f"Warning: Dimension names may be different. Available dims: {list(z500_djf.dims)}")
                continue

        except Exception as e:
            print(f"Error calculating {region_name} Z500 index: {str(e)}")
            continue

    # Dictionary to store results
    lag_results = {}

    # Calculate lag range in days
    lags = range(-max_lag_days, max_lag_days + 1, lag_step)

    # For each cluster, calculate correlations with Z500 indices
    for cluster in clusters:
        lag_results[cluster] = {}

        # Create binary indicator for this cluster
        cluster_df[f'is_cluster_{cluster}'] = (cluster_df['cluster'] == cluster).astype(int)

        # Expand to daily values - for each row in cluster_df, create daily entries for that month
        daily_cluster_data = []

        for _, row in cluster_df.iterrows():
            # Get year and month
            year, month = row['year'], row['month']
            cluster_val = row[f'is_cluster_{cluster}']

            # Get all days for this month in Z500 data
            month_days = z500_df[(z500_df['year'] == year) & (z500_df['month'] == month)]

            # Add to daily data with the same cluster value
            for _, day_row in month_days.iterrows():
                daily_cluster_data.append({
                    'time': day_row['time'],
                    'value': cluster_val,
                    'winter_year': row['winter_year']
                })

        # Convert to DataFrame
        if daily_cluster_data:
            daily_cluster_df = pd.DataFrame(daily_cluster_data)
            daily_cluster_df = daily_cluster_df.set_index('time')

            # For each region
            for region_name, z500_df in z500_indices.items():
                print(f"Processing correlations for Cluster {cluster}, Region {region_name}...")

                # Initialize correlation results
                correlations = []
                p_values = []
                n_samples = []

                # For each lag
                for lag in lags:
                    try:
                        # Get lagged cluster series
                        cluster_series = daily_cluster_df['value']
                        z500_series = z500_df['z500']

                        # Shift according to lag (positive lag = cluster leads)
                        if lag > 0:
                            # Cluster leads Z500
                            cluster_series_aligned = cluster_series.copy()
                            z500_series_aligned = z500_series.shift(periods=lag)
                        else:
                            # Z500 leads cluster
                            cluster_series_aligned = cluster_series.shift(periods=-lag)
                            z500_series_aligned = z500_series.copy()

                        # Combine into a DataFrame and drop NaNs
                        combined = pd.DataFrame({
                            'cluster': cluster_series_aligned,
                            'z500': z500_series_aligned
                        }).dropna()

                        # Calculate correlation if we have enough data
                        if len(combined) >= 30:  # At least 30 days for statistical significance
                            r, p = stats.pearsonr(combined['cluster'], combined['z500'])
                            correlations.append(r)
                            p_values.append(p)
                            n_samples.append(len(combined))
                        else:
                            # Not enough data
                            correlations.append(np.nan)
                            p_values.append(np.nan)
                            n_samples.append(0)

                    except Exception as e:
                        print(f"  Error at lag {lag}: {str(e)}")
                        correlations.append(np.nan)
                        p_values.append(np.nan)
                        n_samples.append(0)

                # Store results
                lag_results[cluster][region_name] = {
                    'lags': list(lags),
                    'correlations': correlations,
                    'p_values': p_values,
                    'n_samples': n_samples
                }

    # Save results
    with open(os.path.join(output_dir, 'z500_lead_lag_results.pkl'), 'wb') as f:
        pickle.dump(lag_results, f)

    return lag_results


def create_lead_lag_plot(lags, correlations, p_values, n_samples, cluster, region_name, output_dir):
    """Create a single lead-lag correlation plot for daily data"""
    plt.figure(figsize=(10, 6))

    # Convert lists to numpy arrays with proper handling of possible errors
    lags = np.array(lags)

    # Handle possible mixed types in correlations
    correlations_array = np.zeros(len(correlations))
    for i, val in enumerate(correlations):
        try:
            correlations_array[i] = float(val)
        except (ValueError, TypeError):
            correlations_array[i] = np.nan
    correlations = correlations_array

    # Handle possible mixed types in p_values
    p_values_array = np.zeros(len(p_values))
    for i, val in enumerate(p_values):
        try:
            p_values_array[i] = float(val)
        except (ValueError, TypeError):
            p_values_array[i] = np.nan
    p_values = p_values_array

    # Calculate significance threshold
    valid_samples = [n for n in n_samples if n > 0]
    avg_n = np.mean(valid_samples) if valid_samples else 0
    sig_threshold = 2 / np.sqrt(avg_n) if avg_n > 0 else 0.5

    # Plot the correlation line
    valid_mask = ~np.isnan(correlations)
    plt.plot(lags[valid_mask], correlations[valid_mask], 'k-', linewidth=1.5)

    # Highlight significant correlations
    significant = p_values < 0.05
    plot_mask = significant & valid_mask
    if np.any(plot_mask):
        plt.plot(lags[plot_mask], correlations[plot_mask], 'ro', markersize=6)

    # Add reference lines
    plt.axhline(y=0, color='k', linestyle=':')
    plt.axvline(x=0, color='k', linestyle=':')
    plt.axhline(y=sig_threshold, color='b', linestyle='--', alpha=0.3)
    plt.axhline(y=-sig_threshold, color='b', linestyle='--', alpha=0.3)

    # Add peak annotation if there are valid correlations
    if np.any(valid_mask):
        # Find peak correlation (largest absolute value)
        abs_corr = np.abs(correlations[valid_mask])
        max_idx = np.argmax(abs_corr)
        max_corr = correlations[valid_mask][max_idx]
        max_lag = lags[valid_mask][max_idx]

        plt.annotate(f'Peak: r={max_corr:.2f} at lag={max_lag} days',
                     xy=(max_lag, max_corr),
                     xytext=(max_lag + 5, max_corr + 0.1 * np.sign(max_corr)),
                     arrowprops=dict(arrowstyle="->", color='blue'),
                     fontsize=10)

    # Add shading for lead vs lag
    plt.axvspan(0.5, max(lags), alpha=0.1, color='blue', label='SST leads Z500')
    plt.axvspan(min(lags), -0.5, alpha=0.1, color='red', label='Z500 leads SST')

    # Format region name for title
    region_title = ' '.join(word.capitalize() for word in region_name.split('_'))
    plt.title(f'Lead-Lag Correlation: Cluster {cluster} vs {region_title} Z500')

    # Add information about sample sizes
    plt.figtext(0.5, 0.01,
                f"Negative lags: Z500 leads SST cluster, Positive lags: SST cluster leads Z500\n"
                f"Average sample size: {avg_n:.1f} days",
                ha='center', fontsize=9)

    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Lag (days)')
    plt.ylabel('Correlation Coefficient')

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cluster{cluster}_{region_name}_lead_lag_daily.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def create_combined_lead_lag_plots(lag_results, clusters, regions, output_dir):
    """Create combined lead-lag plots showing all clusters for each region"""
    for region in regions:
        plt.figure(figsize=(12, 8))

        # Format region name for title
        region_title = ' '.join(word.capitalize() for word in region.split('_'))

        # Plot variables
        max_lag_val = None
        min_lag_val = None
        has_data = False

        # Plot each cluster
        for cluster in clusters:
            if region in lag_results[cluster]:
                result = lag_results[cluster][region]

                # Skip if no results available
                if 'lags' not in result or 'correlations' not in result:
                    continue

                current_lags = np.array(result['lags'])
                correlations = np.array(result['correlations'])
                p_values = np.array(result['p_values'])

                # Update lag range for shading - only if we have valid data
                valid_mask = ~np.isnan(correlations)
                if np.any(valid_mask):
                    has_data = True
                    if max_lag_val is None:
                        max_lag_val = np.max(current_lags[valid_mask])
                        min_lag_val = np.min(current_lags[valid_mask])
                    else:
                        max_lag_val = max(max_lag_val, np.max(current_lags[valid_mask]))
                        min_lag_val = min(min_lag_val, np.min(current_lags[valid_mask]))

                # Plot the correlation line
                plt.plot(current_lags, correlations, linewidth=2, label=f'Cluster {cluster}')

                # Add markers for significant points
                significant = np.array(p_values) < 0.05
                plot_mask = significant & valid_mask
                if np.any(plot_mask):
                    plt.scatter(current_lags[plot_mask], correlations[plot_mask], s=50, marker='o')

        # Add reference lines and shading only if we have data
        if has_data and max_lag_val is not None and min_lag_val is not None:
            # Add reference lines
            plt.axhline(y=0, color='k', linestyle=':')
            plt.axvline(x=0, color='k', linestyle=':')

            # Shade the lead/lag regions - safely using max_lag_val, min_lag_val
            plt.axvspan(0.5, max_lag_val, alpha=0.1, color='blue', label='SST leads Z500')
            plt.axvspan(min_lag_val, -0.5, alpha=0.1, color='red', label='Z500 leads SST')

            # Add labels and title
            plt.xlabel('Lag (months)')
            plt.ylabel('Correlation Coefficient')
            plt.title(f'Lead-Lag Correlations: All Clusters vs {region_title} Z500')

            # Add informative text
            plt.figtext(0.5, 0.01,
                        "Negative lags: Z500 leads SST clusters, Positive lags: SST clusters lead Z500",
                        ha='center', fontsize=9)

            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)

            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{region}_all_clusters_lead_lag.png'),
                        dpi=300, bbox_inches='tight')
        else:
            print(f"No valid data for region {region}, skipping plot")

        plt.close()


def fix_cluster_indices_columns(indices_df):
    """
    Fix cluster indices column names to match expected format
    """
    # Map from actual column names to expected names
    column_mapping = {}

    # Look for cluster columns
    for col in indices_df.columns:
        if col.startswith('cluster_') and not col.endswith('_similarity'):
            # Extract cluster number
            parts = col.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                cluster_num = parts[1]
                # Map to expected standardized column name
                column_mapping[col] = f'cluster_{cluster_num}_freq_std'

    # Rename columns
    indices_df = indices_df.rename(columns=column_mapping)

    # Keep only the mapped columns and essential time columns
    essential_cols = ['year', 'month', 'date']
    cluster_cols = list(column_mapping.values())

    # Filter to keep only existing columns
    available_cols = [col for col in essential_cols + cluster_cols if col in indices_df.columns]
    indices_df = indices_df[available_cols]

    return indices_df, cluster_cols


def analyze_cluster_z500_lead_lag_with_indices(cluster_indices_file, z500_file, output_dir,
                                               max_lag_days=90, lag_step=5):
    """
    Analyze lead-lag correlations between cluster indices and Z500 data
    """
    print(f"Loading cluster indices from {cluster_indices_file}")

    # Check if file exists
    if not os.path.exists(cluster_indices_file):
        print(f"Error: File {cluster_indices_file} not found")
        return None

    try:
        indices_df = pd.read_csv(cluster_indices_file)
        print(f"Loaded indices with shape: {indices_df.shape}")
        print(f"Available columns: {list(indices_df.columns)}")

        # Fix column names and filter to cluster indices
        indices_df, cluster_cols = fix_cluster_indices_columns(indices_df)

        if not cluster_cols:
            print("ERROR: No cluster columns found after transformation.")
            print(f"Available columns: {list(indices_df.columns)}")
            return None

        print(f"Found cluster columns: {cluster_cols}")

        # Handle case when columns are lowercase
        if 'year' in indices_df.columns and 'Year' not in indices_df.columns:
            indices_df['Year'] = indices_df['year']

        # Check if the file has a 'cluster' column instead of 'Cluster_N' columns
        if 'cluster' in indices_df.columns:
            print("Found 'cluster' column - converting to cluster-specific columns")

            # Get unique clusters
            unique_clusters = sorted(indices_df['cluster'].unique())
            print(f"Found {len(unique_clusters)} unique clusters: {unique_clusters}")

            # For each unique cluster, create a binary indicator column
            for cluster_num in unique_clusters:
                col_name = f'Cluster_{cluster_num}'
                # If there's an index_value column, use it for the cluster value
                if 'index_value' in indices_df.columns:
                    # For each year, assign the index_value if cluster matches, otherwise NaN
                    indices_df[col_name] = indices_df.apply(
                        lambda row: row['index_value'] if row['cluster'] == cluster_num else np.nan,
                        axis=1
                    )
                else:
                    # Create binary indicator (1 if matches cluster, 0 otherwise)
                    indices_df[col_name] = (indices_df['cluster'] == cluster_num).astype(int)

            # Now we need to aggregate by year to handle multiple entries per year
            year_col = 'Year' if 'Year' in indices_df.columns else 'year'
            cluster_cols = [f'Cluster_{c}' for c in unique_clusters]

            # Group by year and take mean of cluster columns (this handles NaNs appropriately)
            indices_df = indices_df.groupby(year_col)[cluster_cols].mean().reset_index()

            print(f"Created aggregated dataframe with columns: {indices_df.columns.tolist()}")

    except Exception as e:
        print(f"Error loading cluster indices: {str(e)}")
        return None

    try:
        print(f"Loading daily Z500 data from {z500_file}")
        # Load Z500 data
        ds_z500 = xr.open_dataset(z500_file)

        # Get time dimension
        time_dim = 'valid_time'  # Adjust if needed
        z500_times = pd.to_datetime(ds_z500[time_dim].values)

        # Create DataFrame for Z500 times
        z500_df = pd.DataFrame({'time': z500_times})
        z500_df['year'] = z500_df['time'].dt.year
        z500_df['month'] = z500_df['time'].dt.month
        z500_df['day'] = z500_df['time'].dt.day

        # Filter to DJF months
        djf_mask = z500_df['month'].isin([12, 1, 2])
        z500_df_djf = z500_df[djf_mask].copy()  # Create a copy to avoid SettingWithCopyWarning

        # Use isel() with indices instead of sel() with values
        djf_indices = np.where(djf_mask)[0]
        z500_djf = ds_z500.isel({time_dim: djf_indices})

        # Create winter years
        z500_df_djf['winter_year'] = z500_df_djf.apply(
            lambda x: x['year'] + 1 if x['month'] == 12 else x['year'], axis=1)

        # Define regions
        lon_min, lon_max = float(ds_z500.longitude.min()), float(ds_z500.longitude.max())

        # Define regions based on longitude range (same as before)
        if lon_min >= 0 and lon_max <= 360:
            regions = {
                'north_atlantic': {'lat': slice(60, 30), 'lon': slice(280, 360)},
                'europe': {'lat': slice(65, 35), 'lon': slice(350, 40)},
                'arctic': {'lat': slice(90, 65), 'lon': slice(0, 360)},
                'mediterranean': {'lat': slice(45, 30), 'lon': slice(350, 40)}
            }
        else:
            regions = {
                'north_atlantic': {'lat': slice(60, 30), 'lon': slice(-80, 0)},
                'europe': {'lat': slice(65, 35), 'lon': slice(-10, 40)},
                'arctic': {'lat': slice(90, 65), 'lon': slice(-180, 180)},
                'mediterranean': {'lat': slice(45, 30), 'lon': slice(-10, 40)}
            }

        # Calculate regional Z500 indices
        z500_indices = {}
        for region_name, region_coords in regions.items():
            try:
                # Calculate regional average
                regional_z500 = z500_djf.z.sel(
                    latitude=region_coords['lat'],
                    longitude=region_coords['lon']
                ).mean(dim=['latitude', 'longitude'])

                # Create DataFrame with daily values
                z500_values = pd.DataFrame({
                    'time': pd.to_datetime(z500_djf[time_dim].values),
                    'z500': regional_z500.values.flatten()
                })

                # Add year info
                z500_values['year'] = z500_values['time'].dt.year
                z500_values['month'] = z500_values['time'].dt.month
                z500_values['day'] = z500_values['time'].dt.day
                z500_values['winter_year'] = z500_values.apply(
                    lambda x: x['year'] + 1 if x['month'] == 12 else x['year'], axis=1)

                z500_indices[region_name] = z500_values
                print(f"Created Z500 index for {region_name} with {len(z500_values)} data points")

            except Exception as e:
                print(f"Error calculating {region_name} Z500 index: {str(e)}")
                continue

        # Get cluster columns after transformation
        cluster_cols = [col for col in indices_df.columns if col.startswith('Cluster_')]
        n_clusters = len(cluster_cols)

        # Debug: Check if cluster columns were found
        if not cluster_cols:
            print("ERROR: No cluster columns found after transformation. Available columns:",
                  indices_df.columns.tolist())
            return {}  # Return empty dict to signal failure

        print(f"Found {n_clusters} cluster columns: {cluster_cols}")

        # Dictionary to store results
        lag_results = {}

        # Calculate lag range
        lags = range(-max_lag_days, max_lag_days + 1, lag_step)

        # For each cluster index
        for i, cluster_col in enumerate(cluster_cols):
            # Extract cluster number from column name (e.g., 'Cluster_1' -> 1)
            try:
                cluster_num = int(cluster_col.split('_')[1])
            except:
                cluster_num = i + 1  # Fallback to 1-indexed position

            lag_results[cluster_num] = {}

            for region_name, z500_region_df in z500_indices.items():
                print(f"Processing correlations for {cluster_col}, Region {region_name}...")

                # Debug: print some values before merge
                print(
                    f"  Z500 winter years range: {z500_region_df['winter_year'].min()} to {z500_region_df['winter_year'].max()}")
                print(f"  Index file years range: {indices_df['Year'].min()} to {indices_df['Year'].max()}")

                # Merge with check for empty result
                z500_with_index = z500_region_df.merge(
                    indices_df[['Year', cluster_col]],
                    left_on='winter_year',
                    right_on='Year',
                    how='left'
                )

                # Debug: Check if merge produced valid results
                null_count = z500_with_index[cluster_col].isna().sum()
                total_count = len(z500_with_index)
                print(f"  After merge: {null_count}/{total_count} NaN values ({null_count / total_count * 100:.1f}%)")

                if z500_with_index[cluster_col].isna().all():
                    print(f"  WARNING: All values are NaN after merge for {cluster_col}")
                    correlations = [np.nan] * len(lags)
                    p_values = [np.nan] * len(lags)
                    n_samples = [0] * len(lags)
                else:
                    # Initialize correlation results
                    correlations = []
                    p_values = []
                    n_samples = []

                    # Set up time series
                    ts_z500 = z500_with_index.set_index('time')['z500']
                    ts_cluster = z500_with_index.set_index('time')[cluster_col]

                    # For each lag
                    for lag in lags:
                        try:
                            # Same lag calculation as before
                            if lag > 0:
                                cluster_series_aligned = ts_cluster.copy()
                                z500_series_aligned = ts_z500.shift(periods=lag)
                            else:
                                cluster_series_aligned = ts_cluster.shift(periods=-lag)
                                z500_series_aligned = ts_z500.copy()

                            combined = pd.DataFrame({
                                'cluster': cluster_series_aligned,
                                'z500': z500_series_aligned
                            }).dropna()

                            if len(combined) >= 30:
                                r, p = stats.pearsonr(combined['cluster'], combined['z500'])
                                correlations.append(r)
                                p_values.append(p)
                                n_samples.append(len(combined))
                            else:
                                print(f"  Not enough data points for lag {lag}: only {len(combined)} samples")
                                correlations.append(np.nan)
                                p_values.append(np.nan)
                                n_samples.append(0)

                        except Exception as e:
                            print(f"  Error at lag {lag}: {str(e)}")
                            correlations.append(np.nan)
                            p_values.append(np.nan)
                            n_samples.append(0)

                # Store results
                lag_results[cluster_num][region_name] = {
                    'lags': list(lags),
                    'correlations': correlations,
                    'p_values': p_values,
                    'n_samples': n_samples
                }

        # Save results if we have any
        if lag_results:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, 'z500_lead_lag_results_from_indices.pkl'), 'wb') as f:
                pickle.dump(lag_results, f)
            print(
                f"Successfully saved lag results to {os.path.join(output_dir, 'z500_lead_lag_results_from_indices.pkl')}")

        return lag_results

    except Exception as e:
        print(f"Error processing Z500 data: {str(e)}")

        traceback.print_exc()
        return None

# Main execution section - this runs when you execute the script
if __name__ == "__main__":
    # Configuration

    print("\n" + "=" * 50)
    print("Running daily Z500 lead-lag analysis using cluster indices")
    print("=" * 50 + "\n")

    cluster_indices_file = "som_projection_results_20250508_132053/index_timeseries/full_index_timeseries.csv"  # Path to your cluster indices
    z500_file = "era5_z500_daily_1deg_14_05_2025.nc"  # Path to your daily Z500 file
    output_dir = "lead_lag_results_daily_indices_20250515"
    max_lag_days = 180
    lag_step = 5

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use the new function that works with indices directly
    lag_results = analyze_cluster_z500_lead_lag_with_indices(
        cluster_indices_file=cluster_indices_file,
        z500_file=z500_file,
        output_dir=output_dir,
        max_lag_days=max_lag_days,
        lag_step=lag_step
    )

    # The rest of your plotting code remains the same
    if lag_results:
        # Get unique clusters and regions
        clusters = sorted(lag_results.keys())
        regions = set()
        for cluster in clusters:
            regions.update(lag_results[cluster].keys())
        regions = sorted(regions)

        print(f"Found {len(clusters)} clusters and {len(regions)} regions")

        # Create plots with the same functions you already have
        for cluster in clusters:
            for region in regions:
                if region in lag_results[cluster]:
                    result = lag_results[cluster][region]
                    create_lead_lag_plot(
                        lags=result['lags'],
                        correlations=result['correlations'],
                        p_values=result['p_values'],
                        n_samples=result['n_samples'],
                        cluster=cluster,
                        region_name=region,
                        output_dir=output_dir
                    )

        # Create combined plots
        create_combined_lead_lag_plots(
            lag_results=lag_results,
            clusters=clusters,
            regions=regions,
            output_dir=output_dir
        )

        print(f"Analysis complete. Results saved to {output_dir}")
    else:
        print("Analysis failed.")