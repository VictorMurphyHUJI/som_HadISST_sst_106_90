#!/usr/bin/env python3
"""
Cluster Index Generator for SOM Analysis
Creates standardized cluster indices for correlation analysis with other variables
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import pickle


def create_cluster_indices_for_applications(som, X, ds, clustering_results, output_dir,
                                            start_year=1950, end_year=None):
    """
    Create cluster indices for various applications with proper 1-indexing

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array
    ds : xarray.Dataset
        Original SST dataset with coordinates
    clustering_results : dict
        Results from cluster_som_patterns function (must have 1-indexed clusters)
    output_dir : str
        Directory to save analysis results
    start_year : int
        Starting year for analysis (default: 1950)
    end_year : int or None
        Ending year for analysis (default: None = present)

    Returns:
    --------
    str
        Path to the main cluster indices file
    """
    print(f"\nCreating cluster indices for applications...")
    print(f"Time period: {start_year} to {end_year if end_year else 'present'}")

    # Verify clusters are 1-indexed
    pattern_clusters = clustering_results["pattern_clusters"]
    unique_clusters = sorted(set(pattern_clusters.values()))

    min_cluster = min(unique_clusters) if unique_clusters else 1
    if min_cluster == 0:
        print("ERROR: Clusters are 0-indexed! This function requires 1-indexed clusters.")
        print("Please fix the clustering function to return 1-indexed clusters.")
        raise ValueError("Clusters must be 1-indexed")

    print(f"Verified: Working with {len(unique_clusters)} clusters: {unique_clusters}")

    # Create output directory
    indices_dir = os.path.join(output_dir, 'cluster_indices_for_applications')
    os.makedirs(indices_dir, exist_ok=True)

    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers (1-based)
    n_rows, n_cols = som.get_weights().shape[:2]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Map patterns to clusters (should already be 1-indexed)
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Filter by year range
    if end_year is None:
        end_year = years.max()

    year_mask = (years >= start_year) & (years <= end_year)
    filtered_times = times[year_mask]
    filtered_years = years[year_mask]
    filtered_months = months[year_mask]
    filtered_clusters = cluster_sequence[year_mask]

    print(f"Filtered to {len(filtered_times)} time points from {start_year} to {end_year}")

    # Calculate various types of indices
    all_indices = {}

    # For each cluster, calculate different index types
    for cluster in unique_clusters:
        print(f"Processing cluster {cluster}...")

        cluster_key = f'cluster_{cluster}'
        cluster_indices = []

        # Group by year and month
        df = pd.DataFrame({
            'time': filtered_times,
            'year': filtered_years,
            'month': filtered_months,
            'cluster': filtered_clusters
        })

        # Calculate monthly indices
        for (year, month), group in df.groupby(['year', 'month']):
            # FIX: Ensure year and month are integers
            year = int(year)
            month = int(month)

            # Calculate cluster frequency for this month
            cluster_count = np.sum(group['cluster'] == cluster)
            total_count = len(group)
            frequency = cluster_count / total_count if total_count > 0 else 0.0

            # Calculate various index types
            # 1. Simple frequency index
            frequency_index = frequency

            # 2. Standardized frequency index (will standardize later)
            # 3. Binary index (1 if cluster is dominant, 0 otherwise)
            is_dominant = group['cluster'].mode()[0] == cluster if len(group) > 0 else False
            binary_index = 1.0 if is_dominant else 0.0

            # Store the index values
            cluster_indices.append({
                'year': year,
                'month': month,
                'frequency': frequency,
                'index_value': frequency_index,  # Will be standardized later
                'binary_index': binary_index,
                'cluster_count': cluster_count,
                'total_count': total_count,
                'similarity': frequency * 100  # Percentage
            })

        # Convert to DataFrame for easier manipulation
        cluster_df = pd.DataFrame(cluster_indices)

        # Standardize the frequency index (mean=0, std=1)
        if len(cluster_df) > 0 and cluster_df['frequency'].std() > 0:
            cluster_df['index_value'] = (cluster_df['frequency'] - cluster_df['frequency'].mean()) / cluster_df[
                'frequency'].std()
        else:
            cluster_df['index_value'] = cluster_df['frequency']

        # Store the processed indices
        all_indices[cluster_key] = cluster_df.to_dict('records')

        print(f"  Created {len(cluster_indices)} monthly indices for cluster {cluster}")

    # Save all indices to files
    main_file = save_indices(all_indices, indices_dir, unique_clusters)

    print(f"\nCluster indices creation complete!")
    print(f"Main file: {main_file}")
    print(f"All files saved to: {indices_dir}")

    return main_file

def create_monthly_indices(df, unique_clusters):
    """Create monthly cluster frequency indices"""
    monthly_data = []

    for year in df['year'].unique():
        for month in range(1, 13):
            month_data = df[(df['year'] == year) & (df['month'] == month)]

            if len(month_data) > 0:
                row = {'year': year, 'month': month}

                # Calculate frequency for each cluster
                for cluster in unique_clusters:
                    cluster_freq = (month_data['cluster'] == cluster).mean()
                    row[f'cluster_{cluster}_freq'] = cluster_freq

                monthly_data.append(row)

    monthly_df = pd.DataFrame(monthly_data)

    # Add standardized versions
    for cluster in unique_clusters:
        freq_col = f'cluster_{cluster}_freq'
        if freq_col in monthly_df.columns:
            mean_val = monthly_df[freq_col].mean()
            std_val = monthly_df[freq_col].std()
            if std_val > 0:
                monthly_df[f'cluster_{cluster}_std'] = (monthly_df[freq_col] - mean_val) / std_val
            else:
                monthly_df[f'cluster_{cluster}_std'] = monthly_df[freq_col] - mean_val

    return monthly_df


def create_seasonal_indices(df, unique_clusters):
    """Create seasonal indices with focus on DJF"""
    seasonal_data = []

    # Define seasons
    seasons = {
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11]
    }

    for year in df['year'].unique():
        for season_name, months in seasons.items():
            # For DJF, handle year boundary (Dec belongs to next year's DJF)
            if season_name == 'DJF':
                # Get Dec from previous year and Jan, Feb from current year
                if year > df['year'].min():  # Ensure we have previous year data
                    dec_data = df[(df['year'] == year - 1) & (df['month'] == 12)]
                    jan_feb_data = df[(df['year'] == year) & (df['month'].isin([1, 2]))]
                    season_data = pd.concat([dec_data, jan_feb_data])
                    season_year = year
                else:
                    continue
            else:
                season_data = df[(df['year'] == year) & (df['month'].isin(months))]
                season_year = year

            if len(season_data) > 0:
                row = {'year': season_year, 'season': season_name}

                # Calculate frequency for each cluster
                for cluster in unique_clusters:
                    cluster_freq = (season_data['cluster'] == cluster).mean()
                    row[f'cluster_{cluster}_freq'] = cluster_freq

                seasonal_data.append(row)

    seasonal_df = pd.DataFrame(seasonal_data)

    # Add standardized versions for each season separately
    for season in seasons.keys():
        season_subset = seasonal_df[seasonal_df['season'] == season]
        for cluster in unique_clusters:
            freq_col = f'cluster_{cluster}_freq'
            if freq_col in season_subset.columns and len(season_subset) > 1:
                mean_val = season_subset[freq_col].mean()
                std_val = season_subset[freq_col].std()
                if std_val > 0:
                    seasonal_df.loc[seasonal_df['season'] == season, f'cluster_{cluster}_{season}_std'] = \
                        (season_subset[freq_col] - mean_val) / std_val

    return seasonal_df


def create_annual_indices(df, unique_clusters):
    """Create annual cluster frequency indices"""
    annual_data = []

    for year in df['year'].unique():
        year_data = df[df['year'] == year]

        if len(year_data) > 0:
            row = {'year': year}

            # Calculate frequency for each cluster
            for cluster in unique_clusters:
                cluster_freq = (year_data['cluster'] == cluster).mean()
                row[f'cluster_{cluster}_freq'] = cluster_freq

            annual_data.append(row)

    annual_df = pd.DataFrame(annual_data)

    # Add standardized versions
    for cluster in unique_clusters:
        freq_col = f'cluster_{cluster}_freq'
        if freq_col in annual_df.columns:
            mean_val = annual_df[freq_col].mean()
            std_val = annual_df[freq_col].std()
            if std_val > 0:
                annual_df[f'cluster_{cluster}_std'] = (annual_df[freq_col] - mean_val) / std_val
            else:
                annual_df[f'cluster_{cluster}_std'] = annual_df[freq_col] - mean_val

    return annual_df


def create_winter_indices(df, unique_clusters):
    """Create winter-specific indices (DJF only) with proper year assignment"""
    winter_data = []

    # Filter to DJF months
    djf_data = df[df['month'].isin([12, 1, 2])].copy()

    # Create winter year (December belongs to next year's winter)
    djf_data['winter_year'] = djf_data.apply(
        lambda row: row['year'] + 1 if row['month'] == 12 else row['year'], axis=1
    )

    # Group by winter year
    for winter_year in djf_data['winter_year'].unique():
        winter_season_data = djf_data[djf_data['winter_year'] == winter_year]

        if len(winter_season_data) > 0:
            row = {'winter_year': winter_year}

            # Calculate frequency for each cluster
            for cluster in unique_clusters:
                cluster_freq = (winter_season_data['cluster'] == cluster).mean()
                row[f'cluster_{cluster}_freq'] = cluster_freq

            # Also calculate monthly breakdown within winter
            for month in [12, 1, 2]:
                month_data = winter_season_data[winter_season_data['month'] == month]
                if len(month_data) > 0:
                    for cluster in unique_clusters:
                        cluster_freq = (month_data['cluster'] == cluster).mean()
                        month_name = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month]
                        row[f'cluster_{cluster}_{month_name}_freq'] = cluster_freq

            winter_data.append(row)

    winter_df = pd.DataFrame(winter_data)

    # Add standardized versions
    for cluster in unique_clusters:
        freq_col = f'cluster_{cluster}_freq'
        if freq_col in winter_df.columns:
            mean_val = winter_df[freq_col].mean()
            std_val = winter_df[freq_col].std()
            if std_val > 0:
                winter_df[f'cluster_{cluster}_std'] = (winter_df[freq_col] - mean_val) / std_val
            else:
                winter_df[f'cluster_{cluster}_std'] = winter_df[freq_col] - mean_val

    return winter_df


def create_binary_indices(df, unique_clusters):
    """Create binary indices for composite analysis (high/low states)"""
    binary_data = []

    # First calculate seasonal frequencies
    seasonal_df = create_seasonal_indices(df, unique_clusters)
    djf_data = seasonal_df[seasonal_df['season'] == 'DJF']

    for cluster in unique_clusters:
        freq_col = f'cluster_{cluster}_freq'
        if freq_col in djf_data.columns:
            freq_values = djf_data[freq_col].values

            # Define thresholds (e.g., upper and lower terciles)
            high_threshold = np.percentile(freq_values, 66.7)
            low_threshold = np.percentile(freq_values, 33.3)

            for _, row in djf_data.iterrows():
                year = row['year']
                freq_value = row[freq_col]

                binary_row = {
                    'year': year,
                    'cluster': cluster,
                    f'cluster_{cluster}_value': freq_value,
                    f'cluster_{cluster}_high': 1 if freq_value >= high_threshold else 0,
                    f'cluster_{cluster}_low': 1 if freq_value <= low_threshold else 0,
                    f'cluster_{cluster}_neutral': 1 if low_threshold < freq_value < high_threshold else 0,
                    f'cluster_{cluster}_tercile': 'high' if freq_value >= high_threshold
                    else 'low' if freq_value <= low_threshold
                    else 'neutral'
                }
                binary_data.append(binary_row)

    return pd.DataFrame(binary_data)


def create_dipole_indices(df, unique_clusters):
    """Create dipole indices for teleconnection analysis"""
    # Get seasonal data
    seasonal_df = create_seasonal_indices(df, unique_clusters)
    djf_data = seasonal_df[seasonal_df['season'] == 'DJF']

    dipole_data = []

    # Create all possible dipole combinations
    for i, cluster1 in enumerate(unique_clusters):
        for j, cluster2 in enumerate(unique_clusters):
            if i < j:  # Avoid duplicates
                dipole_name = f'cluster_{cluster1}_minus_{cluster2}'

                for _, row in djf_data.iterrows():
                    year = row['year']
                    freq1 = row[f'cluster_{cluster1}_freq']
                    freq2 = row[f'cluster_{cluster2}_freq']

                    dipole_value = freq1 - freq2

                    dipole_row = {
                        'year': year,
                        'dipole_name': dipole_name,
                        'cluster1': cluster1,
                        'cluster2': cluster2,
                        f'{dipole_name}_value': dipole_value,
                        f'cluster_{cluster1}_freq': freq1,
                        f'cluster_{cluster2}_freq': freq2
                    }
                    dipole_data.append(dipole_row)

    dipole_df = pd.DataFrame(dipole_data)

    # Add standardized dipole indices
    for dipole_name in dipole_df['dipole_name'].unique():
        dipole_subset = dipole_df[dipole_df['dipole_name'] == dipole_name]
        value_col = f'{dipole_name}_value'

        mean_val = dipole_subset[value_col].mean()
        std_val = dipole_subset[value_col].std()

        if std_val > 0:
            dipole_df.loc[dipole_df['dipole_name'] == dipole_name, f'{dipole_name}_std'] = \
                (dipole_subset[value_col] - mean_val) / std_val

    return dipole_df


def save_indices(all_indices, indices_dir, unique_clusters):
    """
    Save all cluster indices to various file formats

    Parameters:
    -----------
    all_indices : dict
        Dictionary containing all cluster indices
    indices_dir : str
        Directory to save files
    unique_clusters : list
        List of unique cluster numbers
    """
    print(f"Saving cluster indices to: {indices_dir}")

    # Save individual cluster files
    for cluster in unique_clusters:
        cluster_key = f'cluster_{cluster}'
        if cluster_key in all_indices:
            cluster_df = pd.DataFrame(all_indices[cluster_key])
            cluster_file = os.path.join(indices_dir, f'cluster_{cluster}_index.csv')
            cluster_df.to_csv(cluster_file, index=False)
            print(f"Saved cluster {cluster} indices: {cluster_file}")

    # Create comprehensive combined file
    print("Creating comprehensive combined file...")
    combined_data = []

    # Get all unique year-month combinations
    all_year_months = set()
    for cluster in unique_clusters:
        cluster_key = f'cluster_{cluster}'
        if cluster_key in all_indices:
            for entry in all_indices[cluster_key]:
                # FIX: Ensure year and month are integers
                year = int(entry['year'])
                month = int(entry['month'])
                all_year_months.add((year, month))

    # Sort year-month combinations
    all_year_months = sorted(all_year_months)

    # Create combined data
    for year, month in all_year_months:
        # FIX: Ensure year and month are integers
        year = int(year)
        month = int(month)

        row = {
            'year': year,
            'month': month,
            'date': f"{year}-{month:02d}-15"
        }

        # Add data for each cluster
        for cluster in unique_clusters:
            cluster_key = f'cluster_{cluster}'

            # Initialize default values
            row[f'cluster_{cluster}'] = 0.0
            row[f'cluster_{cluster}_similarity'] = 0.0

            # Find matching entry
            if cluster_key in all_indices:
                for entry in all_indices[cluster_key]:
                    # FIX: Ensure comparison values are integers
                    entry_year = int(entry['year'])
                    entry_month = int(entry['month'])

                    if entry_year == year and entry_month == month:
                        row[f'cluster_{cluster}'] = entry['index_value']
                        row[f'cluster_{cluster}_similarity'] = entry.get('similarity', 0.0)
                        break

        combined_data.append(row)

    # Convert to DataFrame and save
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sort_values(['year', 'month'])

    combined_file = os.path.join(indices_dir, 'cluster_indices_for_applications.csv')
    combined_df.to_csv(combined_file, index=False)

    print(f"Combined indices file saved: {combined_file}")
    print(f"File contains {len(combined_df)} time points")

    # Create lag analysis files
    try:
        lag_file, simple_file = create_lag_analysis_file(all_indices, indices_dir, unique_clusters)
        print(f"Lag analysis files created successfully")
    except Exception as e:
        print(f"Error creating lag analysis files: {str(e)}")
        import traceback
        traceback.print_exc()

    # Create summary statistics
    create_summary_statistics(all_indices, indices_dir, unique_clusters)

    print("All index files saved successfully!")

    return combined_file


def create_lag_analysis_file(all_indices, indices_dir, unique_clusters):
    """
    Create a file specifically formatted for lag correlation analysis

    Parameters:
    -----------
    all_indices : dict
        Dictionary containing all cluster indices
    indices_dir : str
        Directory to save the file
    unique_clusters : list
        List of unique cluster numbers
    """
    print("Creating lag analysis file...")

    # Create a comprehensive dataset for lag analysis
    lag_data = []

    # Get all available years and months from the indices
    all_years = set()
    all_months = set()

    for cluster in unique_clusters:
        cluster_key = f'cluster_{cluster}'
        if cluster_key in all_indices:
            for entry in all_indices[cluster_key]:
                all_years.add(entry['year'])
                all_months.add(entry['month'])

    # Sort years and months
    all_years = sorted(all_years)
    all_months = sorted(all_months)

    print(f"Creating lag analysis for {len(all_years)} years and {len(all_months)} months")

    # Create entries for each year-month combination
    for year in all_years:
        for month in all_months:
            # FIX: Ensure year and month are integers
            year = int(year)
            month = int(month)

            # Create date string (fixed formatting)
            date_str = f"{year}-{month:02d}-15"  # Use 15th of each month

            # Initialize row
            row = {
                'year': year,
                'month': month,
                'date': date_str
            }

            # Add cluster values
            for cluster in unique_clusters:
                cluster_key = f'cluster_{cluster}'

                # Find matching entry
                cluster_value = 0.0  # Default value
                similarity_value = 0.0

                if cluster_key in all_indices:
                    for entry in all_indices[cluster_key]:
                        # FIX: Ensure comparison values are integers
                        entry_year = int(entry['year'])
                        entry_month = int(entry['month'])

                        if entry_year == year and entry_month == month:
                            cluster_value = entry['index_value']
                            similarity_value = entry.get('similarity', 0.0)
                            break

                row[f'cluster_{cluster}'] = cluster_value
                row[f'cluster_{cluster}_similarity'] = similarity_value

            lag_data.append(row)

    # Convert to DataFrame
    lag_df = pd.DataFrame(lag_data)

    # Sort by year and month
    lag_df = lag_df.sort_values(['year', 'month'])

    # Save to CSV
    lag_file = os.path.join(indices_dir, 'cluster_indices_for_lag_analysis.csv')
    lag_df.to_csv(lag_file, index=False)

    print(f"Lag analysis file saved: {lag_file}")
    print(f"File contains {len(lag_df)} time points with {len(unique_clusters)} cluster indices")

    # Also create a simplified version with just cluster indices
    simple_columns = ['year', 'month', 'date'] + [f'cluster_{c}' for c in unique_clusters]
    simple_df = lag_df[simple_columns]

    simple_file = os.path.join(indices_dir, 'cluster_indices_simple.csv')
    simple_df.to_csv(simple_file, index=False)

    print(f"Simple cluster indices file saved: {simple_file}")

    return lag_file, simple_file

def create_metadata(indices_dir, unique_clusters, start_year, end_year, clustering_results):
    """Create metadata file describing the indices"""

    metadata = {
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'Cluster indices derived from SOM analysis of North Atlantic SST patterns',
        'clusters': unique_clusters,
        'time_period': f'{start_year}-{end_year}',
        'season_focus': 'DJF (December-January-February)',
        'spatial_domain': 'North Atlantic (0-60°N, 80-5°W)',
        'standardization': 'Z-scores (mean=0, std=1) calculated over the full time period',
        'files': {
            'monthly_cluster_indices.csv': 'Monthly cluster frequency indices',
            'seasonal_cluster_indices.csv': 'Seasonal cluster frequency indices (DJF, MAM, JJA, SON)',
            'annual_cluster_indices.csv': 'Annual cluster frequency indices',
            'winter_cluster_indices.csv': 'Winter-specific (DJF) cluster indices with monthly breakdown',
            'binary_cluster_indices.csv': 'Binary indices for composite analysis (high/low states)',
            'dipole_cluster_indices.csv': 'Dipole indices for teleconnection analysis',
            'cluster_indices_for_applications.csv': 'Master file with standardized indices for applications',
            'cluster_indices_for_lag_analysis.csv': 'Time series format for lag correlation analysis'
        },
        'cluster_interpretation': {
            '1': 'Cluster 1 - [Physical interpretation to be added]',
            '2': 'Cluster 2 - [Physical interpretation to be added]',
            '3': 'Cluster 3 - [Physical interpretation to be added]',
            '4': 'Cluster 4 - [Physical interpretation to be added]'
        },
        'applications': {
            'Z500_correlation': 'Use cluster_indices_for_applications.csv with Z500 data',
            'CDI_correlation': 'Use cluster_indices_for_applications.csv with Combined Drought Index data',
            'lag_analysis': 'Use cluster_indices_for_lag_analysis.csv for lead-lag correlation studies',
            'composite_analysis': 'Use binary_cluster_indices.csv to identify high/low cluster states'
        },
        'som_parameters': {
            'grid_size': f"{clustering_results.get('som_grid_size', 'Unknown')}",
            'n_clusters': len(unique_clusters),
            'clustering_method': 'K-means clustering on SOM patterns'
        }
    }

    # Save metadata
    import json
    metadata_filepath = os.path.join(indices_dir, 'cluster_indices_metadata.json')
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Also create a README file
    readme_content = f"""
# Cluster Indices for SOM Analysis

Created: {metadata['creation_date']}
Clusters: {unique_clusters}
Time Period: {start_year}-{end_year}
Focus Season: DJF (December-January-February)

## Files Description:

### Primary Application Files:
- **cluster_indices_for_applications.csv**: Main file for correlation analysis with Z500, CDI, etc.
- **cluster_indices_for_lag_analysis.csv**: Time series format for lead-lag correlation studies

### Detailed Index Files:
- **winter_cluster_indices.csv**: Winter-specific indices with monthly breakdown
- **binary_cluster_indices.csv**: High/low states for composite analysis  
- **dipole_cluster_indices.csv**: Dipole indices for teleconnection studies

### Index Columns:
- cluster_X_index: Standardized index for cluster X (mean=0, std=1)
- cluster_X_freq: Raw frequency of cluster X occurrence
- cluster_X_std: Standardized frequency

## Usage Examples:

### For Z500 Correlation:
```python
import pandas as pd
indices = pd.read_csv('cluster_indices_for_applications.csv')
# Correlate indices['cluster_1_index'] with your Z500 data
```

### For Lag Analysis:
```python
lag_indices = pd.read_csv('cluster_indices_for_lag_analysis.csv')
# Use for lead-lag correlation studies with monthly resolution
```

### For Composite Analysis:
```python
binary = pd.read_csv('binary_cluster_indices.csv')
high_years = binary[binary['cluster_1_high'] == 1]['year']
# Use high_years for composite analysis
```

## Notes:
- All indices are calculated from DJF North Atlantic SST patterns
- Standardized indices have mean=0 and standard deviation=1
- Winter year assignment: December belongs to following year's winter
- Clusters are 1-indexed (Cluster 1, 2, 3, 4)
"""

    readme_filepath = os.path.join(indices_dir, 'README.md')
    with open(readme_filepath, 'w') as f:
        f.write(readme_content)

    print(f"  Saved metadata: {metadata_filepath}")
    print(f"  Saved README: {readme_filepath}")


# Example usage function
def example_usage():
    """
    Example of how to use the cluster indices after creation
    """
    print("\n" + "=" * 60)
    print("EXAMPLE USAGE OF CLUSTER INDICES")
    print("=" * 60)

    print("""
# 1. Load the main application indices
import pandas as pd
indices = pd.read_csv('cluster_indices_for_applications.csv')

# 2. For Z500 correlation analysis:
# Correlate each cluster index with your Z500 data
for cluster in [1, 2, 3, 4]:
    cluster_index = indices[f'cluster_{cluster}_index']
    # correlate with your Z500 field at each grid point
    # correlation_map = correlate_field_with_index(z500_data, cluster_index)

# 3. For CDI correlation:
# Load your Combined Drought Index data
# cdi_correlation = correlate_timeseries(cdi_data, cluster_index)

# 4. For lag analysis (use monthly data):
lag_indices = pd.read_csv('cluster_indices_for_lag_analysis.csv')
lag_indices['date'] = pd.to_datetime(lag_indices['date'])

# Calculate lead-lag correlations
# for lag in range(-6, 7):  # ±6 months
#     lagged_correlation = calculate_lagged_correlation(
#         index_series=lag_indices['cluster_1_index'], 
#         target_series=your_target_data,
#         lag_months=lag
#     )

# 5. For composite analysis:
binary = pd.read_csv('binary_cluster_indices.csv')
cluster_1_high_years = binary[
    (binary['cluster'] == 1) & (binary['cluster_1_high'] == 1)
]['year'].values

# Use these years to create composites:
# high_composite = your_data.sel(time=your_data.time.dt.year.isin(cluster_1_high_years)).mean('time')
""")


def create_summary_statistics(all_indices, indices_dir, unique_clusters):
    """
    Create summary statistics for cluster indices

    Parameters:
    -----------
    all_indices : dict
        Dictionary containing all cluster indices
    indices_dir : str
        Directory to save summary statistics
    unique_clusters : list
        List of unique cluster numbers
    """
    print("Creating summary statistics...")

    import json
    import numpy as np

    # Create summary statistics
    summary_stats = {
        'metadata': {
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'description': 'Summary statistics for cluster indices',
            'n_clusters': len(unique_clusters),
            'clusters': unique_clusters,
            'n_time_points': 0
        },
        'cluster_statistics': {}
    }

    # Calculate statistics for each cluster
    for cluster in unique_clusters:
        cluster_key = f'cluster_{cluster}'
        if cluster_key in all_indices:
            cluster_data = all_indices[cluster_key]

            if cluster_data:  # Check if not empty
                # Extract index values
                index_values = [entry['index_value'] for entry in cluster_data]
                frequencies = [entry['frequency'] for entry in cluster_data]

                # Calculate statistics
                stats = {
                    'n_observations': len(cluster_data),
                    'index_stats': {
                        'mean': float(np.mean(index_values)),
                        'std': float(np.std(index_values)),
                        'min': float(np.min(index_values)),
                        'max': float(np.max(index_values)),
                        'median': float(np.median(index_values))
                    },
                    'frequency_stats': {
                        'mean_frequency': float(np.mean(frequencies)),
                        'std_frequency': float(np.std(frequencies)),
                        'min_frequency': float(np.min(frequencies)),
                        'max_frequency': float(np.max(frequencies))
                    }
                }

                summary_stats['cluster_statistics'][cluster] = stats

                # Update total time points
                if summary_stats['metadata']['n_time_points'] == 0:
                    summary_stats['metadata']['n_time_points'] = len(cluster_data)

    # Save summary statistics
    summary_file = os.path.join(indices_dir, 'cluster_indices_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=4)

    # Create a text summary
    summary_text_file = os.path.join(indices_dir, 'cluster_indices_summary.txt')
    with open(summary_text_file, 'w') as f:
        f.write("CLUSTER INDICES SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Created: {summary_stats['metadata']['creation_date']}\n")
        f.write(f"Number of clusters: {summary_stats['metadata']['n_clusters']}\n")
        f.write(f"Clusters: {summary_stats['metadata']['clusters']}\n")
        f.write(f"Time points: {summary_stats['metadata']['n_time_points']}\n\n")

        for cluster in unique_clusters:
            if cluster in summary_stats['cluster_statistics']:
                stats = summary_stats['cluster_statistics'][cluster]
                f.write(f"CLUSTER {cluster}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Observations: {stats['n_observations']}\n")
                f.write(f"Index Value - Mean: {stats['index_stats']['mean']:.4f}, "
                        f"Std: {stats['index_stats']['std']:.4f}\n")
                f.write(f"Index Value - Range: [{stats['index_stats']['min']:.4f}, "
                        f"{stats['index_stats']['max']:.4f}]\n")
                f.write(f"Frequency - Mean: {stats['frequency_stats']['mean_frequency']:.4f}, "
                        f"Std: {stats['frequency_stats']['std_frequency']:.4f}\n\n")

    print(f"  Saved summary statistics: {summary_file}")
    print(f"  Saved text summary: {summary_text_file}")


if __name__ == "__main__":
    print("Cluster Index Generator for SOM Analysis")
    print("This script creates comprehensive cluster indices for applications.")
    print("Import this module and call create_cluster_indices_for_applications()")
    example_usage()