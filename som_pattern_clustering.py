import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
import xarray as xr
import os
from scipy.spatial.distance import cosine
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE, MDS
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import networkx as nx
import plotly.express as px
from scipy.stats import pearsonr
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import html
import random

from matplotlib import gridspec
from scipy import stats


def cluster_som_patterns(som, method='hierarchical', n_clusters=None, metric='cosine'):
    """
    Cluster SOM patterns using different methods - FIXED VERSION

    Parameters:
    -----------
    som : MiniSom
        Trained SOM object
    method : str
        Clustering method ('kmeans', 'hierarchical', or 'custom')
    n_clusters : int or None
        Number of clusters (if None, will be determined automatically)
    metric : str
        Distance metric ('cosine', 'euclidean', 'correlation')

    Returns:
    --------
    dict
        Dictionary containing clustering results
    """
    # Get SOM weights
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    # Reshape to 2D array (patterns x features)
    patterns_2d = weights.reshape(n_rows * n_cols, n_features)

    # Compute distance matrix between patterns
    if metric == 'cosine':
        # Cosine distance matrix
        dist_matrix = squareform(pdist(patterns_2d, 'cosine'))
    elif metric == 'correlation':
        # Correlation distance matrix
        dist_matrix = squareform(pdist(patterns_2d, 'correlation'))
    else:
        # Euclidean distance matrix
        dist_matrix = squareform(pdist(patterns_2d, 'euclidean'))

    # Determine optimal number of clusters if not provided
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(patterns_2d, dist_matrix, max_clusters=min(10, n_rows * n_cols // 2))
        print(f"Automatically determined number of clusters: {n_clusters}")

    # Perform clustering
    if method == 'kmeans':
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(patterns_2d)

    elif method == 'hierarchical':
        # Hierarchical clustering
        hc = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = hc.fit_predict(dist_matrix)

    elif method == 'custom':
        # Custom similarity-based clustering
        cluster_labels = custom_similarity_clustering(dist_matrix, n_clusters, threshold=0.5)

    # IMPORTANT FIX: Ensure clusters are 1-indexed
    # sklearn clustering algorithms return 0-based labels, we need 1-based
    cluster_labels = cluster_labels + 1  # Convert 0-indexed to 1-indexed

    # Create mapping from pattern index to cluster label
    pattern_clusters = {}
    for i in range(len(cluster_labels)):
        pattern_idx = i + 1  # 1-based pattern index
        pattern_clusters[pattern_idx] = int(cluster_labels[i])  # Now properly 1-indexed

    # Calculate cluster statistics
    cluster_stats = calculate_cluster_statistics(patterns_2d, cluster_labels - 1, metric)  # Pass 0-indexed for calculations

    # Update cluster_stats to be 1-indexed for output
    updated_cluster_stats = {}
    for key, value in cluster_stats.items():
        if isinstance(value, dict):
            updated_value = {}
            for k, v in value.items():
                if isinstance(k, int):
                    updated_value[k + 1] = v  # Convert to 1-indexed
                else:
                    updated_value[k] = v
            updated_cluster_stats[key] = updated_value
        else:
            updated_cluster_stats[key] = value

    # Also update representative patterns to be 1-indexed
    if 'representative_patterns' in updated_cluster_stats:
        rep_patterns = {}
        for cluster, pattern_idx in updated_cluster_stats['representative_patterns'].items():
            rep_patterns[cluster] = pattern_idx + 1  # Make pattern index 1-based
        updated_cluster_stats['representative_patterns'] = rep_patterns

    # Prepare linkage matrix for dendrogram
    linkage_matrix = linkage(dist_matrix, method='average')

    return {
        "pattern_clusters": pattern_clusters,
        "cluster_labels": cluster_labels,  # Now 1-indexed
        "n_clusters": n_clusters,
        "distance_matrix": dist_matrix,
        "linkage_matrix": linkage_matrix,
        "cluster_stats": updated_cluster_stats  # Now 1-indexed
    }

#######################################################################################################################
#######################################################################################################################

def determine_optimal_clusters(patterns, distance_matrix, max_clusters=10):
    """
    Determine optimal number of clusters using silhouette score
    """
    silhouette_scores = []

    # Try different numbers of clusters
    for n_clusters in range(2, min(max_clusters + 1, len(patterns))):
        # Use hierarchical clustering with the distance matrix
        hc = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        labels = hc.fit_predict(distance_matrix)

        # Calculate silhouette score
        if len(np.unique(labels)) > 1:  # Ensure more than one cluster
            score = silhouette_score(patterns, labels)
            silhouette_scores.append((n_clusters, score))
            print(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

    # Find the number of clusters with the highest silhouette score
    if silhouette_scores:
        optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    else:
        optimal_clusters = 2  # Default if scores couldn't be calculated

    return optimal_clusters


#######################################################################################################################
#######################################################################################################################

def custom_similarity_clustering(distance_matrix, n_clusters, threshold=0.5):
    """
    Custom clustering based on similarity threshold
    """
    n_patterns = distance_matrix.shape[0]

    # Initialize each pattern as its own cluster
    labels = np.arange(n_patterns)

    # Convert distance to similarity (1 - distance)
    similarity_matrix = 1 - distance_matrix

    # Iteratively merge most similar clusters
    while len(np.unique(labels)) > n_clusters:
        current_clusters = np.unique(labels)

        # Find the most similar pair of clusters
        max_similarity = -1
        merge_clusters = None

        for i in range(len(current_clusters)):
            for j in range(i + 1, len(current_clusters)):
                cluster1 = current_clusters[i]
                cluster2 = current_clusters[j]

                # Get patterns in each cluster
                patterns1 = np.where(labels == cluster1)[0]
                patterns2 = np.where(labels == cluster2)[0]

                # Calculate average similarity between clusters
                cluster_similarity = 0
                count = 0
                for p1 in patterns1:
                    for p2 in patterns2:
                        cluster_similarity += similarity_matrix[p1, p2]
                        count += 1

                if count > 0:
                    avg_similarity = cluster_similarity / count

                    if avg_similarity > max_similarity and avg_similarity > threshold:
                        max_similarity = avg_similarity
                        merge_clusters = (cluster1, cluster2)

        if merge_clusters is None:
            # No clusters to merge above threshold
            break

        # Merge the clusters
        old_label, new_label = merge_clusters
        labels[labels == old_label] = new_label

    # Relabel to ensure consecutive integers
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    new_labels = np.array([label_map[l] for l in labels])

    return new_labels


#######################################################################################################################
#######################################################################################################################

def calculate_cluster_statistics(patterns, cluster_labels, metric='cosine'):
    """
    Calculate statistics for each cluster
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    stats = {
        "cluster_sizes": {},
        "cluster_cohesion": {},
        "cluster_separation": {},
        "representative_patterns": {}
    }

    # Calculate centroid for each cluster
    centroids = []
    for cluster in unique_clusters:
        cluster_patterns = patterns[cluster_labels == cluster]
        centroid = np.mean(cluster_patterns, axis=0)
        centroids.append(centroid)

        # Cluster size
        stats["cluster_sizes"][int(cluster)] = len(cluster_patterns)

        # Cluster cohesion (average distance to centroid)
        cohesion = 0
        for pattern in cluster_patterns:
            if metric == 'cosine':
                dist = cosine(pattern, centroid)
            else:
                dist = np.linalg.norm(pattern - centroid)
            cohesion += dist

        if len(cluster_patterns) > 0:
            stats["cluster_cohesion"][int(cluster)] = cohesion / len(cluster_patterns)
        else:
            stats["cluster_cohesion"][int(cluster)] = 0

        # Find representative pattern (closest to centroid)
        min_dist = float('inf')
        rep_idx = -1

        for i, pattern in enumerate(cluster_patterns):
            if metric == 'cosine':
                dist = cosine(pattern, centroid)
            else:
                dist = np.linalg.norm(pattern - centroid)

            if dist < min_dist:
                min_dist = dist
                rep_idx = i

        if rep_idx >= 0:
            # Get the original pattern index
            orig_indices = np.where(cluster_labels == cluster)[0]
            stats["representative_patterns"][int(cluster)] = int(orig_indices[rep_idx]) + 1  # 1-based

    # Calculate separation between clusters
    for i, cluster1 in enumerate(unique_clusters):
        stats["cluster_separation"][int(cluster1)] = {}

        for j, cluster2 in enumerate(unique_clusters):
            if i != j:
                if metric == 'cosine':
                    dist = cosine(centroids[i], centroids[j])
                else:
                    dist = np.linalg.norm(centroids[i] - centroids[j])

                stats["cluster_separation"][int(cluster1)][int(cluster2)] = dist

    return stats


#######################################################################################################################
#######################################################################################################################

def plot_som_pattern_clusters(som, clustering_results, original_shape, ds, output_dir):
    """
    Create visualizations of SOM pattern clusters with error handling
    """
    # Create cluster directory
    cluster_dir = os.path.join(output_dir, 'pattern_clusters')
    os.makedirs(cluster_dir, exist_ok=True)

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]
    cluster_stats = clustering_results.get("cluster_stats", {})

    # Get unique clusters that actually exist
    unique_clusters = sorted(set(pattern_clusters.values()))
    n_clusters = len(unique_clusters)

    # Get SOM weights
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    # Create cluster colors - make sure we have enough colors
    # cluster_colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    standard_colors = ['blue', 'green', 'orange', 'red']

    # Create a mapping from cluster value to index
    cluster_to_index = {c: i for i, c in enumerate(unique_clusters)}

    # 1. Create grid visualization showing pattern clusters - try to skip any problematic calls
    try:
        plt.figure(figsize=(10, 8))
        for i in range(n_rows):
            for j in range(n_cols):
                pattern_idx = i * n_cols + j + 1
                if pattern_idx in pattern_clusters:
                    cluster = pattern_clusters[pattern_idx]
                    cluster_idx = cluster - 1  # Adjust for 1-indexed clusters
                    color = standard_colors[cluster_idx % len(standard_colors)]
                    plt.scatter(j, n_rows - 1 - i, color=color, s=100)

                    plt.text(j, n_rows - 1 - i, str(pattern_idx),
                             ha='center', va='center', fontsize=10, fontweight='bold')

        plt.title("SOM Pattern Clustering")
        plt.xlim(-0.5, n_cols - 0.5)
        plt.ylim(-0.5, n_rows - 0.5)
        plt.grid(True)

        # Create legend
        for c_idx, cluster in enumerate(unique_clusters):
            if c_idx < len(cluster_colors):
                color = cluster_colors[c_idx]
            else:
                color = (0.5, 0.5, 0.5, 1.0)  # Default gray

            plt.scatter([], [], color=color, label=f"Cluster {cluster}")

        plt.legend(loc='best')
        plt.savefig(os.path.join(cluster_dir, 'som_pattern_clusters_grid.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Error creating cluster grid visualization: {str(e)}")

    # Skip other problematic visualizations
    try:
        # Try to create the interactive cluster visualization
        create_interactive_cluster_visualization(
            som, clustering_results, original_shape, ds, cluster_dir
        )
    except Exception as e:
        print(f"Warning: Error creating interactive cluster visualization: {str(e)}")

    # Return results dictionary
    return {
        'cluster_dir': cluster_dir,
        'cluster_df': None  # We might not have this if there was an error
    }


#######################################################################################################################
#######################################################################################################################

def create_interactive_cluster_visualization(som, clustering_results, original_shape, ds, output_dir):
    """
    Create interactive visualization of SOM patterns clusters with specific color scheme

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    clustering_results : dict
        Results from cluster_som_patterns
    original_shape : tuple
        Original shape of the data
    ds : xarray.Dataset
        Original dataset with coordinates
    output_dir : str
        Directory to save output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]
    distance_matrix = clustering_results["distance_matrix"]

    # Get unique clusters that actually exist in the data
    unique_clusters = sorted(set(pattern_clusters.values()))
    n_clusters = len(unique_clusters)

    # Get SOM weights
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    # Create 2D projection of patterns using MDS
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pattern_positions = mds.fit_transform(distance_matrix)

    # Define specific colors for clusters 1-4 (matching your cluster frequency plot)
    specific_colors = {
        1: 'blue',  # Cluster 1
        2: 'orange',  # Cluster 2
        3: 'green',  # Cluster 3
        4: 'red'  # Cluster 4
    }

    # Create 2D projection plot
    fig = plt.figure(figsize=(12, 10))

    # Plot each pattern with proper cluster coloring
    for pattern_idx in range(1, n_rows * n_cols + 1):
        if pattern_idx in pattern_clusters:
            cluster = pattern_clusters[pattern_idx]
            color = specific_colors.get(cluster, 'gray')

            plt.scatter(
                pattern_positions[pattern_idx - 1, 0],
                pattern_positions[pattern_idx - 1, 1],
                c=color,
                s=80,
                edgecolors='black'
            )

            # Add pattern number label
            plt.text(
                pattern_positions[pattern_idx - 1, 0],
                pattern_positions[pattern_idx - 1, 1],
                str(pattern_idx),
                fontsize=10,
                ha='center',
                va='center'
            )

    # Add legend
    for cluster in unique_clusters:
        plt.scatter([], [], c=specific_colors.get(cluster, 'gray'),
                    label=f'Cluster {cluster}', s=80, edgecolors='black')

    plt.legend(loc='upper right')
    plt.title('2D Projection of SOM Patterns')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, 'pattern_2d_projection.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Create grid visualization of pattern clusters
    fig_grid = go.Figure()

    # Create a legend with cluster colors
    for cluster in unique_clusters:
        color = specific_colors.get(cluster, 'gray')
        fig_grid.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=f'Cluster {cluster}'
        ))

    # Add pattern points on the grid
    for i in range(n_rows):
        for j in range(n_cols):
            pattern_idx = i * n_cols + j + 1
            if pattern_idx in pattern_clusters:
                cluster = pattern_clusters[pattern_idx]
                color = specific_colors.get(cluster, 'gray')

                fig_grid.add_trace(go.Scatter(
                    x=[j],
                    y=[n_rows - i - 1],  # Invert y-axis to match typical SOM visualization
                    mode='markers+text',
                    marker=dict(
                        size=30,
                        color=color,
                        line=dict(width=1, color='black')
                    ),
                    text=str(pattern_idx),
                    textposition='middle center',
                    showlegend=False
                ))

    fig_grid.update_layout(
        title='SOM Pattern Clustering',
        xaxis=dict(
            range=[-0.5, n_cols - 0.5],
            tickvals=list(range(n_cols)),
            ticktext=list(range(n_cols)),
            title='Column'
        ),
        yaxis=dict(
            range=[-0.5, n_rows - 0.5],
            tickvals=list(range(n_rows)),
            ticktext=list(range(n_rows)),
            title='Row',
            scaleanchor='x',
            scaleratio=1
        ),
        width=800,
        height=800
    )

    fig_grid.write_html(os.path.join(output_dir, 'interactive_pattern_grid.html'))

    # 3. Create distance matrix heatmap
    fig_dist = go.Figure(data=go.Heatmap(
        z=distance_matrix,
        x=[f"P{i + 1}" for i in range(n_rows * n_cols)],
        y=[f"P{i + 1}" for i in range(n_rows * n_cols)],
        colorscale='Viridis_r',
        hoverongaps=False,
        colorbar=dict(title="Distance")
    ))

    fig_dist.update_layout(
        title='Pattern Distance Matrix',
        xaxis_title='Pattern',
        yaxis_title='Pattern',
        height=800,
        width=800
    )

    fig_dist.write_html(os.path.join(output_dir, 'interactive_distance_matrix.html'))

    # Create additional cluster-focused visualization
    fig_summary = go.Figure()

    # Add pattern data points colored by cluster
    for cluster in unique_clusters:
        # Get patterns in this cluster
        pattern_indices = [idx for idx, c in pattern_clusters.items() if c == cluster]

        if pattern_indices:
            color = specific_colors.get(cluster, 'gray')

            # Extract pattern positions
            cluster_x = [pattern_positions[i - 1, 0] for i in pattern_indices]
            cluster_y = [pattern_positions[i - 1, 1] for i in pattern_indices]

            # Add trace for this cluster
            fig_summary.add_trace(go.Scatter(
                x=cluster_x,
                y=cluster_y,
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=color,
                    line=dict(width=1, color='black')
                ),
                text=[str(i) for i in pattern_indices],
                textposition='middle center',
                name=f"Cluster {cluster}"
            ))

    fig_summary.update_layout(
        title='Cluster Summary View',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        height=800,
        width=1000
    )

    fig_summary.write_html(os.path.join(output_dir, 'cluster_summary.html'))

    print(f"Interactive cluster visualizations saved to: {output_dir}")
    return {'output_dir': output_dir}


#######################################################################################################################
#######################################################################################################################

def apply_clustering_to_som(som, original_shape, ds, method='hierarchical', n_clusters=None, output_dir=None):
    """
    Main function to apply clustering to SOM patterns with proper 1-indexed clusters
    """
    print(f"\nApplying {method} clustering to SOM patterns...")

    # Perform clustering
    clustering_results = cluster_som_patterns(
        som=som,
        method=method,
        n_clusters=n_clusters
    )

    # Get pattern clusters (should now be properly 1-indexed)
    pattern_clusters = clustering_results["pattern_clusters"]
    unique_clusters = sorted(set(pattern_clusters.values()))

    print(f"\nIdentified {len(unique_clusters)} pattern clusters")

    # Print clustering results to confirm 1-indexing
    for cluster in sorted(unique_clusters):
        patterns = [idx for idx, c in pattern_clusters.items() if c == cluster]
        print(f"Cluster {cluster}: {len(patterns)} patterns - {patterns}")

    # Verify clusters are 1-indexed
    min_cluster = min(unique_clusters) if unique_clusters else 1
    if min_cluster == 0:
        print("ERROR: Clusters are still 0-indexed! Fixing...")
        # Emergency fix if something went wrong
        new_pattern_clusters = {}
        for pattern_idx, cluster in pattern_clusters.items():
            new_pattern_clusters[pattern_idx] = cluster + 1
        clustering_results["pattern_clusters"] = new_pattern_clusters
        unique_clusters = sorted(set(new_pattern_clusters.values()))
        print("Fixed: Clusters are now 1-indexed")
    else:
        print("Confirmed: Clusters are properly 1-indexed")

    # Update the n_clusters to reflect the actual number of unique clusters
    clustering_results["n_clusters"] = len(unique_clusters)

    # Create visualizations if output directory is provided
    if output_dir is not None:
        viz_results = plot_som_pattern_clusters(
            som=som,
            clustering_results=clustering_results,
            original_shape=original_shape,
            ds=ds,
            output_dir=output_dir
        )
        print(f"\nCluster visualizations saved to: {viz_results['cluster_dir']}")

    return clustering_results

#######################################################################################################################
#######################################################################################################################

def analyze_cluster_occurrences(som, X, ds, clustering_results):
    """Analyze temporal distribution of pattern clusters"""
    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]

    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert BMU grid indices to pattern numbers (1-based)
    pattern_sequence = np.array([i * som.get_weights().shape[1] + j + 1
                                 for i, j in bmu_indices])

    # Convert pattern sequence to cluster sequence
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)

    # Analyze cluster frequency by year, month, etc.
    results = {
        'times': times,
        'cluster_sequence': cluster_sequence,
        'yearly_distribution': {},
        'monthly_distribution': {}
    }

    # Calculate yearly distribution
    years = times.year
    for year in np.unique(years):
        year_mask = (years == year)
        year_clusters = cluster_sequence[year_mask]
        if len(year_clusters) > 0:
            # Count occurrences of each cluster
            cluster_counts = {}
            for cluster in np.unique(year_clusters):
                cluster_counts[int(cluster)] = np.sum(year_clusters == cluster)
            results['yearly_distribution'][int(year)] = cluster_counts

    # Calculate monthly distribution
    months = times.month
    for month in np.unique(months):
        month_mask = (months == month)
        month_clusters = cluster_sequence[month_mask]
        if len(month_clusters) > 0:
            # Count occurrences of each cluster
            cluster_counts = {}
            for cluster in np.unique(month_clusters):
                cluster_counts[int(cluster)] = np.sum(month_clusters == cluster)
            results['monthly_distribution'][int(month)] = cluster_counts

    return results


#######################################################################################################################
#######################################################################################################################

def correlate_clusters_with_indices(cluster_occurrences, climate_indices):
    """Correlate cluster occurrences with climate indices"""
    # Example with NAO index
    if 'NAO' in climate_indices:
        nao_index = climate_indices['NAO']

        # For each cluster, calculate correlation with NAO
        results = {}
        for cluster in np.unique(cluster_occurrences['cluster_sequence']):
            # Create binary occurrence series (1 when this cluster occurs, 0 otherwise)
            cluster_binary = (cluster_occurrences['cluster_sequence'] == cluster).astype(int)

            # Calculate correlation
            corr, p_value = pearsonr(cluster_binary, nao_index)
            results[int(cluster)] = {
                'correlation': corr,
                'p_value': p_value
            }

        return results


#######################################################################################################################
#######################################################################################################################

def plot_cluster_frequency_over_time(cluster_occurrences, output_dir):
    # Create results directory
    os.makedirs(os.path.join(output_dir, 'cluster_analysis'), exist_ok=True)

    # Extract yearly distribution
    yearly_dist = cluster_occurrences['yearly_distribution']

    # Convert to DataFrame for easier plotting
    years = sorted(yearly_dist.keys())
    clusters = set()
    for year_data in yearly_dist.values():
        clusters.update(year_data.keys())
    clusters = sorted(clusters)  # These should already be 1-indexed

    # Initialize DataFrame
    df = pd.DataFrame(index=years, columns=clusters)
    df = df.fillna(0)

    # Fill DataFrame with data
    for year, year_data in yearly_dist.items():
        for cluster, count in year_data.items():
            df.loc[year, cluster] = count

    # Calculate relative frequencies
    df_rel = df.div(df.sum(axis=1), axis=0)

    # Create a matplotlib figure instead of Plotly for PNG output
    plt.figure(figsize=(12, 8))

    # Define colors for clusters - matching the colors in your second image
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    # Plot each cluster
    for i, cluster in enumerate(sorted(clusters)):
        color_idx = i % len(colors)  # In case we have more clusters than colors

        plt.plot(df_rel.index, df_rel[cluster],
                 marker='o',
                 linewidth=2,
                 label=f'Cluster {cluster}',  # This will show actual cluster number
                 color=colors[color_idx])
        # Use the color scheme

    plt.title('Cluster Frequency Over Time (1950 onwards)')
    plt.xlabel('Year')
    plt.ylabel('Relative Frequency')
    plt.legend()
    plt.grid(alpha=0.3)

    # Save as PNG
    plt.savefig(os.path.join(output_dir, 'cluster_analysis', 'cluster_frequency_over_time.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Still keep the original HTML version for interactive viewing
    fig = go.Figure()
    for i, cluster in enumerate(clusters):
        fig.add_trace(
            go.Scatter(
                x=df_rel.index,
                y=df_rel[cluster],
                mode='lines+markers',
                name=f'Cluster {cluster}',
                line=dict(width=2, color=colors[i % len(colors)]),  # Use the same colors
                marker=dict(size=8)
            )
        )

    fig.update_layout(
        title='Cluster Frequency over Time',
        xaxis_title='Year',
        yaxis_title='Relative Frequency',
        height=600,
        width=1000,
        legend_title='Cluster',
        hovermode='x unified'
    )

    fig.write_html(os.path.join(output_dir, 'cluster_analysis', 'cluster_frequency_over_time.html'))

    # Create heatmap of yearly distribution
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=df.values.T,
        x=df.index,
        y=[f'Cluster {c}' for c in df.columns],
        colorscale='Viridis',
        colorbar=dict(title='Count')
    ))

    fig_heatmap.update_layout(
        title='Cluster Occurrence Heatmap',
        xaxis_title='Year',
        yaxis_title='Cluster',
        height=600,
        width=1000
    )

    fig_heatmap.write_html(os.path.join(output_dir, 'cluster_analysis', 'cluster_occurrence_heatmap.html'))

    # Plot monthly distribution
    monthly_dist = cluster_occurrences['monthly_distribution']
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Convert to DataFrame
    months = sorted(monthly_dist.keys())

    # Initialize DataFrame
    df_monthly = pd.DataFrame(index=months, columns=clusters)
    df_monthly = df_monthly.fillna(0)

    # Fill DataFrame with data
    for month, month_data in monthly_dist.items():
        for cluster, count in month_data.items():
            df_monthly.loc[month, cluster] = count

    # Replace month numbers with names
    df_monthly.index = [month_names[i - 1] for i in df_monthly.index]

    # Calculate relative frequencies
    df_monthly_rel = df_monthly.div(df_monthly.sum(axis=1), axis=0)

    # Create bar chart of monthly distribution
    fig_monthly = go.Figure()

    for cluster in clusters:
        fig_monthly.add_trace(
            go.Bar(
                x=df_monthly_rel.index,
                y=df_monthly_rel[cluster],
                name=f'Cluster {cluster}'
            )
        )

    fig_monthly.update_layout(
        title='Cluster Distribution by Month',
        xaxis_title='Month',
        yaxis_title='Relative Frequency',
        height=600,
        width=1000,
        legend_title='Cluster',
        barmode='stack'
    )

    fig_monthly.write_html(os.path.join(output_dir, 'cluster_analysis', 'cluster_monthly_distribution.html'))

    return {
        'yearly_df': df,
        'monthly_df': df_monthly
    }


#######################################################################################################################
#######################################################################################################################

def create_cluster_composites(clustering_results, som, ds, output_dir):
    """
    Create composite maps for each cluster

    Parameters:
    -----------
    clustering_results : dict
        Results from cluster_som_patterns
    som : MiniSom
        Trained SOM object
    ds : xarray.Dataset
        Original dataset with coordinates
    output_dir : str
        Directory to save output plots
    """

    # Create composites directory
    composites_dir = os.path.join(output_dir, 'cluster_composites')
    os.makedirs(composites_dir, exist_ok=True)

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]
    n_clusters = clustering_results["n_clusters"]

    # Get SOM weights
    weights = som.get_weights()
    n_rows, n_cols, n_features = weights.shape

    # Create land mask and coordinates
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='black',
                                        facecolor='lightgray')
    lons, lats = np.meshgrid(ds.longitude.values, ds.latitude.values)

    # For each cluster, create a composite of patterns
    for cluster in range(1, n_clusters + 1):
        # Get patterns in this cluster
        cluster_patterns = [idx for idx, c in pattern_clusters.items() if c == cluster]

        if not cluster_patterns:
            continue

        # Get weights for patterns in this cluster
        cluster_weights = []
        for pattern_idx in cluster_patterns:
            i, j = (pattern_idx - 1) // n_cols, (pattern_idx - 1) % n_cols
            pattern = weights[i, j].reshape(ds.sst.shape[1:])
            cluster_weights.append(pattern)

        # Calculate composite (mean of all patterns in cluster)
        composite = np.mean(cluster_weights, axis=0)

        # Plot composite
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

        # Plot SST anomaly composite
        im = ax.pcolormesh(lons, lats, composite,
                           transform=ccrs.PlateCarree(),
                           cmap='RdBu_r',
                           shading='auto')

        # Add land and coastlines
        ax.add_feature(land)
        ax.coastlines(resolution='50m')

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        # Add colorbar and title
        plt.colorbar(im, ax=ax, label='Normalized SST Anomaly')
        ax.set_title(
            f'Cluster {cluster} Composite\n({len(cluster_patterns)} patterns: {", ".join(map(str, cluster_patterns))})')

        # Set extent
        ax.set_extent([ds.longitude.min(), ds.longitude.max(),
                       ds.latitude.min(), ds.latitude.max()],
                      crs=ccrs.PlateCarree())

        # Save figure
        plt.savefig(os.path.join(composites_dir, f'cluster_{cluster}_composite.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save individual patterns in this cluster
        for pattern_idx in cluster_patterns:
            i, j = (pattern_idx - 1) // n_cols, (pattern_idx - 1) % n_cols
            pattern = weights[i, j].reshape(ds.sst.shape[1:])

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

            im = ax.pcolormesh(lons, lats, pattern,
                               transform=ccrs.PlateCarree(),
                               cmap='RdBu_r',
                               shading='auto')

            ax.add_feature(land)
            ax.coastlines(resolution='50m')

            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            plt.colorbar(im, ax=ax, label='Normalized SST Anomaly')
            ax.set_title(f'Pattern {pattern_idx} (Cluster {cluster})')

            ax.set_extent([ds.longitude.min(), ds.longitude.max(),
                           ds.latitude.min(), ds.latitude.max()],
                          crs=ccrs.PlateCarree())

            plt.savefig(os.path.join(composites_dir, f'cluster_{cluster}_pattern_{pattern_idx}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)

    return composites_dir


#######################################################################################################################
#######################################################################################################################

def get_nao_index_data(start_year=1950, end_year=None):
    """
    Generate or load NAO index data for correlation analysis

    Parameters:
    -----------
    start_year : int
        Start year for NAO data
    end_year : int or None
        End year for NAO data (if None, uses current year)

    Returns:
    --------
    numpy.ndarray
        NAO index values
    """

    # If end_year not provided, use current year
    if end_year is None:
        end_year = datetime.now().year

    # For demonstration, generate synthetic NAO index data
    # In a real implementation, you would load this from a file or API
    # This is just a placeholder that creates random data with some trend and seasonality

    years = np.arange(start_year, end_year + 1)
    n_months = len(years) * 12

    # Create time index for monthly data
    time_index = np.arange(n_months)

    # Generate synthetic NAO with trend, seasonality, and noise
    trend = 0.0001 * time_index  # Small trend
    seasonality = 0.5 * np.sin(2 * np.pi * time_index / 12)  # Seasonal cycle
    noise = np.random.normal(0, 1, n_months)  # Random noise

    # Combine components
    nao_index = trend + seasonality + noise

    # Filter to DJF months only
    is_djf = np.zeros(n_months, dtype=bool)

    # Mark December, January, February
    for i, month_idx in enumerate(time_index):
        month = month_idx % 12 + 1  # Convert to 1-12 month numbering
        if month in [12, 1, 2]:
            is_djf[i] = True

    # Return DJF-only data
    return nao_index[is_djf]


#######################################################################################################################
#######################################################################################################################

def create_cluster_network(clustering_results, output_dir):
    """
    Create a network visualization showing relationships between clusters
    with proper error handling for 1-indexed clusters
    """
    # Create network directory
    network_dir = os.path.join(output_dir, 'cluster_network')
    os.makedirs(network_dir, exist_ok=True)

    # Get distance matrix and pattern clusters
    distance_matrix = clustering_results["distance_matrix"]
    pattern_clusters = clustering_results["pattern_clusters"]

    # Get unique clusters that actually exist
    unique_clusters = sorted(set(pattern_clusters.values()))
    n_clusters = len(unique_clusters)

    # Create a graph
    G = nx.Graph()

    # Add nodes (patterns)
    for pattern_idx in pattern_clusters:
        cluster = pattern_clusters[pattern_idx]
        G.add_node(pattern_idx, cluster=cluster)

    # Add edges (connections between patterns)
    n_patterns = len(pattern_clusters)

    # Calculate threshold for including edges (25th percentile of distances)
    flat_distances = []
    pattern_indices = sorted(pattern_clusters.keys())

    for i, p1 in enumerate(pattern_indices):
        for j, p2 in enumerate(pattern_indices):
            if i < j:  # Only consider unique pairs
                # Convert to 0-based indexing for distance matrix
                idx1 = p1 - 1 if isinstance(p1, int) else int(p1) - 1
                idx2 = p2 - 1 if isinstance(p2, int) else int(p2) - 1

                if 0 <= idx1 < len(distance_matrix) and 0 <= idx2 < len(distance_matrix):
                    flat_distances.append(distance_matrix[idx1, idx2])

    if flat_distances:
        distance_threshold = np.percentile(flat_distances, 25)  # 25th percentile

        # Add edges where distance is below threshold
        for i, p1 in enumerate(pattern_indices):
            for j, p2 in enumerate(pattern_indices):
                if i < j:  # Only consider unique pairs
                    # Convert to 0-based indexing for distance matrix
                    idx1 = p1 - 1 if isinstance(p1, int) else int(p1) - 1
                    idx2 = p2 - 1 if isinstance(p2, int) else int(p2) - 1

                    if 0 <= idx1 < len(distance_matrix) and 0 <= idx2 < len(distance_matrix):
                        distance = distance_matrix[idx1, idx2]

                        if distance < distance_threshold:
                            G.add_edge(p1, p2, weight=1.0 - distance)  # Convert distance to similarity

    # Create position layout using force-directed algorithm
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Create cluster-to-index mapping for colors
    cluster_to_index = {c: i for i, c in enumerate(unique_clusters)}

    # Create node color mapping based on clusters
    standard_colors = ['blue', 'green', 'orange', 'red']

    node_colors = []
    for node in G.nodes():
        if node in pattern_clusters:
            cluster = pattern_clusters[node]
            cluster_idx = cluster - 1  # Adjust for 1-indexed clusters
            node_colors.append(standard_colors[cluster_idx % len(standard_colors)])
        else:
            node_colors.append('gray')  # Default color index

    # Plot with matplotlib
    plt.figure(figsize=(12, 12))
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        cmap=plt.cm.tab10,
        node_size=300,
        with_labels=True,
        font_weight='bold',
        edge_color='gray',
        alpha=0.8
    )

    plt.title('Pattern Similarity Network (colored by cluster)')
    plt.axis('off')
    plt.savefig(os.path.join(network_dir, 'pattern_similarity_network.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create interactive version with Plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Create color mapping - use standard Plotly colors
    colors = px.colors.qualitative.Plotly[:n_clusters]
    while len(colors) < n_clusters:  # Ensure we have enough colors
        colors.extend(px.colors.qualitative.Set1[:min(9, n_clusters - len(colors))])

    node_color_map = {c: colors[i % len(colors)] for i, c in enumerate(unique_clusters)}

    # Get colors for nodes with proper fallback
    node_colors = []
    for node in G.nodes():
        if node in pattern_clusters:
            cluster = pattern_clusters[node]
            if cluster in node_color_map:
                node_colors.append(node_color_map[cluster])
            else:
                node_colors.append('rgb(128, 128, 128)')  # Gray fallback
        else:
            node_colors.append('rgb(128, 128, 128)')  # Gray fallback

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(node) for node in G.nodes()],
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_colors,
            size=15,
            line_width=2,
            line=dict(color='white')
        ),
        hovertext=[f'Pattern {node}<br>Cluster {pattern_clusters.get(node, "N/A")}' for node in G.nodes()]
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Pattern Similarity Network',
                        title_font=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    fig.write_html(os.path.join(network_dir, 'interactive_pattern_network.html'))

    return network_dir


#######################################################################################################################
#######################################################################################################################

def create_physical_interpretation_summary(clustering_results, output_dir):
    """
    Create a summary document with physical interpretations of clusters

    Parameters:
    -----------
    clustering_results : dict
        Results from cluster_som_patterns
    output_dir : str
        Directory to save output
    """

    # Create summary directory
    summary_dir = os.path.join(output_dir, 'cluster_interpretation')
    os.makedirs(summary_dir, exist_ok=True)

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]
    n_clusters = clustering_results["n_clusters"]

    # Group patterns by cluster
    cluster_patterns = {}
    for pattern_idx, cluster in pattern_clusters.items():
        if cluster not in cluster_patterns:
            cluster_patterns[cluster] = []
        cluster_patterns[cluster].append(pattern_idx)

    # Create text summary file
    with open(os.path.join(summary_dir, 'cluster_interpretations.txt'), 'w') as f:
        f.write(f"SOM Pattern Cluster Interpretations\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Number of clusters: {n_clusters}\n\n")

        for cluster in sorted(cluster_patterns.keys()):
            patterns = cluster_patterns[cluster]
            f.write(f"Cluster {cluster}:\n")
            f.write(f"  Patterns: {', '.join(map(str, patterns))}\n")
            f.write(f"  Physical Interpretation:\n")
            f.write(f"    This cluster represents a distinct North Atlantic SST pattern.\n")
            f.write(f"    Key features include: [To be filled with domain knowledge]\n")
            f.write(f"    Potential relation to climate phenomena: [To be filled with domain knowledge]\n\n")

    # Create HTML summary file with placeholders for images
    with open(os.path.join(summary_dir, 'cluster_interpretations.html'), 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>SOM Pattern Cluster Interpretations</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #3498db; }}
        .cluster-box {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .pattern-list {{ 
            background-color: #f8f9fa; 
            padding: 10px; 
            border-radius: 3px;
        }}
        .interpretation {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin-top: 10px;
        }}
        img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <h1>SOM Pattern Cluster Interpretations</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Number of clusters: {n_clusters}</p>
""")

        for cluster in sorted(cluster_patterns.keys()):
            patterns = cluster_patterns[cluster]
            patterns_str = ', '.join(map(str, patterns))

            f.write(f"""
    <div class="cluster-box">
        <h2>Cluster {cluster}</h2>
        <div class="pattern-list">
            <strong>Patterns:</strong> {patterns_str}
        </div>

        <p><img src="../cluster_composites/cluster_{cluster}_composite.png" alt="Cluster {cluster} Composite"></p>

        <div class="interpretation">
            <h3>Physical Interpretation</h3>
            <p>This cluster represents a distinct North Atlantic SST pattern.</p>
            <p><strong>Key features include:</strong></p>
            <ul>
                <li>[To be filled with domain knowledge]</li>
                <li>[To be filled with domain knowledge]</li>
            </ul>
            <p><strong>Potential relation to climate phenomena:</strong></p>
            <ul>
                <li>[To be filled with domain knowledge]</li>
                <li>[To be filled with domain knowledge]</li>
            </ul>
        </div>
    </div>
""")

        f.write("""
</body>
</html>
""")

    return summary_dir


#######################################################################################################################
#######################################################################################################################

def create_climate_mode_correlations(clustering_results, output_dir):
    """
    Create document describing correlations between clusters and known climate modes

    Parameters:
    -----------
    clustering_results : dict
        Results from cluster_som_patterns
    output_dir : str
        Directory to save output
    """
    import os
    from datetime import datetime

    # Create correlations directory
    correlations_dir = os.path.join(output_dir, 'climate_correlations')
    os.makedirs(correlations_dir, exist_ok=True)

    # Define known climate modes
    climate_modes = [
        {
            'name': 'North Atlantic Oscillation (NAO)',
            'description': 'The NAO is characterized by a dipole pattern in sea level pressure between the subtropical high and the subpolar low.',
            'sst_pattern': 'The NAO+ is associated with a tripole pattern of SST anomalies: cool in the subpolar region, warm in the mid-latitudes, and cool in the subtropics.'
        },
        {
            'name': 'East Atlantic Pattern (EA)',
            'description': 'The EA is the second prominent mode of low-frequency variability over the North Atlantic, with a center near 55°N, 20-35°W and another center with opposite sign near 25-35°N, 0-10°W.',
            'sst_pattern': 'The EA+ is associated with warm SST anomalies in the central North Atlantic.'
        },
        {
            'name': 'Atlantic Multidecadal Oscillation (AMO)',
            'description': 'The AMO is a pattern of long-term variability in North Atlantic SSTs with a period of 60-80 years.',
            'sst_pattern': 'The AMO+ is characterized by basin-wide warm SST anomalies, particularly in the subpolar region.'
        },
        {
            'name': 'North Atlantic Tripole',
            'description': 'The North Atlantic Tripole is the dominant pattern of SST variability in the North Atlantic on interannual timescales.',
            'sst_pattern': 'It consists of three centers of action: one in the subpolar region, another of opposite sign in the mid-latitudes, and a third in the subtropics.'
        }
    ]

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]
    n_clusters = clustering_results["n_clusters"]

    # Group patterns by cluster
    cluster_patterns = {}
    for pattern_idx, cluster in pattern_clusters.items():
        if cluster not in cluster_patterns:
            cluster_patterns[cluster] = []
        cluster_patterns[cluster].append(pattern_idx)

    # Create text summary file
    with open(os.path.join(correlations_dir, 'climate_mode_correlations.txt'), 'w') as f:
        f.write(f"SOM Pattern Cluster Correlations with Climate Modes\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Known Climate Modes:\n")
        for mode in climate_modes:
            f.write(f"- {mode['name']}: {mode['description']}\n")
            f.write(f"  SST Pattern: {mode['sst_pattern']}\n\n")

        f.write("\nCluster Correlations:\n")
        for cluster in sorted(cluster_patterns.keys()):
            patterns = cluster_patterns[cluster]
            f.write(f"\nCluster {cluster} (Patterns: {', '.join(map(str, patterns))}):\n")

            # Here you would normally calculate actual correlations
            # This is a placeholder for demonstration
            f.write("  Potential correlations with climate modes:\n")
            f.write("  [To be calculated based on actual data]\n")
            f.write("  [Requires time series of climate indices for quantitative assessment]\n")

    # Create HTML version with more details
    with open(os.path.join(correlations_dir, 'climate_mode_correlations.html'), 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>SOM Cluster Correlations with Climate Modes</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .mode-box {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 20px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }}
        .cluster-box {{ 
            border: 1px solid #ddd; 
            padding: 15px; 
            margin-bottom: 20px;
            border-radius: 5px;
        }}
        .correlation-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .correlation-table, th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        .correlation-table th {{
            background-color: #f2f2f2;
        }}
        .correlation-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <h1>SOM Cluster Correlations with Climate Modes</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Known Climate Modes</h2>
""")

        for mode in climate_modes:
            f.write(f"""
    <div class="mode-box">
        <h3>{mode['name']}</h3>
        <p><strong>Description:</strong> {mode['description']}</p>
        <p><strong>SST Pattern:</strong> {mode['sst_pattern']}</p>
    </div>
""")

        f.write("""
    <h2>Cluster Correlations</h2>
    <p>The following table shows the correlation coefficients between each cluster and known climate modes.</p>
    <p>Note: These are placeholder values and should be replaced with actual correlation calculations.</p>

    <table class="correlation-table">
        <tr>
            <th>Cluster</th>
            <th>NAO</th>
            <th>EA</th>
            <th>AMO</th>
            <th>Tripole</th>
            <th>Most Similar Mode</th>
        </tr>
""")

        # Generate placeholder correlations

        for cluster in sorted(cluster_patterns.keys()):
            # Generate random correlations for demonstration
            corr_nao = random.uniform(-0.7, 0.7)
            corr_ea = random.uniform(-0.7, 0.7)
            corr_amo = random.uniform(-0.7, 0.7)
            corr_tripole = random.uniform(-0.7, 0.7)

            # Find highest correlation (absolute value)
            corrs = [('NAO', abs(corr_nao)), ('EA', abs(corr_ea)),
                     ('AMO', abs(corr_amo)), ('Tripole', abs(corr_tripole))]
            most_similar = max(corrs, key=lambda x: x[1])[0]

            f.write(f"""
        <tr>
            <td>Cluster {cluster}</td>
            <td>{corr_nao:.2f}</td>
            <td>{corr_ea:.2f}</td>
            <td>{corr_amo:.2f}</td>
            <td>{corr_tripole:.2f}</td>
            <td>{most_similar}</td>
        </tr>
""")

        f.write("""
    </table>

    <h2>Detailed Cluster Analysis</h2>
""")

        for cluster in sorted(cluster_patterns.keys()):
            patterns = cluster_patterns[cluster]
            f.write(f"""
    <div class="cluster-box">
        <h3>Cluster {cluster}</h3>
        <p><strong>Patterns:</strong> {', '.join(map(str, patterns))}</p>
        <img src="../cluster_composites/cluster_{cluster}_composite.png" alt="Cluster {cluster} Composite" style="max-width: 600px;">

        <h4>Correlation Analysis</h4>
        <p>This cluster shows [strong/moderate/weak] similarity to the [most similar climate mode] pattern.</p>
        <p>Key similarities include:</p>
        <ul>
            <li>[To be filled with domain knowledge]</li>
            <li>[To be filled with domain knowledge]</li>
        </ul>

        <p>Temporal correlation with index time series would provide additional evidence of relationship.</p>
    </div>
""")

        f.write("""
</body>
</html>
""")

    return correlations_dir


#######################################################################################################################
#######################################################################################################################

def analyze_pattern_transitions_with_lag(som, X, ds, max_lag=3):
    """
    Analyze transitions between SOM patterns with different time lags

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input data array
    ds : xarray.Dataset
        Original dataset with time coordinates
    max_lag : int
        Maximum lag to consider (in time steps)

    Returns:
    --------
    dict
        Dictionary containing transition matrices for different lags
    """
    # Get Best Matching Units (BMUs) for all time points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get time coordinates and sort chronologically
    times = pd.to_datetime(ds.time.values)
    sorted_indices = np.argsort(times)

    # Apply sorting
    chronological_patterns = pattern_sequence[sorted_indices]
    chronological_times = times[sorted_indices]

    # Create transition matrices for different lags
    n_patterns = som.get_weights().shape[0] * som.get_weights().shape[1]
    lag_transitions = {}

    for lag in range(1, max_lag + 1):
        # Initialize transition matrix
        transition_matrix = np.zeros((n_patterns, n_patterns))

        # Count transitions for this lag
        for i in range(len(chronological_patterns) - lag):
            from_pattern = chronological_patterns[i] - 1  # Convert to 0-based
            to_pattern = chronological_patterns[i + lag] - 1  # Convert to 0-based
            transition_matrix[from_pattern, to_pattern] += 1

        # Calculate transition probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            dec_to_jan_prob = np.divide(dec_to_jan, dec_to_jan.sum(axis=1, keepdims=True),
                                        where=dec_to_jan.sum(axis=1, keepdims=True) != 0)
            jan_to_feb_prob = np.divide(jan_to_feb, jan_to_feb.sum(axis=1, keepdims=True),
                                        where=jan_to_feb.sum(axis=1, keepdims=True) != 0)
            dec_to_feb_prob = np.divide(dec_to_feb, dec_to_feb.sum(axis=1, keepdims=True),
                                        where=dec_to_feb.sum(axis=1, keepdims=True) != 0)

        # Fill NaN values with zeros
        dec_to_jan_prob = np.nan_to_num(dec_to_jan_prob)
        jan_to_feb_prob = np.nan_to_num(jan_to_feb_prob)
        dec_to_feb_prob = np.nan_to_num(dec_to_feb_prob)

        # Store results
        lag_transitions[lag] = {
            'transition_matrix': transition_matrix,
            'transition_probabilities': transition_probabilities
        }

    return lag_transitions


#######################################################################################################################
#######################################################################################################################

def analyze_z500_lag_correlations(som, X, ds, z500_file, clustering_results, max_lag_months=3):
    """
    Analyze lagged correlations between SST patterns/clusters and Z500 fields

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array
    ds : xarray.Dataset
        Original SST dataset with coordinates
    z500_file : str
        Path to Z500 data file
    clustering_results : dict
        Results from cluster_som_patterns
    max_lag_months : int
        Maximum lag to consider (in months)

    Returns:
    --------
    dict
        Dictionary containing lagged correlations between clusters and Z500
    """
    # Load Z500 data
    print(f"Loading Z500 data from {z500_file} for lag analysis...")
    z500_ds = xr.open_dataset(z500_file)

    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]
    n_clusters = clustering_results["n_clusters"]

    # Get BMUs and cluster sequence
    bmu_indices = np.array([som.winner(x) for x in X])
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time information
    sst_times = pd.to_datetime(ds.time.values)
    z500_times = pd.to_datetime(z500_ds.valid_time.values)

    # Create DataFrame with SST clusters
    sst_df = pd.DataFrame({
        'time': sst_times,
        'year': sst_times.year,
        'month': sst_times.month,
        'cluster': cluster_sequence
    })

    # Filter to DJF months
    sst_df = sst_df[sst_df['month'].isin([12, 1, 2])]

    # Create meteorological winter year (Dec belongs to next year's winter)
    sst_df['winter_year'] = sst_df.apply(
        lambda x: x['year'] if x['month'] > 2 else x['year'] - 1, axis=1)

    # Create one-hot encoding for clusters
    for cluster in range(n_clusters):
        sst_df[f'cluster_{cluster}'] = (sst_df['cluster'] == cluster).astype(int)

    # Prepare Z500 data
    # Filter to DJF months
    z500_months = pd.DatetimeIndex(z500_times).month
    z500_years = pd.DatetimeIndex(z500_times).year

    djf_mask = np.isin(z500_months, [12, 1, 2])
    z500_djf = z500_ds.isel(valid_time=djf_mask)

    # Re-extract times after filtering
    z500_times_djf = pd.to_datetime(z500_djf.valid_time.values)
    z500_months_djf = pd.DatetimeIndex(z500_times_djf).month
    z500_years_djf = pd.DatetimeIndex(z500_times_djf).year

    # Create winter years for Z500
    z500_winter_years = np.array([
        year if month > 2 else year - 1
        for year, month in zip(z500_years_djf, z500_months_djf)
    ])

    # Add winter_year coordinate to Z500
    z500_djf = z500_djf.assign_coords(winter_year=('valid_time', z500_winter_years))

    # Create dictionary to store results
    lag_results = {}

    # For each cluster, compute lagged correlation maps
    for cluster in range(n_clusters):
        print(f"  Computing Z500 lag correlations for Cluster {cluster}...")

        # Get cluster binary indicator time series
        cluster_ts = sst_df.groupby(['winter_year', 'month'])[f'cluster_{cluster}'].mean()

        # Store correlation maps for different lags
        lag_maps = {}

        # Compute correlations for different lags
        for lag in range(-max_lag_months, max_lag_months + 1):
            print(f"    Processing lag {lag} months...")

            # Create correlation and p-value maps
            corr_map = np.zeros(z500_djf.z.isel(valid_time=0).shape)
            p_map = np.ones_like(corr_map)

            # For each grid point, compute lagged correlation
            for y in range(corr_map.shape[0]):
                for x in range(corr_map.shape[1]):
                    # Extract Z500 time series at this grid point
                    z500_ts = z500_djf.z.isel(latitude=y, longitude=x)

                    # Group by winter year and month to match SST data structure
                    z500_monthly = z500_ts.groupby('winter_year').mean('valid_time')

                    # Apply lag to either cluster or Z500 data
                    if lag < 0:
                        # Z500 leads cluster (negative lag)
                        # Cluster is shifted forward by |lag|
                        cluster_ts_lagged = cluster_ts.groupby('winter_year').mean()
                        z500_ts_lagged = z500_monthly

                        # Align by winter_year + lag
                        cluster_years = np.array(cluster_ts_lagged.index.values)
                        z500_years = z500_ts_lagged.coords[
                            'year'].values if 'year' in z500_ts_lagged.coords else np.array([])
                        common_years = sorted(set(cluster_years) & set(z500_years + abs(lag)))

                        if len(common_years) < 5:
                            continue

                        # Extract lagged series
                        cluster_values = [cluster_ts_lagged.loc[year] for year in common_years]
                        z500_values = [z500_ts_lagged.sel(winter_year=year - abs(lag)).values
                                       for year in common_years]

                    elif lag > 0:
                        # Cluster leads Z500 (positive lag)
                        # Z500 is shifted forward by lag
                        cluster_ts_lagged = cluster_ts.groupby('winter_year').mean()
                        z500_ts_lagged = z500_monthly

                        # Align by winter_year - lag
                        common_years = sorted(set(cluster_ts_lagged.index) &
                                              set(z500_ts_lagged.index.values - lag))

                        if len(common_years) < 5:
                            continue

                        # Extract lagged series
                        cluster_values = [cluster_ts_lagged.loc[year] for year in common_years]
                        z500_values = [z500_ts_lagged.sel(winter_year=year + lag).values
                                       for year in common_years]

                    else:
                        # No lag (lag = 0)
                        cluster_ts_lagged = cluster_ts.groupby('winter_year').mean()
                        z500_ts_lagged = z500_monthly

                        # Align by winter_year
                        common_years = sorted(set(cluster_ts_lagged.index) &
                                              set(z500_ts_lagged.index.values))

                        if len(common_years) < 5:
                            continue

                        # Extract series
                        cluster_values = [cluster_ts_lagged.loc[year] for year in common_years]
                        z500_values = [z500_ts_lagged.sel(winter_year=year).values
                                       for year in common_years]

                    # Calculate correlation
                    try:
                        corr, p_value = stats.pearsonr(cluster_values, z500_values)
                        corr_map[y, x] = corr
                        p_map[y, x] = p_value
                    except:
                        # In case of errors (e.g., constant values)
                        corr_map[y, x] = np.nan
                        p_map[y, x] = np.nan

            # Store maps
            lag_maps[lag] = {
                'correlation_map': corr_map,
                'p_value_map': p_map,
                'lats': z500_djf.latitude.values,
                'lons': z500_djf.longitude.values
            }

        lag_results[cluster] = lag_maps

    return lag_results


#######################################################################################################################
#######################################################################################################################

def analyze_cluster_transitions_with_lag(som, X, ds, clustering_results, max_lag=3):
    """
    Analyze transitions between clusters with different time lags

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input data array
    ds : xarray.Dataset
        Original dataset with time coordinates
    clustering_results : dict
        Results from cluster_som_patterns
    max_lag : int
        Maximum lag to consider (in time steps)

    Returns:
    --------
    dict
        Dictionary containing cluster transition matrices for different lags
    """
    # Get pattern-to-cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]
    n_clusters = clustering_results["n_clusters"]

    # Get Best Matching Units (BMUs) for all time points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Convert to cluster sequence
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates and sort chronologically
    times = pd.to_datetime(ds.time.values)
    sorted_indices = np.argsort(times)

    # Apply sorting
    chronological_clusters = cluster_sequence[sorted_indices]
    chronological_times = times[sorted_indices]

    # Create transition matrices for different lags
    lag_transitions = {}

    for lag in range(1, max_lag + 1):
        # Initialize transition matrix
        transition_matrix = np.zeros((n_clusters, n_clusters))

        # Count transitions for this lag
        for i in range(len(chronological_clusters) - lag):
            from_cluster = chronological_clusters[i]
            to_cluster = chronological_clusters[i + lag]
            transition_matrix[from_cluster, to_cluster] += 1

        # Calculate transition probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            dec_to_jan_prob = np.divide(dec_to_jan, dec_to_jan.sum(axis=1, keepdims=True),
                                        where=dec_to_jan.sum(axis=1, keepdims=True) != 0)
            jan_to_feb_prob = np.divide(jan_to_feb, jan_to_feb.sum(axis=1, keepdims=True),
                                        where=jan_to_feb.sum(axis=1, keepdims=True) != 0)
            dec_to_feb_prob = np.divide(dec_to_feb, dec_to_feb.sum(axis=1, keepdims=True),
                                        where=dec_to_feb.sum(axis=1, keepdims=True) != 0)

        # Fill NaN values with zeros
        dec_to_jan_prob = np.nan_to_num(dec_to_jan_prob)
        jan_to_feb_prob = np.nan_to_num(jan_to_feb_prob)
        dec_to_feb_prob = np.nan_to_num(dec_to_feb_prob)

        # Store results
        lag_transitions[lag] = {
            'transition_matrix': transition_matrix,
            'transition_probabilities': transition_probabilities
        }

    return lag_transitions


#######################################################################################################################
#######################################################################################################################

def calculate_lagged_correlations(cluster_occurrences, index_data, max_lag=12):
    """
    Calculate lagged correlations between cluster occurrences and a climate index

    Parameters:
    -----------
    cluster_occurrences : dict
        Results from analyze_cluster_occurrences
    index_data : numpy.ndarray
        Climate index time series data
    max_lag : int
        Maximum lag to consider (in time steps)

    Returns:
    --------
    dict
        Dictionary containing lagged correlations for each cluster
    """
    # Get cluster sequence
    cluster_sequence = cluster_occurrences['cluster_sequence']

    # Ensure the length matches
    min_length = min(len(cluster_sequence), len(index_data))
    cluster_sequence = cluster_sequence[:min_length]
    index_data = index_data[:min_length]

    # Create one-hot encoding for each cluster
    clusters = np.unique(cluster_sequence)
    cluster_onehot = {}

    for cluster in clusters:
        cluster_onehot[cluster] = (cluster_sequence == cluster).astype(int)

    # Calculate lagged correlations
    lagged_correlations = {}

    for cluster in clusters:
        lagged_correlations[cluster] = {}

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Index leads cluster
                x = index_data[:-abs(lag)]
                y = cluster_onehot[cluster][abs(lag):]
            elif lag > 0:
                # Cluster leads index
                x = index_data[lag:]
                y = cluster_onehot[cluster][:-lag]
            else:
                # No lag
                x = index_data
                y = cluster_onehot[cluster]

            # Calculate correlation
            if len(x) > 0 and len(y) > 0:
                corr, p_value = stats.pearsonr(x, y)
                lagged_correlations[cluster][lag] = {
                    'correlation': corr,
                    'p_value': p_value
                }

    return lagged_correlations


#######################################################################################################################
#######################################################################################################################

def analyze_z500_lagged_response(som, X, ds, z500_data, clustering_results, lags=[0, 1, 2]):
    """
    Analyze lagged response of Z500 fields to SST patterns/clusters

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input SST data array
    ds : xarray.Dataset
        Original SST dataset with coordinates
    z500_data : xarray.Dataset
        Z500 geopotential height data
    clustering_results : dict
        Results from cluster_som_patterns
    lags : list
        List of lags to analyze (in months)

    Returns:
    --------
    dict
        Dictionary containing lagged Z500 composites for each cluster
    """
    pattern_clusters = clustering_results["pattern_clusters"]
    n_clusters = clustering_results["n_clusters"]

    # Get cluster sequence from SST data
    bmu_indices = np.array([som.winner(x) for x in X])
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])
    cluster_sequence = np.array([pattern_clusters[p] for p in pattern_sequence])

    # Get time coordinates and convert to pandas DatetimeIndex
    sst_times = pd.to_datetime(ds.time.values)
    z500_times = pd.to_datetime(z500_data.valid_time.values)

    # Calculate z500 climatology
    z500_climatology = z500_data.z.mean('valid_time')

    # Prepare results storage
    lagged_z500 = {cluster: {} for cluster in range(n_clusters)}

    # For each lag, create Z500 composites
    for lag in lags:
        # For each cluster
        for cluster in range(n_clusters):
            # Find occurrences of this cluster
            cluster_occurs = np.where(cluster_sequence == cluster)[0]

            if len(cluster_occurs) == 0:
                continue

            # For each occurrence, find the corresponding lagged Z500 field
            lagged_z500_fields = []

            for idx in cluster_occurs:
                sst_time = sst_times[idx]
                lagged_time = sst_time + pd.DateOffset(months=lag)

                # Find closest Z500 time
                time_diffs = np.abs(z500_times - lagged_time)
                closest_idx = np.argmin(time_diffs)

                # Only include if within a threshold (e.g., 5 days)
                if time_diffs[closest_idx].days <= 5:
                    z500_field = z500_data.z.isel(valid_time=closest_idx)
                    lagged_z500_fields.append(z500_field)

            # Create composite if we have enough fields
            if len(lagged_z500_fields) >= 3:
                z500_composite = sum(lagged_z500_fields) / len(lagged_z500_fields)
                z500_anomaly = z500_composite - z500_climatology

                lagged_z500[cluster][lag] = {
                    'composite': z500_composite,
                    'anomaly': z500_anomaly,
                    'n_samples': len(lagged_z500_fields)
                }

    return lagged_z500


#######################################################################################################################
#######################################################################################################################

def plot_lagged_correlations(lagged_correlations, output_dir):
    """
    Plot lagged correlations between clusters and climate indices

    Parameters:
    -----------
    lagged_correlations : dict
        Results from calculate_lagged_correlations
    output_dir : str
        Directory to save output plots
    """
    # Create output directory
    lag_dir = os.path.join(output_dir, 'lag_correlations')
    os.makedirs(lag_dir, exist_ok=True)

    # Get clusters and lags
    clusters = sorted(lagged_correlations.keys())
    lags = sorted(lagged_correlations[clusters[0]].keys())

    # Create a separate plot for each cluster
    for cluster in clusters:
        plt.figure(figsize=(10, 6))

        # Get correlation and p-value for each lag
        correlations = [lagged_correlations[cluster][lag]['correlation'] for lag in lags]
        p_values = [lagged_correlations[cluster][lag]['p_value'] for lag in lags]
        significant = [p < 0.05 for p in p_values]

        # Create bars with different colors for significant correlations
        colors = ['darkblue' if sig else 'lightblue' for sig in significant]

        plt.bar(lags, correlations, color=colors)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Add significance markers
        for i, (lag, corr, sig) in enumerate(zip(lags, correlations, significant)):
            if sig:
                plt.text(lag, corr + 0.02 * np.sign(corr), '*',
                         ha='center', fontweight='bold')

        plt.xlabel('Lag (months)')
        plt.ylabel('Correlation Coefficient')
        plt.title(f'Lagged Correlations for Cluster {cluster}')
        plt.grid(alpha=0.3)

        # Add explanation of lags
        plt.figtext(0.5, 0.01,
                    "Negative lags: Index leads Cluster, Positive lags: Cluster leads Index",
                    ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(lag_dir, f'cluster_{cluster}_lagged_correlations.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create a combined plot with all clusters
    plt.figure(figsize=(12, 8))

    for cluster in clusters:
        correlations = [lagged_correlations[cluster][lag]['correlation'] for lag in lags]
        plt.plot(lags, correlations, marker='o', linewidth=2, label=f'Cluster {cluster}')

    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Lag (months)')
    plt.ylabel('Correlation Coefficient')
    plt.title('Lagged Correlations for All Clusters')
    plt.grid(alpha=0.3)
    plt.legend()

    # Add significance threshold line
    n_samples = 100  # This should be the actual number of time points
    sig_threshold = 2 / np.sqrt(n_samples)  # Approximate significance threshold
    plt.axhline(y=sig_threshold, color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=-sig_threshold, color='r', linestyle='--', alpha=0.3)
    plt.text(lags[0], sig_threshold, 'p=0.05', color='r', va='bottom')

    plt.figtext(0.5, 0.01,
                "Negative lags: Index leads Cluster, Positive lags: Cluster leads Index",
                ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(lag_dir, 'all_clusters_lagged_correlations.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


#######################################################################################################################
#######################################################################################################################

def analyze_within_winter_transitions(som, X, ds):
    """
    Analyze transitions within the same winter season
    """
    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    months = times.month
    years = times.year

    # Create meteorological winter year (Dec belongs to next year's winter)
    winter_years = np.array([year if month > 2 else year - 1 for year, month in zip(years, months)])

    # Initialize transition matrices
    n_patterns = som.get_weights().shape[0] * som.get_weights().shape[1]
    dec_to_jan = np.zeros((n_patterns, n_patterns))
    jan_to_feb = np.zeros((n_patterns, n_patterns))
    dec_to_feb = np.zeros((n_patterns, n_patterns))

    # Group by winter year
    unique_winters = np.unique(winter_years)

    for winter in unique_winters:
        # Get patterns for this winter
        winter_mask = (winter_years == winter)
        winter_patterns = pattern_sequence[winter_mask]
        winter_months = months[winter_mask]

        # Sort by month (Dec, Jan, Feb)
        month_order = np.argsort(winter_months)
        ordered_patterns = winter_patterns[month_order]
        ordered_months = winter_months[month_order]

        # Only process if we have all three months
        if len(ordered_patterns) >= 3 and set(ordered_months) == {12, 1, 2}:
            # Find the indexes for each month
            dec_idx = np.where(ordered_months == 12)[0][0]
            jan_idx = np.where(ordered_months == 1)[0][0]
            feb_idx = np.where(ordered_months == 2)[0][0]

            # Get patterns
            dec_pattern = ordered_patterns[dec_idx] - 1  # Convert to 0-based
            jan_pattern = ordered_patterns[jan_idx] - 1
            feb_pattern = ordered_patterns[feb_idx] - 1

            # Count transitions
            dec_to_jan[dec_pattern, jan_pattern] += 1
            jan_to_feb[jan_pattern, feb_pattern] += 1
            dec_to_feb[dec_pattern, feb_pattern] += 1

    # Calculate transition probabilities
    dec_to_jan_prob = dec_to_jan / dec_to_jan.sum(axis=1, keepdims=True)
    jan_to_feb_prob = jan_to_feb / jan_to_feb.sum(axis=1, keepdims=True)
    dec_to_feb_prob = dec_to_feb / dec_to_feb.sum(axis=1, keepdims=True)

    # Handle NaN values from division by zero
    dec_to_jan_prob = np.nan_to_num(dec_to_jan_prob)
    jan_to_feb_prob = np.nan_to_num(jan_to_feb_prob)
    dec_to_feb_prob = np.nan_to_num(dec_to_feb_prob)

    return {
        'dec_to_jan': dec_to_jan_prob,
        'jan_to_feb': jan_to_feb_prob,
        'dec_to_feb': dec_to_feb_prob
    }


#######################################################################################################################
#######################################################################################################################

def analyze_year_to_year_transitions(som, X, ds, max_lag_years=3):
    """
    Analyze transitions between winter patterns across multiple years

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input data array
    ds : xarray.Dataset
        Original dataset with time coordinates
    max_lag_years : int
        Maximum number of years to look ahead
    """
    # Get BMUs and pattern sequence
    bmu_indices = np.array([som.winner(x) for x in X])
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get time info
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Create meteorological winter year
    winter_years = np.array([year if month > 2 else year - 1 for year, month in zip(years, months)])

    # For each winter, compute the dominant pattern (most frequent)
    dominant_patterns = {}
    for winter in np.unique(winter_years):
        winter_mask = (winter_years == winter)
        if np.any(winter_mask):
            winter_patterns = pattern_sequence[winter_mask]
            pattern_counts = np.bincount(winter_patterns)
            dominant_pattern = np.argmax(pattern_counts)
            dominant_patterns[winter] = dominant_pattern

    # Create yearly transition matrices (for each lag)
    n_patterns = som.get_weights().shape[0] * som.get_weights().shape[1]
    yearly_transitions = {}

    for lag in range(1, max_lag_years + 1):
        transition_matrix = np.zeros((n_patterns, n_patterns))

        # Count transitions between winters separated by 'lag' years
        winter_years_list = sorted(dominant_patterns.keys())

        for i, winter in enumerate(winter_years_list[:-lag]):
            from_pattern = dominant_patterns[winter] - 1  # Convert to 0-based
            if winter + lag in dominant_patterns:
                to_pattern = dominant_patterns[winter + lag] - 1
                transition_matrix[from_pattern, to_pattern] += 1

        # Calculate transition probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            dec_to_jan_prob = np.divide(dec_to_jan, dec_to_jan.sum(axis=1, keepdims=True),
                                        where=dec_to_jan.sum(axis=1, keepdims=True) != 0)
            jan_to_feb_prob = np.divide(jan_to_feb, jan_to_feb.sum(axis=1, keepdims=True),
                                        where=jan_to_feb.sum(axis=1, keepdims=True) != 0)
            dec_to_feb_prob = np.divide(dec_to_feb, dec_to_feb.sum(axis=1, keepdims=True),
                                        where=dec_to_feb.sum(axis=1, keepdims=True) != 0)

        # Fill NaN values with zeros
        dec_to_jan_prob = np.nan_to_num(dec_to_jan_prob)
        jan_to_feb_prob = np.nan_to_num(jan_to_feb_prob)
        dec_to_feb_prob = np.nan_to_num(dec_to_feb_prob)

        yearly_transitions[lag] = {
            'transition_count': transition_matrix,
            'transition_probability': transition_probs
        }

    return yearly_transitions


#######################################################################################################################
#######################################################################################################################

def analyze_within_winter_transitions(som, X, ds):
    """
    Analyze transitions within the same winter season
    """
    # Get BMUs for all data points
    bmu_indices = np.array([som.winner(x) for x in X])

    # Convert to pattern numbers
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get time coordinates
    times = pd.to_datetime(ds.time.values)
    months = times.month
    years = times.year

    # Create meteorological winter year (Dec belongs to next year's winter)
    winter_years = np.array([year if month > 2 else year - 1 for year, month in zip(years, months)])

    # Initialize transition matrices
    n_patterns = som.get_weights().shape[0] * som.get_weights().shape[1]
    dec_to_jan = np.zeros((n_patterns, n_patterns))
    jan_to_feb = np.zeros((n_patterns, n_patterns))
    dec_to_feb = np.zeros((n_patterns, n_patterns))

    # Group by winter year
    unique_winters = np.unique(winter_years)

    for winter in unique_winters:
        # Get patterns for this winter
        winter_mask = (winter_years == winter)
        winter_patterns = pattern_sequence[winter_mask]
        winter_months = months[winter_mask]

        # Sort by month (Dec, Jan, Feb)
        month_order = np.argsort(winter_months)
        ordered_patterns = winter_patterns[month_order]
        ordered_months = winter_months[month_order]

        # Only process if we have all three months
        if len(ordered_patterns) >= 3 and set(ordered_months) == {12, 1, 2}:
            # Find the indexes for each month
            dec_idx = np.where(ordered_months == 12)[0][0]
            jan_idx = np.where(ordered_months == 1)[0][0]
            feb_idx = np.where(ordered_months == 2)[0][0]

            # Get patterns
            dec_pattern = ordered_patterns[dec_idx] - 1  # Convert to 0-based
            jan_pattern = ordered_patterns[jan_idx] - 1
            feb_pattern = ordered_patterns[feb_idx] - 1

            # Count transitions
            dec_to_jan[dec_pattern, jan_pattern] += 1
            jan_to_feb[jan_pattern, feb_pattern] += 1
            dec_to_feb[dec_pattern, feb_pattern] += 1

    # Calculate transition probabilities
    dec_to_jan_prob = dec_to_jan / dec_to_jan.sum(axis=1, keepdims=True)
    jan_to_feb_prob = jan_to_feb / jan_to_feb.sum(axis=1, keepdims=True)
    dec_to_feb_prob = dec_to_feb / dec_to_feb.sum(axis=1, keepdims=True)

    # Handle NaN values from division by zero
    dec_to_jan_prob = np.nan_to_num(dec_to_jan_prob)
    jan_to_feb_prob = np.nan_to_num(jan_to_feb_prob)
    dec_to_feb_prob = np.nan_to_num(dec_to_feb_prob)

    return {
        'dec_to_jan': dec_to_jan_prob,
        'jan_to_feb': jan_to_feb_prob,
        'dec_to_feb': dec_to_feb_prob
    }


#######################################################################################################################
#######################################################################################################################

def analyze_year_to_year_transitions(som, X, ds, max_lag_years=3):
    """
    Analyze transitions between winter patterns across multiple years

    Parameters:
    -----------
    som : MiniSom
        Trained SOM model
    X : numpy.ndarray
        Input data array
    ds : xarray.Dataset
        Original dataset with time coordinates
    max_lag_years : int
        Maximum number of years to look ahead
    """
    # Get BMUs and pattern sequence
    bmu_indices = np.array([som.winner(x) for x in X])
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Get time info
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Create meteorological winter year
    winter_years = np.array([year if month > 2 else year - 1 for year, month in zip(years, months)])

    # For each winter, compute the dominant pattern (most frequent)
    dominant_patterns = {}
    for winter in np.unique(winter_years):
        winter_mask = (winter_years == winter)
        if np.any(winter_mask):
            winter_patterns = pattern_sequence[winter_mask]
            pattern_counts = np.bincount(winter_patterns)
            dominant_pattern = np.argmax(pattern_counts)
            dominant_patterns[winter] = dominant_pattern

    # Create yearly transition matrices (for each lag)
    n_patterns = som.get_weights().shape[0] * som.get_weights().shape[1]
    yearly_transitions = {}

    for lag in range(1, max_lag_years + 1):
        transition_matrix = np.zeros((n_patterns, n_patterns))

        # Count transitions between winters separated by 'lag' years
        winter_years_list = sorted(dominant_patterns.keys())

        for i, winter in enumerate(winter_years_list[:-lag]):
            from_pattern = dominant_patterns[winter] - 1  # Convert to 0-based
            if winter + lag in dominant_patterns:
                to_pattern = dominant_patterns[winter + lag] - 1
                transition_matrix[from_pattern, to_pattern] += 1

        # Calculate transition probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            transition_probs = np.divide(transition_matrix, row_sums,
                                         where=row_sums != 0)

        # Fill NaN values with zeros
        transition_probs = np.nan_to_num(transition_probs)

        yearly_transitions[lag] = {
            'transition_count': transition_matrix,
            'transition_probability': transition_probs
        }

    return yearly_transitions


#######################################################################################################################
#######################################################################################################################

def calculate_seasonal_index_correlations(cluster_occurrences, climate_index, lag_months=[-2, -1, 0, 1, 2]):
    """
    Calculate correlations between cluster occurrences and climate index
    with various lags, specifically for DJF data

    Parameters:
    -----------
    cluster_occurrences : dict
        Results from analyze_cluster_occurrences
    climate_index : dict
        Dictionary with 'time' and 'value' keys for the climate index
    lag_months : list
        List of lags to consider (negative: index leads, positive: cluster leads)
    """
    # Convert clusters to DataFrame for easier manipulation
    cluster_sequence = cluster_occurrences['cluster_sequence']
    times = pd.to_datetime(cluster_occurrences['times'])

    cluster_df = pd.DataFrame({
        'time': times,
        'year': times.year,
        'month': times.month,
        'cluster': cluster_sequence
    })

    # Convert index to DataFrame
    index_df = pd.DataFrame({
        'time': pd.to_datetime(climate_index['time']),
        'value': climate_index['value']
    })
    index_df['year'] = index_df['time'].dt.year
    index_df['month'] = index_df['time'].dt.month

    # Filter both to DJF months
    cluster_df = cluster_df[cluster_df['month'].isin([12, 1, 2])]
    index_df = index_df[index_df['month'].isin([12, 1, 2])]

    # Set winter year (Dec belongs to next year's winter)
    cluster_df['winter_year'] = cluster_df.apply(
        lambda x: x['year'] if x['month'] > 2 else x['year'] - 1, axis=1)
    index_df['winter_year'] = index_df.apply(
        lambda x: x['year'] if x['month'] > 2 else x['year'] - 1, axis=1)

    # Get unique clusters
    clusters = sorted(cluster_df['cluster'].unique())

    # Initialize results
    correlations = {}

    for cluster in clusters:
        correlations[cluster] = {}

        # Create binary indicator for this cluster
        cluster_df[f'is_cluster_{cluster}'] = (cluster_df['cluster'] == cluster).astype(int)

        # Group by year-month for proper time sequencing
        cluster_monthly = cluster_df.groupby(['winter_year', 'month'])[f'is_cluster_{cluster}'].mean().reset_index()
        cluster_monthly['ym'] = cluster_monthly['winter_year'].astype(str) + '-' + cluster_monthly['month'].astype(str)
        cluster_monthly = cluster_monthly.sort_values(['winter_year', 'month'])

        # Merge with index data for each lag
        for lag in lag_months:
            # Create shifted time keys
            shifted_df = cluster_monthly.copy()

            # Apply lag within the DJF sequence
            if lag != 0:
                # Calculate target year-month
                shifted_df['lagged_month'] = shifted_df['month']
                shifted_df['lagged_year'] = shifted_df['winter_year']

                for idx, row in shifted_df.iterrows():
                    month = row['month']
                    year = row['winter_year']

                    # Calculate new month and year
                    new_month = month + lag
                    new_year = year

                    # Adjust for month boundaries
                    if new_month < 1:  # Going back from Jan to previous year's Dec
                        new_month = 12 + new_month  # e.g., -1 from Jan = Dec
                        new_year = year - 1
                    elif new_month > 12:  # Going forward from Dec to next year's Jan
                        new_month = new_month - 12  # e.g., +1 from Dec = Jan
                        new_year = year + 1

                    # For months outside DJF, map to nearest DJF month
                    if new_month not in [12, 1, 2]:
                        if new_month < 12 and new_month > 2:
                            new_month = 2  # Map to February
                        else:
                            new_month = 12  # Map to December

                    shifted_df.at[idx, 'lagged_month'] = new_month
                    shifted_df.at[idx, 'lagged_year'] = new_year

            # Create merge key
            shifted_df['lagged_ym'] = shifted_df['lagged_year'].astype(str) + '-' + shifted_df['lagged_month'].astype(
                str)

            # Prepare index data for merge
            index_monthly = index_df.groupby(['winter_year', 'month'])['value'].mean().reset_index()
            index_monthly['ym'] = index_monthly['winter_year'].astype(str) + '-' + index_monthly['month'].astype(str)

            # Merge cluster with lagged index
            merged_df = shifted_df.merge(index_monthly, left_on='lagged_ym', right_on='ym',
                                         suffixes=('_cluster', '_index'))

            if len(merged_df) > 0:
                # Calculate correlation
                corr, p_value = stats.pearsonr(merged_df[f'is_cluster_{cluster}'], merged_df['value'])

                correlations[cluster][lag] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n_samples': len(merged_df)
                }
            else:
                correlations[cluster][lag] = {
                    'correlation': np.nan,  # Use np.nan, not numpy.nan
                    'p_value': np.nan,  # Use np.nan, not numpy.nan
                    'n_samples': 0
                }

    return correlations


#######################################################################################################################
#######################################################################################################################


def analyze_seasonal_cluster_transitions(som, X, ds, clustering_results):
    """
    Analyze transitions between clusters within and across winter seasons
    """
    # Get pattern to cluster mapping
    pattern_clusters = clustering_results["pattern_clusters"]
    n_clusters = clustering_results["n_clusters"]

    # Get BMUs and convert to clusters
    bmu_indices = np.array([som.winner(x) for x in X])
    n_cols = som.get_weights().shape[1]
    pattern_sequence = np.array([i * n_cols + j + 1 for i, j in bmu_indices])

    # Map patterns to clusters - IMPORTANT: Ensure clusters are 0-based for array indexing
    cluster_sequence = np.array(
        [pattern_clusters[p] - 1 for p in pattern_sequence])  # Subtract 1 to make 0-based for array indexing

    # Get time info
    times = pd.to_datetime(ds.time.values)
    years = times.year
    months = times.month

    # Create meteorological winter year
    winter_years = np.array([year if month > 2 else year - 1 for year, month in zip(years, months)])

    # Initialize transition matrices
    dec_to_jan = np.zeros((n_clusters, n_clusters))
    jan_to_feb = np.zeros((n_clusters, n_clusters))
    dec_to_feb = np.zeros((n_clusters, n_clusters))
    feb_to_next_dec = np.zeros((n_clusters, n_clusters))

    # Group by winter year
    unique_winters = np.unique(winter_years)

    for winter in unique_winters:
        # Get clusters for this winter
        winter_mask = (winter_years == winter)
        winter_clusters = cluster_sequence[winter_mask]
        winter_months = months[winter_mask]

        if len(winter_clusters) < 2 or len(set(winter_months)) < 2:
            continue

        # Group by month
        month_clusters = {}
        for m in [12, 1, 2]:
            month_mask = (winter_months == m)
            if np.any(month_mask):
                month_clusters[m] = winter_clusters[month_mask]

        # Process transitions within this winter
        if 12 in month_clusters and 1 in month_clusters:
            # If multiple patterns in a month, use most frequent
            dec_cluster = np.bincount(month_clusters[12]).argmax()
            jan_cluster = np.bincount(month_clusters[1]).argmax()
            # Ensure indices are within bounds
            if dec_cluster < n_clusters and jan_cluster < n_clusters:
                dec_to_jan[dec_cluster, jan_cluster] += 1

        if 1 in month_clusters and 2 in month_clusters:
            jan_cluster = np.bincount(month_clusters[1]).argmax()
            feb_cluster = np.bincount(month_clusters[2]).argmax()
            # Ensure indices are within bounds
            if jan_cluster < n_clusters and feb_cluster < n_clusters:
                jan_to_feb[jan_cluster, feb_cluster] += 1

        if 12 in month_clusters and 2 in month_clusters:
            dec_cluster = np.bincount(month_clusters[12]).argmax()
            feb_cluster = np.bincount(month_clusters[2]).argmax()
            # Ensure indices are within bounds
            if dec_cluster < n_clusters and feb_cluster < n_clusters:
                dec_to_feb[dec_cluster, feb_cluster] += 1

    # Process transitions from Feb to next Dec
    for i, winter in enumerate(unique_winters[:-1]):
        current_winter_mask = (winter_years == winter)
        next_winter_mask = (winter_years == winter + 1)

        current_months = months[current_winter_mask]
        feb_mask = (current_months == 2)

        next_months = months[next_winter_mask]
        next_dec_mask = (next_months == 12)

        if np.any(feb_mask) and np.any(next_dec_mask):
            feb_clusters = cluster_sequence[current_winter_mask][feb_mask]
            next_dec_clusters = cluster_sequence[next_winter_mask][next_dec_mask]

            if len(feb_clusters) > 0 and len(next_dec_clusters) > 0:
                feb_cluster = np.bincount(feb_clusters).argmax()
                next_dec_cluster = np.bincount(next_dec_clusters).argmax()

                # Ensure indices are within bounds
                if feb_cluster < n_clusters and next_dec_cluster < n_clusters:
                    feb_to_next_dec[feb_cluster, next_dec_cluster] += 1

    # Calculate transition probabilities
    dec_to_jan_prob = np.zeros_like(dec_to_jan)
    jan_to_feb_prob = np.zeros_like(jan_to_feb)
    dec_to_feb_prob = np.zeros_like(dec_to_feb)
    feb_to_next_dec_prob = np.zeros_like(feb_to_next_dec)

    for i in range(n_clusters):
        if dec_to_jan[i].sum() > 0:
            dec_to_jan_prob[i] = dec_to_jan[i] / dec_to_jan[i].sum()
        if jan_to_feb[i].sum() > 0:
            jan_to_feb_prob[i] = jan_to_feb[i] / jan_to_feb[i].sum()
        if dec_to_feb[i].sum() > 0:
            dec_to_feb_prob[i] = dec_to_feb[i] / dec_to_feb[i].sum()
        if feb_to_next_dec[i].sum() > 0:
            feb_to_next_dec_prob[i] = feb_to_next_dec[i] / feb_to_next_dec[i].sum()

    # When returning results, add 1 back to make clusters 1-indexed in the labels
    return {
        'dec_to_jan': {
            'counts': dec_to_jan,
            'probabilities': dec_to_jan_prob
        },
        'jan_to_feb': {
            'counts': jan_to_feb,
            'probabilities': jan_to_feb_prob
        },
        'dec_to_feb': {
            'counts': dec_to_feb,
            'probabilities': dec_to_feb_prob
        },
        'feb_to_next_dec': {
            'counts': feb_to_next_dec,
            'probabilities': feb_to_next_dec_prob
        }
    }


#######################################################################################################################
#######################################################################################################################

def plot_seasonal_transitions(seasonal_transitions, output_dir):
    """
    Visualize seasonal transition matrices between clusters

    Parameters:
    -----------
    seasonal_transitions : dict
        Results from analyze_seasonal_cluster_transitions
    output_dir : str
        Directory to save output plots
    """
    # Create output directory
    lag_dir = os.path.join(output_dir, 'seasonal_transitions')
    os.makedirs(lag_dir, exist_ok=True)

    # Define transition types and labels
    transition_types = [
        ('dec_to_jan', 'December to January'),
        ('jan_to_feb', 'January to February'),
        ('dec_to_feb', 'December to February'),
        ('feb_to_next_dec', 'February to Next December (9-month lag)')
    ]

    # Create heatmaps for each transition type
    for transition_key, transition_label in transition_types:
        transition_data = seasonal_transitions[transition_key]['probabilities']

        # Create figure
        plt.figure(figsize=(8, 7))

        # Plot heatmap
        sns.heatmap(
            transition_data,
            annot=True,
            cmap='viridis',
            vmin=0,
            vmax=1,
            fmt='.2f',
            cbar_kws={'label': 'Transition Probability'}
        )

        plt.title(f'Cluster Transition Probabilities: {transition_label}')
        plt.xlabel('To Cluster')
        plt.ylabel('From Cluster')

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(lag_dir, f'{transition_key}_transitions.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Create Sankey diagram showing DJF flow
    try:
        import plotly.graph_objects as go

        # Extract data
        dec_to_jan = seasonal_transitions['dec_to_jan']['counts']
        jan_to_feb = seasonal_transitions['jan_to_feb']['counts']

        # Create nodes for all months and clusters
        n_clusters = dec_to_jan.shape[0]

        # Node labels: Dec-C0, Dec-C1, ..., Jan-C0, Jan-C1, ..., Feb-C0, Feb-C1, ...
        node_labels = []
        for month in ['Dec', 'Jan', 'Feb']:
            for cluster in range(n_clusters):
                node_labels.append(f'{month}-C{cluster}')

        # Create source, target, and value lists for Sankey diagram
        source = []
        target = []
        value = []

        # Dec to Jan transitions
        for i in range(n_clusters):
            for j in range(n_clusters):
                if dec_to_jan[i, j] > 0:
                    source.append(i)  # Dec-Ci
                    target.append(n_clusters + j)  # Jan-Cj
                    value.append(dec_to_jan[i, j])

        # Jan to Feb transitions
        for i in range(n_clusters):
            for j in range(n_clusters):
                if jan_to_feb[i, j] > 0:
                    source.append(n_clusters + i)  # Jan-Ci
                    target.append(2 * n_clusters + j)  # Feb-Cj
                    value.append(jan_to_feb[i, j])

        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color="blue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )])

        fig.update_layout(
            title="Cluster Transitions Through DJF Season",
            font=dict(size=12)
        )

        fig.write_html(os.path.join(lag_dir, 'djf_flow_diagram.html'))

    except ImportError:
        print("Plotly not available. Skipping Sankey diagram.")

#######################################################################################################################
#######################################################################################################################

