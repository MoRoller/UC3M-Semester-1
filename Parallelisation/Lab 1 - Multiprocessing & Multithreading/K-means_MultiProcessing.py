from time import time
import numpy as np
import pandas as pd
from matplotlib import colormaps
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.spatial.distance import cdist
import seaborn as sns
import multiprocessing as mp

# Step 0
def get_data(df_input):
    df = pd.read_csv(df_input)
    # map non numerical data
    mapping = {'yes': 1, 'no': 0}
    df['cd'] = df['cd'].map(mapping)
    df['laptop'] = df['laptop'].map(mapping)
    # remove index column
    df = df.drop(df.columns[0], axis=1)
    return df.to_numpy()


# Step 1: Initialize centroids
# kmeans++ initialization
def initial_centroids_PP(data, k):
    np.random.seed(8)
    
    n_samples = data.shape[0]
    centroids = [data[np.random.randint(n_samples), :]]

    for _ in range(k - 1):
        # Calculate distances using broadcasting
        dist_sq = np.sum((data - centroids[-1])**2, axis=1)
        for centroid in centroids[:-1]:
            dist_sq = np.minimum(dist_sq, np.sum((data - centroid)**2, axis=1))

        centroids.append(data[np.argmax(dist_sq), :])

    return np.array(centroids)


# Step 2: assign all points to a cluster
def assign_to_centroid(centroids, data):
    # calculate distances to centroids for all datapoints
    distances = cdist(data, centroids)

    # Assign each observation to the closest centroid
    initial_clusters = np.argmin(distances,
                                 axis=1)  # for each datapint: get (column) index of min value for distance to centroid
    return initial_clusters  # output: array with closest centroid for each datarow



# Step 3: calculate new centroids as mean of the clusters
def get_new_centroids(current_clusters, data):  # inputs: array of current clusters, data (ohne cluster)
    new_centroids = []
    unique_clusters = np.unique(current_clusters)  # Find unique cluster labels

    # Iterate over each cluster
    for cluster_label in unique_clusters:
        # Filter data points belonging to the current cluster
        cluster_data = data[current_clusters == cluster_label]
        # Calculate the mean of data points in the cluster
        cluster_mean = np.mean(cluster_data, axis=0)

        # Assign mean as new centroid
        new_centroids.append(cluster_mean)

    new_centroids = np.array(new_centroids)  # convert list to array
    return new_centroids  # output: array of new clusters


# Step 4: loop until convergence:  Calculate new clusters, recalculate centroids
def get_final_clusters(centroids, data, clustering, k):
    changed_clusters = True
    i = 0
    
    # Stack clusters to data_computer
    data_mit_cluster = np.column_stack((data, np.array([None] * data.shape[0])))
    data_mit_cluster[:, -1] = clustering
    

    # exit while when clusters don't change anymore or after 500 iterations
    while changed_clusters and i < 500:
        # Store previous clusters
        clusters_0 = data_mit_cluster[:, -1].copy()

        # Calculate distance for datapoints to centroids
        distances = cdist(data_mit_cluster[:, :-1].astype(float), centroids.astype(float))
        # output: for each datapoint, distance to each centroid

        # Assign datapoints to closest centroid (centroid with minimum distance)
        updated_clusters = np.argmin(distances, axis=1)

        # Update centroids based on new cluster means
        data_mit_cluster[:, -1] = updated_clusters

        # Update centroids
        centroids = get_new_centroids(current_clusters = updated_clusters, data= data)

        # Check if clusters have changed
        changed_clusters = not all(np.equal(clusters_0, updated_clusters))

        i += 1

    print(f'k = {k}')
    print(f'No. of iterations: {i}')
    #if not changed_clusters:
    #    print("Clusters didn't change")
    #else:
    #    print('Max. number of iterations reached')

    return updated_clusters, centroids  # output: arrays containing final clusters, final centroids


# Step 5: Calculate WCSS for eac
def get_WCSS(data, clustering, centroids, k):
    WCSS = 0
    
    # Stack clusters to data_computer
    data_mit_cluster = np.column_stack((data, np.array([None] * data.shape[0])))
    data_mit_cluster[:, -1] = clustering
    
    # Calculate WCSS per cluster and sum up for all clusters
    for cluster_label in range(k):
        # Filter for each cluster_label (/wo cluster label column)
        filtered_data = data_mit_cluster[data_mit_cluster[:, -1] == cluster_label, :-1]
        
        # Get centroid of current cluster_label
        centroid = centroids[cluster_label, :]
        
        # Calculate squared distances (to centroid) for each cluster         
        dist_per_cluster = np.power(filtered_data - centroid, 2)
        dist_per_k = np.sum(dist_per_cluster)
        
        # Sum distances per k to get WCSS
        WCSS += dist_per_k
    
    print(f'WCSS = {round(WCSS / (10 ** 9), 2)} 10e9')
    return WCSS


# Plots
def elbow(WCSS_values, x_values):
    plt.plot([i for i in range(1, x_values)], WCSS_values, marker='o', c='mediumpurple')
    plt.ylabel('WCSS')
    plt.xlabel('No of Clusters')
    plt.xticks(range(1, x_values))
    #plt.savefig('Elbow_MP.png')
    plt.show()


# Plot first 2 dimensions of data
def scatter(data, centroids):
    # Extract the first two dimensions
    x = data[:, 0]
    y = data[:, 1]
    clusters = data[:, -1]
    cmap = colormaps.get_cmap('plasma')

    # Centroids
    xc = centroids[:, 0]  # values for first dimension of centroids
    xy = centroids[:, 1]  # values for second dimension of centroids

    # Create plot
    plt.scatter(x, y, c=clusters, cmap=cmap, s=10, alpha=0.1)
    plt.scatter(xc, xy, c='red', label='centroids', s=50)

    # Labeling of centroids
    for i, (cx, cy) in enumerate(zip(xc, xy)):
        plt.text(cx, cy, f'Cluster {i}', fontsize=10, ha='right')
    plt.xlabel('Price')
    plt.ylabel('Speed')
    plt.title('Price & Speed')
    plt.legend()
    #plt.savefig('Clustering_MP.png')
    plt.show()


# Heatmap
def heatmap(centroids):
    # standardize data (same shape as centroids)
    dist_stand = (centroids - np.mean(centroids, axis=0)) / np.std(centroids, axis=0)
    x_labels = [f'Cluster {i}' for i in range(0, centroids.shape[0])]
    y_labels = ['Price', 'Speed', 'HD', 'Ram', 'Screen', 'Cores', 'CD', 'Laptop', 'Trend']

    sns.heatmap(dist_stand.T, cmap='BuPu', xticklabels=x_labels, yticklabels=y_labels, annot=True, fmt='.2f')

    plt.title('Cluster centroids (standardized)')
    #plt.savefig('Heatmap_MP.png')
    plt.show()

#def run_all(args: Tuple[np.ndarray, int]) -> Tuple[float, np.ndarray, np.ndarray]:
def run_all(d, k):

    # Step 1: Initialize centroids
    centroids = initial_centroids_PP(data = d, k = k)  # output: k centroids as arrays
    
    # Step 2: assign all points to a cluster
    initial_clusters = assign_to_centroid(centroids = centroids, data = d)

    # Step 3: calculate new centroids as mean of the clusters
    centroids = get_new_centroids(current_clusters = initial_clusters,
                                  data = d)  # output: k centroids as arrays

    # Step 4: Get final clusters
    final_clusters, final_centroids = get_final_clusters(
        centroids = centroids, data = d, clustering = initial_clusters, k=k)
    
    # Step 5: Calculate WCSS for each k
    WCSS = get_WCSS(data = d, clustering = final_clusters, centroids= final_centroids, k=k)
    
    print(f'final_clusters: {np.unique(final_clusters)}') 
    print('_____________________________________________________________________')
    
    return WCSS, final_clusters, final_centroids



if __name__ == '__main__':
    np.random.seed(8)
    t0 = time()
    k_max = 8  # maximum number of clusters

    # Read data
    data_computer = get_data('computers.csv') #original file contains 1 million rows
    #data_computer = data_computer[:100000, :]

    print(f'Run with {data_computer.shape[0]} rows started!\n')
    
    
    # Multiprocessing
    # n_proc = mp.cpu_count() # number of processes = number of cores
    n_proc = min(mp.cpu_count(), k_max - 1)   # limit max number of processes to max number of k
    
    with mp.Pool(processes = n_proc) as pool:
        results = pool.starmap(
            run_all,
            [(data_computer, k) for k in range(1, k_max)]
        )
    pool.close()
    pool.join()
    
    WCSS_list, final_clusters_list, final_centroids_list = zip(*results)
    print('_____________________________________________________________________')
    
    # set ideal_k
    ideal_k = 4
    
    final_clusters = final_clusters_list[ideal_k -1]
    final_centroids = final_centroids_list[ideal_k -1]
    
    print(f'final clusters: {np.unique(final_clusters)}')
    
    # use clustering of ideal k as final result
    data_mit_cluster = np.column_stack((data_computer, np.array([None] * data_computer.shape[0])))
    data_mit_cluster[:, -1] = final_clusters
    
    
    t1 = time()
    processing_time = t1 - t0
    print('_____________________________________________________________________')
    print(f'Processing time: {round(processing_time, 2)} seconds')
    print('_____________________________________________________________________')
    
    # Plot elbow
    elbow(WCSS_values = WCSS_list, x_values = k_max)
    
    
    # Plot first two dimensions
    scatter(data = data_mit_cluster, centroids=final_centroids)

    
    # Calculate average price per cluster
    cluster_labels = np.unique(final_clusters)
    average_prices = np.zeros_like(cluster_labels)

    for cluster in cluster_labels:
        # create subset for each cluster and extract prices (col 0)
        prices = data_mit_cluster[data_mit_cluster[:, -1] == cluster][:, 0]
        average_prices[cluster_labels == cluster] = np.mean(prices)

    # Cluster with highest average price
    highest = np.argmax(average_prices)

    print(f'Cluster with highest average price: Cluster {highest}')
    print(f'Average price of Cluster {highest}: {max(average_prices)}')    

    # Heatmap
    heatmap(final_centroids)








    