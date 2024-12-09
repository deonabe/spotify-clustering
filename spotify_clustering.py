import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("dataset.csv")

# Preview the data
print(df.head())

# Drop unnecessary columns
df = df.drop(columns=["track_id", "artists", "album_name"])

# Handle missing values by removing rows with NaN values
df = df.dropna()

# Alternatively, you could fill missing values with the median for numerical columns
# df.fillna(df.median(), inplace=True)

# Check for any remaining missing values
df.isnull().sum()

# Select relevant features for clustering
features = df[["danceability", "energy", "loudness", "tempo", "valence"]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Compute the sum of squared distances (inertia) for different k values
inertia = []
k_range = range(1, 11)  # Testing k values from 1 to 10
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot the Elbow curve
plt.plot(k_range, inertia)
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.show()

# Apply K-Means clustering with the chosen k
kmeans = KMeans(n_clusters=3, random_state=42)  # Use the optimal k value
kmeans.fit(scaled_features)

# Add cluster labels to the dataframe
df["cluster"] = kmeans.labels_

# Reduce the dimensions to 2D for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Plot the clusters in 2D
scatter = plt.scatter(
    pca_components[:, 0], pca_components[:, 1], c=df["cluster"], cmap="viridis"
)
# Add a colorbar to indicate cluster labels
plt.colorbar(scatter, label="Cluster ID")

# Get cluster centroids in the reduced 2D space
centroids = pca.transform(kmeans.cluster_centers_)
# Plot the centroids
plt.scatter(
    centroids[:, 0], centroids[:, 1], s=200, c="red", marker="X", label="Centroids"
)
plt.legend()

# Add titles and axis labels
plt.title("PCA of Songs Clusters Based on Audio Features")
plt.xlabel("Principal Component 1 (e.g., Energy, Danceability)")
plt.ylabel("Principal Component 2 (e.g., Loudness, Valence)")


# Show the plot
plt.show()
