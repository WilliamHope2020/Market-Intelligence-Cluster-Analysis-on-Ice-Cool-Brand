import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.metrics import silhouette_score

# Load survey data into a DataFrame (replace 'IceCool_MI.csv' with your dataset)
survey_data = pd.read_csv('IceCool_MI.csv')

# Select relevant features for clustering
selected_features = survey_data[['brand_switching', 'price_sense', 'brand_aware', 'packaging', 'taste']]

# Standardize the data (scaling to mean=0 and variance=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_features)

# Determine the number of clusters (you can use various methods like the Elbow method)
num_clusters = 3  # Adjust as needed

# Create combinations of variables for scatter plots
variables = selected_features.columns
combinations = list(combinations(variables, 2))

# Initialize a list to store silhouette scores
silhouette_scores = []

# Loop through combinations of variables
for var1, var2 in combinations:
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Calculate silhouette score
    silhouette = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    cluster_colors = sns.color_palette("hsv", n_colors=num_clusters)
    scatter = sns.scatterplot(data=survey_data, x=var1, y=var2, hue=cluster_labels, palette=cluster_colors, s=100, alpha=0.7)

    # Set plot labels and title
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f'Survey Data Clusters: {var1} vs. {var2}\nSilhouette Score: {silhouette:.2f}')

    # Customize the legend
    legend = plt.gca().get_legend()
    legend.set_title('Cluster')
    legend.get_title().set_fontsize('12')

    # Display the plot
    plt.show()

# Print silhouette scores
for i, (var1, var2) in enumerate(combinations):
    print(f'Silhouette Score for {var1} vs. {var2}: {silhouette_scores[i]:.2f}')
