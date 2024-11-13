import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset
file_path = 'C:/Users/HP-PC/PycharmProjects/Anime_Recommendation_System/anime_genres_scores.xlsx'  # Replace with your actual file path if needed
df = pd.read_excel(file_path)

# Step 2: Remove rows with empty values in 'title_english' or 'genres' columns
df.dropna(subset=['title_english', 'genres'], inplace=True)

# Step 3: Inspect the data (optional)
print(df.head())

# Step 4: Clean and preprocess the genres column
# Remove any leading/trailing whitespace from genre names and split genres by commas
df['genres'] = df['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')])

# Step 5: Create the feature matrix using One-Hot Encoding
# Generate a list of all unique genres
all_genres = set([genre for sublist in df['genres'] for genre in sublist if genre])

# Initialize a DataFrame with columns for each genre, filled with zeros
for genre in all_genres:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

# Step 6: Create the feature matrix
# Drop unnecessary columns, keeping only the genre columns
genre_columns = list(all_genres)
feature_matrix = df[['title_english'] + genre_columns]
print(df.shape);
print(feature_matrix.shape)

# Print the final feature matrix
print(feature_matrix.head())




# Code to decide number of k means cluster

# # Define a range for the number of clusters
# #  within-cluster sum of squares (WCSS)
# wcss = []
# for k in range(1, 30):  # Try clusters from 1 to 20
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(feature_matrix.drop(columns=['title_english']))  # Exclude title column
#     wcss.append(kmeans.inertia_)
#
# # Plot the WCSS to find the elbow
# plt.plot(range(1, 30), wcss, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.title('Elbow Method')
# plt.show()
#
#
# silhouette_scores = []
# for k in range(2, 30):  # Avoid 1 as silhouette score requires at least two clusters
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(feature_matrix.drop(columns=['title_english']))
#     silhouette_avg = silhouette_score(feature_matrix.drop(columns=['title_english']), cluster_labels)
#     silhouette_scores.append(silhouette_avg)
#
# # Plot the silhouette scores
# plt.plot(range(2, 30), silhouette_scores, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score Method')
# plt.show()



# Separate the titles for reference and keep only the genre features for clustering
titles = feature_matrix['title_english']
X = feature_matrix.drop(columns=['title_english'])


# Step 1: Run K-Means with 15 clusters
kmeans = KMeans(n_clusters=15, random_state=42)
cluster_labels = kmeans.fit_predict(X)

df['Cluster'] = cluster_labels

# Step 2: Evaluation Results
# Calculate inertia and silhouette score for evaluation
inertia = kmeans.inertia_
silhouette_avg = silhouette_score(X, cluster_labels)

print(f'Inertia (within-cluster sum of squares): {inertia}')
print(f'Silhouette Score for 15 clusters: {silhouette_avg:.4f}')

# Step 3: Dimensionality Reduction for Visualization
# Use PCA to reduce the data to 2 dimensions for easy plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 4: Plot the clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', s=10)
plt.colorbar(scatter, label='Cluster Label')
plt.title('K-Means Clustering of Anime Genres (15 Clusters)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Add cluster labels to the original dataset for inspection
feature_matrix['Cluster'] = cluster_labels
print(feature_matrix[['title_english', 'Cluster']].head(20))

# Save the trained KMeans model to a file
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

# Save as CSV
df.to_csv('df.csv', index=False)
feature_matrix.to_csv('feature_matrix.csv', index=False)

print("Model trained and saved.")




