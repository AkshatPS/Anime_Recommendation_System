import pandas as pd
import numpy as np
import re
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering



# Step 1: Load the data
df = pd.read_csv('anime_synopsis_data.csv')

# Step 2: Keep only the 'Name' and 'Synopsis' columns
df = df[['Name', 'Synopsis']]

# Step 3: Remove rows with null or empty values
df = df.dropna(subset=['Name', 'Synopsis'])  # Drop rows where 'Name' or 'Synopsis' are NaN
df = df[df['Synopsis'].str.strip() != '']  # Drop rows where 'Synopsis' is empty

# Step 4: Remove rows where the Synopsis is 'No description available for this anime.'
df = df[df['Synopsis'] != 'No description available for this anime.']

# Step 5: Remove unwanted symbols from the 'Synopsis' column (non-English characters, punctuation)
# Regular expression to keep only English letters, digits, spaces, and basic punctuation like periods, commas
df['Synopsis'] = df['Synopsis'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.,;?!\'\"-]', '', x))

# Step 6: Reset index after dropping rows
df = df.reset_index(drop=True)

# Step 7: Save the cleaned data to a new CSV file (optional)
df.to_csv('cleaned_anime_synopsis_data.csv', index=False)

# Show the cleaned DataFrame
print(df.head())

# Step 7: Load BERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


def get_bert_embeddings_batch(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        all_embeddings.append(batch_embeddings)

        # Print progress
        print(f"Processed batch {i // batch_size + 1}/{len(texts) // batch_size + 1}")
    return np.vstack(all_embeddings)

# Use the function with your dataset
# embeddings = get_bert_embeddings_batch(df['Synopsis'].tolist(), batch_size=100)  # Adjust batch_size as needed


# Define a range for the number of clusters
#  within-cluster sum of squares (WCSS)
# wcss = []
# for k in range(1, 30):  # Try clusters from 1 to 20
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(embeddings)  # Exclude title column
#     wcss.append(kmeans.inertia_)

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
#     cluster_labels = kmeans.fit_predict(embeddings)
#     silhouette_avg = silhouette_score(embeddings, cluster_labels)
#     silhouette_scores.append(silhouette_avg)
#
# # Plot the silhouette scores
# plt.plot(range(2, 30), silhouette_scores, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score Method')
# plt.show()

# # Save embeddings
# np.save("embeddings.npy", embeddings)
# print("Embeddings saved to embeddings.npy")

# Load embeddings
embeddings = np.load("embeddings.npy")

n_clusters = 20  # Set this to the number of clusters you'd like to try

# Initialize Agglomerative Clustering with 'metric' instead of 'affinity'
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')

# Step 2: Fit the model and predict cluster labels
# This will cluster the embeddings based on hierarchical grouping
cluster_labels = agg_clustering.fit_predict(embeddings)

# Save cluster labels
np.save("cluster_labels.npy", cluster_labels)
print("Cluster labels saved to cluster_labels.npy")

# Step 3: Attach the cluster labels to your DataFrame (optional)
df['Cluster'] = cluster_labels


# Save the DataFrame with cluster labels
df.to_csv("anime_synopsis_with_clusters.csv", index=False)
print("Cluster labels saved to anime_synopsis_with_clusters.csv")

silhouette_avg = silhouette_score(embeddings, cluster_labels)
print(f"Silhouette Score for Agglomerative Clustering with {n_clusters} clusters: {silhouette_avg}")

# Calculate the Calinski-Harabasz Index
calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
print(f"Calinski-Harabasz Index for Agglomerative Clustering with {n_clusters} clusters: {calinski_harabasz}")


# Step 4: Display some of the clustered data (optional)
print(df[['Name', 'Synopsis', 'Cluster']].head(10))
print(df.shape)