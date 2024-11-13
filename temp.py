import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

anime_title_input = input("Enter the name of anime: ")
# Load from CSV
df = pd.read_csv('df.csv')
feature_matrix = pd.read_csv('feature_matrix.csv')

X = feature_matrix.drop(columns=['title_english'])

# Function to find the cluster of the input anime
def find_cluster_of_anime(anime_title):
    anime_row = df[df['title_english'].str.lower() == anime_title.lower()]
    if anime_row.empty:
        return None
    anime_cluster = anime_row['Cluster'].values[0]
    return anime_cluster


# Function to recommend top 10 similar anime within the same cluster
def recommend_similar_animes(anime_title):
    anime_cluster = find_cluster_of_anime(anime_title)

    if anime_cluster is None:
        return "Anime not found in the dataset!"

    # Retrieve anime in the same cluster
    cluster_animes = df[df['Cluster'] == anime_cluster]

    # Get the genre vector of the input anime for similarity calculation
    input_anime_row = df[df['title_english'].str.lower() == anime_title.lower()]
    input_anime_index = input_anime_row.index[0]
    input_anime_vector = X.iloc[input_anime_index]

    # Calculate cosine similarity between the input anime and others in the same cluster
    cluster_animes = cluster_animes.reset_index(drop=True)
    cluster_vectors = X.iloc[cluster_animes.index]

    similarity_scores = cosine_similarity(input_anime_vector.values.reshape(1, -1), cluster_vectors).flatten()

    # Rank anime by similarity score (excluding the input anime itself)
    cluster_animes['similarity'] = similarity_scores
    cluster_animes = cluster_animes[cluster_animes['title_english'] != anime_title]
    top_similar_animes = cluster_animes.sort_values(by='similarity', ascending=False).head(10)
    top_similar_animes = top_similar_animes.reset_index(drop = True)

    return top_similar_animes[['title_english', 'similarity']]


# Example usage
recommended_animes = recommend_similar_animes(anime_title_input)

print("Suggesting 10 animes with the similar genres: ")
if isinstance(recommended_animes, pd.DataFrame):
    print(recommended_animes)
else:
    print(recommended_animes)

print("")
print("")
print("")
print("")
print("")




















# Predicting based on Synopsis by distilBERT Model and Synopsis Data

# Load embeddings
embeddings = np.load("embeddings.npy")

# Load cluster labels
cluster_labels = np.load("cluster_labels.npy")

# If you need the DataFrame back with cluster labels
df_synopsis = pd.read_csv("anime_synopsis_with_clusters.csv")

# Find the unique clusters
unique_clusters = np.unique(cluster_labels)

# Calculate the centroid for each cluster
centroids = []
for cluster in unique_clusters:
    cluster_embeddings = embeddings[cluster_labels == cluster]
    centroid = np.mean(cluster_embeddings, axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


# Function to get embedding for a new input text
def get_single_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Mean of token embeddings
    return embedding.numpy().flatten()  # Flatten to a 1D array


# Predict function for user input
def predict_cluster(user_input, centroids):
    # Get the embedding for the user input
    user_embedding = get_single_embedding(user_input)

    # Calculate cosine similarity between user embedding and cluster centroids
    similarities = cosine_similarity([user_embedding], centroids)

    # Find the index of the closest cluster
    predicted_cluster = np.argmax(similarities)
    return predicted_cluster


def find_similar_animes_by_name(anime_name, df, embeddings, cluster_labels, centroids, num_similar=10):
    # Step 1: Retrieve the synopsis for the given anime name
    anime_row = df[df['Name'].str.lower() == anime_name.lower()]

    # Check if the anime name exists in the dataset
    if anime_row.empty:
        print(f"Anime '{anime_name}' not found in the dataset.")
        return None

    # Get the synopsis text
    synopsis_text = anime_row.iloc[0]['Synopsis']

    # Step 2: Get the embedding for the synopsis
    user_embedding = get_single_embedding(synopsis_text)

    # Step 3: Find the cluster for the user input
    similarities_to_centroids = cosine_similarity([user_embedding], centroids)
    predicted_cluster = np.argmax(similarities_to_centroids)

    # Step 4: Filter embeddings and titles in the predicted cluster
    cluster_indices = np.where(cluster_labels == predicted_cluster)[0]
    cluster_embeddings = embeddings[cluster_indices]
    cluster_animes = df.iloc[cluster_indices]

    # Step 5: Calculate similarity between the input and animes in the cluster
    similarities = cosine_similarity([user_embedding], cluster_embeddings).flatten()

    # Step 6: Get the top N most similar animes
    top_indices = similarities.argsort()[-num_similar:][::-1]
    top_similar_animes = cluster_animes.iloc[top_indices]
    top_similar_animes = top_similar_animes.copy()
    top_similar_animes.loc[:, 'Similarity'] = similarities[top_indices]
    top_similar_animes = top_similar_animes.reset_index(drop=True)

    return top_similar_animes[['Name', 'Synopsis', 'Similarity']]


top_animes = find_similar_animes_by_name(anime_title_input, df_synopsis, embeddings, cluster_labels, centroids)
if top_animes is not None:
    print("Top 10 most similar animes based on the anime plot:")
    print(top_animes)

