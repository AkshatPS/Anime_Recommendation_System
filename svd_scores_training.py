import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_excel('anime_genres_scores.xlsx')

# Step 2: Clean the data by ensuring 'score' and 'scored_by' columns are numeric
# Remove rows where 'score' or 'scored_by' are non-numeric
df = df[pd.to_numeric(df['score'], errors='coerce').notna()]  # Remove non-numeric in 'score'
df = df[pd.to_numeric(df['scored_by'], errors='coerce').notna()]  # Remove non-numeric in 'scored_by'

# Step 3: Convert 'score' and 'scored_by' columns to numeric type
df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['scored_by'] = pd.to_numeric(df['scored_by'], errors='coerce')

# Step 4: Handle missing values by filling with appropriate values (mean or median)
df['score'] = df['score'].fillna(df['score'].mean())  # Replace NaN with the mean score
df['scored_by'] = df['scored_by'].fillna(df['scored_by'].mean())  # Replace NaN with the mean 'scored_by'

df = df[['title_english', 'score', 'scored_by']]

# Step 5: Display the cleaned data
print(df.head())

# Save the cleaned data for future steps
df.to_csv('cleaned_anime_genres_scores.csv', index=False)

# Pivot the dataframe to create a matrix where rows are anime titles and columns are 'scored_by' and 'score'
user_item_matrix = df.pivot_table(index='title_english', values=['score', 'scored_by'], aggfunc='mean')

# Step 3: Display the resulting user-item matrix
print(user_item_matrix.head())

# Optional: Save the matrix to a CSV file for further use
user_item_matrix.to_csv('user_item_matrix.csv', index=True)

# Pivot the dataset to create a user-item matrix
user_item_matrix = df.pivot_table(index='title_english', columns='scored_by', values='score', aggfunc='mean')

# Check the shape again to see if we now have more features (columns)
print(user_item_matrix.shape)


n_latent_factors_range = [5, 10, 20, 30, 40, 50, 60, 100, 200, 300, 500, 700, 1000]
errors = []

for n_factors in n_latent_factors_range:
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    latent_matrix = svd.fit_transform(user_item_matrix.fillna(0))
    reconstructed_matrix = np.dot(latent_matrix, svd.components_)

    # Compute the error (e.g., RMSE)
    mse = mean_squared_error(user_item_matrix.fillna(0), reconstructed_matrix)
    rmse = np.sqrt(mse)
    errors.append(rmse)

plt.plot(n_latent_factors_range, errors, marker='o')
plt.xlabel('Number of Latent Factors')
plt.ylabel('RMSE')
plt.title('RMSE vs. Number of Latent Factors')
plt.show()

best_n_factors = n_latent_factors_range[np.argmin(errors)]
print(f"Best number of latent factors: {best_n_factors}")

# # Step 3: Apply SVD (TruncatedSVD for efficiency)
# # Set the number of components to keep (latent factors)
# n_latent_factors = 50  # You can adjust this based on your needs
#
# svd = TruncatedSVD(n_components=n_latent_factors, random_state=42)
# latent_matrix = svd.fit_transform(user_item_matrix)
#
# # Step 4: Reconstruct the user-item matrix using the latent factors
# reconstructed_matrix = np.dot(latent_matrix, svd.components_)
#
# # Step 5: Evaluate the model (using RMSE)
# # Split the data into training and testing sets
# train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)
#
# # Rebuild the train and test matrices
# train_matrix = train_data.fillna(0)
# test_matrix = test_data.fillna(0)
#
# # Predict values for the test data using the reconstructed matrix
# test_preds = reconstructed_matrix[test_matrix.index]
#
# # Compute the RMSE
# mse = mean_squared_error(test_matrix, test_preds)
# rmse = np.sqrt(mse)
# print(f"RMSE: {rmse}")
#
# # Step 6: Save the SVD model
# with open('svd_model.pkl', 'wb') as f:
#     pickle.dump(svd, f)
#
# # Step 7: Save the latent matrix for future use
# latent_matrix_df = pd.DataFrame(latent_matrix, index=user_item_matrix.index)
# latent_matrix_df.to_csv('latent_matrix.csv')
#
# # Step 8: Save the reconstructed matrix
# reconstructed_matrix_df = pd.DataFrame(reconstructed_matrix, columns=user_item_matrix.columns, index=user_item_matrix.index)
# reconstructed_matrix_df.to_csv('reconstructed_matrix.csv')