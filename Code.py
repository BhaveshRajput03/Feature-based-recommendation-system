#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset
data = {
    'User ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Genre 1 (Action)': [0.9, 0.3, 0.7, 0.6, 0.5, 0.2, 0.8, 0.4, 0.6, 0.3, 0.9, 0.1, 0.7, 0.5, 0.4, 0.6, 0.8, 0.2, 0.7, 0.3],
    'Genre 2 (Drama)': [0.3, 0.8, 0.4, 0.2, 0.6, 0.7, 0.3, 0.5, 0.4, 0.8, 0.1, 0.6, 0.2, 0.7, 0.3, 0.5, 0.2, 0.6, 0.5, 0.4],
    'Genre 3 (Comedy)': [0.2, 0.1, 0.5, 0.4, 0.3, 0.6, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.8, 0.4, 0.9, 0.2, 0.7, 0.1, 0.4, 0.6],
    'Genre 4 (Thriller)': [0.6, 0.5, 0.3, 0.8, 0.7, 0.4, 0.5, 0.3, 0.9, 0.2, 0.7, 0.3, 0.6, 0.8, 0.5, 0.7, 0.4, 0.9, 0.3, 0.2],
    'Genre 5 (Romance)': [0.4, 0.2, 0.6, 0.3, 0.5, 0.8, 0.2, 0.6, 0.7, 0.5, 0.4, 0.8, 0.1, 0.6, 0.2, 0.3, 0.5, 0.7, 0.8, 0.5]
}

# Create DataFrame from sample dataset
df = pd.DataFrame(data)
df


# In[3]:


# Function to compute user-user similarity matrix
def compute_similarity_matrix(df):
    # Extract genre columns
    genre_columns = [col for col in df.columns if col.startswith('Genre')]
    # Compute cosine similarity between users based on genre preferences
    similarity_matrix = cosine_similarity(df[genre_columns])
    return similarity_matrix

# Function to generate genre recommendations for a given user
def get_genre_recommendations(user_id, similarity_matrix, df, n=5):
    # Get similarity scores for the given user
    user_similarity_scores = similarity_matrix[user_id - 1]
    # Get indices of most similar users (excluding the user itself)
    similar_user_indices = sorted(range(len(user_similarity_scores)),
                                  key=lambda i: user_similarity_scores[i],
                                  reverse=True)[1:]
    # Get genre preferences of similar users
    similar_users_genre_preferences = df.iloc[similar_user_indices]
    # Calculate average genre preferences of similar users
    average_genre_preferences = similar_users_genre_preferences.mean(axis=0)
    # Sort genres based on average preference scores
    sorted_genres = average_genre_preferences.sort_values(ascending=False)
    # Get top N recommended genres
    top_recommendations = sorted_genres.head(n)
    return top_recommendations

# Compute user-user similarity matrix
similarity_matrix = compute_similarity_matrix(df)

# Example usage: generate genre recommendations for user 1
user_id = 1
recommendations = get_genre_recommendations(user_id,
                                            similarity_matrix, df)
print("Top 5 genre recommendations for user", user_id,
      ":", recommendations)


# In[ ]:




