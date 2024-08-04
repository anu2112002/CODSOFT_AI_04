import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample user-item interaction data
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4],
    'movie_id': [1, 2, 3, 1, 4, 2, 3, 4, 1],
    'rating': [5, 4, 3, 4, 5, 2, 5, 4, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Pivot the DataFrame to create a user-item matrix
user_movie_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
print("User-Item Matrix:")
print(user_movie_matrix)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
print("\nUser Similarity Matrix:")
print(user_similarity_df)

def get_user_recommendations(user_id, user_movie_matrix, user_similarity_df):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    user_ratings = user_movie_matrix.loc[user_id]
    
    recommendations = pd.Series(dtype=float)
    for similar_user in similar_users:
        similar_user_ratings = user_movie_matrix.loc[similar_user]
        for movie in similar_user_ratings.index:
            if user_ratings[movie] == 0 and similar_user_ratings[movie] > 0:
                if movie not in recommendations:
                    recommendations[movie] = 0
                recommendations[movie] += similar_user_ratings[movie] * user_similarity_df.loc[user_id, similar_user]

    recommendations = recommendations.sort_values(ascending=False)
    return recommendations

# Example: Get recommendations for user 1
user_id = 1
recommendations = get_user_recommendations(user_id, user_movie_matrix, user_similarity_df)
print(f"\nRecommendations for User {user_id}:")
print(recommendations)
