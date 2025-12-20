import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ============================================
# 1. Setup Dummy Data (Visual/Genre Focus)
# ============================================
# Imagine this is your database of media content.
data = {
    'media_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'title': [
        'Space Explorers', 
        'Medieval Knights', 
        'Future City A.I.', 
        'The Funny Cop', 
        'Galactic Wars', 
        'Castle Siege', 
        'Stand-up Special', 
        'Cyberpunk Detective'
    ],
    # Genres are crucial for visual media recommendations
    'genre': [
        'Sci-Fi Action', 
        'Action Historical', 
        'Sci-Fi Thriller', 
        'Comedy Action', 
        'Sci-Fi Action Adventure', 
        'Historical Drama', 
        'Comedy', 
        'Sci-Fi Mystery Neo-Noir'
    ],
    # Descriptions add depth beyond just simple genre tags
    'description': [
        'Brave astronauts travel to distant planets to fight alien threats with advanced ships.',
        'Knights defend a castle in the middle ages using swords and shields.',
        'Artificial intelligence takes over a futuristic metropolis, threatening humanity.',
        'A hilarious cop tries to solve a major crime while cracking nonstop jokes.',
        'Epic large-scale battles in deep space featuring massive spaceships and laser guns.',
        'A gritty, dramatic retelling of a famous historical castle siege and the people inside.',
        'Just pure jokes, laughs, and stand-up routines on a stage.',
        'A lone detective solves gritty crimes in a dark, rainy, neon-lit future city.'
    ]
}

# Load into a Pandas DataFrame
df_media = pd.DataFrame(data)

print("--- Original Database ---")
print(df_media[['title', 'genre']].head())
print("-------------------------\n")


# ============================================
# 2. Feature Engineering (Creating the "Content Profile")
# ============================================

# To get the best recommendation, we combine important text features.
# We create a "soup" containing both the genre tags and the descriptive text.
df_media['content_soup'] = df_media['genre'] + " " + df_media['description']

# Initialize TF-IDF Vectorizer.
# It converts text into a matrix of numbers.
# 'stop_words="english"' removes common words like "the", "a", "is" that don't add meaning.
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF Matrix.
# Rows = Movies, Columns = Unique words found in the content_soup.
tfidf_matrix = tfidf.fit_transform(df_media['content_soup'])

# Output for understanding:
# Shape says (8, 58) -> We have 8 movies described by 58 unique meaningful words.
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")


# ============================================
# 3. Calculating Similarity (The Math Engine)
# ============================================

# We use the Linear Kernel to calculate Cosine Similarity.
# This computes similarity scores between EVERY movie pair.
# Result is an 8x8 matrix where cell [i][j] is the similarity score between movie i and j.
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of indices and movie titles.
# This lets us quickly look up a movie ID by its title.
indices = pd.Series(df_media.index, index=df_media['title']).drop_duplicates()


# ============================================
# 4. The Recommendation Function
# ============================================

def get_content_recommendations(title, cosine_sim=cosine_sim_matrix):
    """
    Takes a movie title, finds movies with similar content profiles based on genre/description.
    """
    # 1. Get the index of the movie that matches the title
    if title not in indices:
        return "Movie title not found in database."
    
    idx = indices[title]

    # 2. Get the pairwsie similarity scores of all movies with that movie
    # This returns a list of tuples like: [(0, 1.0), (1, 0.23), (2, 0.45)...]
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 3. Sort the movies based on the similarity scores (descending order)
    # We use a lambda function to sort by the score (the second item in the tuple)
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 4. Get the scores of the top 3 most similar movies.
    # We skip index [0] because that is the movie itself (similarity score of 1.0).
    top_scores = sorted_sim_scores[1:4]

    # 5. Get the movie indices from those top scores
    movie_indices = [i[0] for i in top_scores]

    # 6. Return the titles and genres of the top recommendations
    return df_media[['title', 'genre', 'description']].iloc[movie_indices]


# ============================================
# 5. Testing the System
# ============================================

# Scenario 1: User just liked "Space Explorers" (A Sci-Fi Action movie)
# We expect other Sci-Fi movies, particularly those with action elements.
target_movie = 'Space Explorers'
print(f"--- Recommendations because you liked: '{target_movie}' ---")
recommendations_1 = get_content_recommendations(target_movie)
print(recommendations_1[['title', 'genre']])
print("\n")

# Scenario 2: User just liked "The Funny Cop" (Comedy Action)
# We expect a mix of Comedy and Action movies.
target_movie_2 = 'The Funny Cop'
print(f"--- Recommendations because you liked: '{target_movie_2}' ---")
recommendations_2 = get_content_recommendations(target_movie_2)
print(recommendations_2[['title', 'genre']])
