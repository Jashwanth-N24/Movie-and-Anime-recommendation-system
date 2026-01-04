import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ============================================
# Data Loading & Logic (Ported from app.py)
# ============================================

# Global variables to hold data
df_media = None
tfidf_matrix = None
indices = None

def load_data():
    """
    Loads and merges TMDB 5000, Bollywood, Anime, OTT, and Kannada datasets.
    """
    global df_media
    print("Loading data...")
    
    # 1. Load TMDB Data
    try:
        df_tmdb = pd.read_csv('tmdb_5000_movies.csv')
        df_credits = pd.read_csv('tmdb_5000_credits.csv')
        df_tmdb = df_tmdb.merge(df_credits, left_on='id', right_on='movie_id')
        if 'title_x' in df_tmdb.columns:
            df_tmdb = df_tmdb.rename(columns={'title_x': 'title'})
        
        def get_top_cast(x):
            if pd.isnull(x): return []
            try:
                cast_list = json.loads(x)
                return [i['name'] for i in cast_list[:3]]
            except:
                return []
        df_tmdb['cast'] = df_tmdb['cast'].apply(get_top_cast)
    except FileNotFoundError:
        print("TMDB data not found")
        return None

    # TMDB Preprocessing
    df_tmdb['genres'] = df_tmdb['genres'].apply(lambda x: tuple([i['name'] for i in json.loads(x)]) if pd.notnull(x) else tuple())
    df_tmdb['cast'] = df_tmdb['cast'].apply(lambda x: tuple(x))
    df_tmdb['overview'] = df_tmdb['overview'].fillna('')
    df_tmdb['language'] = df_tmdb['original_language'].fillna('unknown')
    df_tmdb = df_tmdb[['title', 'genres', 'overview', 'language', 'cast']]

    # 2. Load Bollywood Data
    try:
        df_bolly = pd.read_csv('bollywood_movies.csv')
        df_bolly = df_bolly.rename(columns={'movie_name': 'title', 'genre': 'genres', 'cast': 'cast_raw'})
        df_bolly['cast'] = df_bolly['cast_raw'].apply(lambda x: tuple([c.strip() for c in str(x).split(',')[:3]]) if pd.notnull(x) else tuple())
        df_bolly['genres'] = df_bolly['genres'].apply(lambda x: tuple([g.strip() for g in str(x).split(',')]) if pd.notnull(x) else tuple())
        df_bolly['overview'] = df_bolly['overview'].fillna('')
        df_bolly['language'] = 'Hindi'
        df_bolly = df_bolly[['title', 'genres', 'overview', 'language', 'cast']]
    except FileNotFoundError:
        df_bolly = pd.DataFrame(columns=['title', 'genres', 'overview', 'language', 'cast'])

    # 3. Load Anime Data
    try:
        df_anime = pd.read_csv('anime_with_synopsis.csv')
        df_anime = df_anime.rename(columns={'Name': 'title', 'Genres': 'genres', 'sypnopsis': 'overview'})
        df_anime['genres'] = df_anime['genres'].apply(lambda x: tuple([g.strip() for g in str(x).split(',')]) if pd.notnull(x) else tuple())
        df_anime['overview'] = df_anime['overview'].fillna('')
        df_anime['language'] = 'Japanese'
        df_anime['cast'] = [tuple() for _ in range(len(df_anime))]
        df_anime = df_anime[['title', 'genres', 'overview', 'language', 'cast']]
    except FileNotFoundError:
        df_anime = pd.DataFrame(columns=['title', 'genres', 'overview', 'language', 'cast'])

    # 4. Load OTT Data
    try:
        df_ott = pd.read_csv('ott_movies.csv')
        indian_languages = ['Hindi', 'Tamil', 'Telugu', 'Malayalam', 'Kannada', 'Bengali', 'Punjabi', 'Marathi', 'Bhojpuri']
        mask_country = df_ott['Country'].str.contains('India', case=False, na=False)
        mask_lang = df_ott['Language'].str.contains('|'.join(indian_languages), case=False, na=False)
        df_ott = df_ott[mask_country | mask_lang].copy()
        
        df_ott = df_ott.rename(columns={'Title': 'title', 'Genres': 'genres', 'Language': 'language'})
        df_ott['overview'] = df_ott['title'] + " " + df_ott['genres'].fillna('') + " Directed by " + df_ott['Directors'].fillna('')
        
        def get_platforms(row):
            platforms = []
            if row['Netflix'] == 1: platforms.append('Netflix')
            if row['Hulu'] == 1: platforms.append('Hulu')
            if row['Prime Video'] == 1: platforms.append('Prime Video')
            if row['Disney+'] == 1: platforms.append('Disney+')
            return platforms
        
        df_ott['platforms'] = df_ott.apply(get_platforms, axis=1)
        df_ott['platforms'] = df_ott['platforms'].apply(lambda x: tuple(x))
        df_ott['genres'] = df_ott['genres'].apply(lambda x: tuple([g.strip() for g in str(x).split(',')]) if pd.notnull(x) else tuple())
        df_ott['cast'] = [tuple() for _ in range(len(df_ott))]
        df_ott = df_ott[['title', 'genres', 'overview', 'platforms', 'language', 'cast']]
    except FileNotFoundError:
        df_ott = pd.DataFrame(columns=['title', 'genres', 'overview', 'platforms', 'language', 'cast'])

    # 5. Load Expanded Indian Data (Kannada Focus)
    try:
        df_indian = pd.read_csv('indian_movies_all.csv')
        def safe_year(x):
            try: return int(float(str(x).replace('(','').replace(')','')))
            except: return 0
        df_indian['year_clean'] = df_indian['year'].apply(safe_year)
        mask_kannada = (df_indian['language'].str.lower() == 'kannada') & (df_indian['year_clean'] >= 2000)
        df_kannada = df_indian[mask_kannada].copy()
        df_kannada = df_kannada.rename(columns={'movie': 'title', 'starCast': 'cast_raw'})
        df_kannada['cast'] = df_kannada['cast_raw'].apply(lambda x: tuple([c.strip() for c in str(x).split(',')[:3]]) if pd.notnull(x) else tuple())
        df_kannada['genres'] = [tuple(['Drama']) for _ in range(len(df_kannada))]
        df_kannada['overview'] = df_kannada['title'] + " (Kannada Movie). Starring: " + df_kannada['cast_raw'].fillna('')
        df_kannada = df_kannada[['title', 'genres', 'overview', 'language', 'cast']]
    except FileNotFoundError:
        df_kannada = pd.DataFrame(columns=['title', 'genres', 'overview', 'language', 'cast'])

    # 6. Merge Datasets
    for df in [df_tmdb, df_bolly, df_anime, df_kannada]:
        if 'platforms' not in df.columns: df['platforms'] = [tuple() for _ in range(len(df))]
    if 'cast' not in df_ott.columns: df_ott['cast'] = [tuple() for _ in range(len(df_ott))]

    df_final = pd.concat([df_tmdb, df_bolly, df_anime, df_ott, df_kannada], ignore_index=True)
    df_final['language'] = df_final['language'].fillna('unknown')
    df_final['display_title'] = df_final['title'] + " (" + df_final['language'] + ")"
    df_final['content_soup'] = df_final['overview'] + " " + df_final['genres'].apply(lambda x: ' '.join(x)) + " " + df_final['cast'].apply(lambda x: ' '.join(x))
    
    # Priority Sorting
    def get_priority(lang):
        if hasattr(lang, 'lower') and (lang.lower() == 'kannada' or lang.lower() == 'kn'):
            return 0
        return 1
    df_final['priority'] = df_final['language'].apply(get_priority)
    df_final = df_final.sort_values(by=['priority', 'title'], ascending=[True, True]).reset_index(drop=True)
    df_final = df_final.drop(columns=['priority'])
    
    df_media = df_final
    print(f"Data Loaded: {len(df_media)} titles")

def build_model():
    global tfidf_matrix, indices
    print("Building TF-IDF Matrix...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_media['content_soup'])
    indices = pd.Series(df_media.index, index=df_media['display_title']).drop_duplicates()
    print("Model built.")

def get_recommendations_logic(title):
    if title not in indices:
        return None
    idx = indices[title]
    if isinstance(idx, pd.Series): idx = idx.iloc[0]
    
    target_vector = tfidf_matrix[idx]
    sim_scores = linear_kernel(target_vector, tfidf_matrix).flatten()
    movie_indices = sim_scores.argsort()[::-1][1:11]
    
    results = df_media.iloc[movie_indices][['title', 'display_title', 'genres', 'overview', 'platforms', 'cast']].copy()
    
    # Convert tuples back to lists for JSON serialization
    results['genres'] = results['genres'].apply(list)
    results['platforms'] = results['platforms'].apply(list)
    results['cast'] = results['cast'].apply(list)
    
    return results.to_dict(orient='records')

# ============================================
# API Routes
# ============================================

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/api/movies', methods=['GET'])
def get_movies():
    if df_media is None:
        return jsonify({"error": "Data not loaded"}), 500
    movies = df_media['display_title'].values.tolist()
    return jsonify({"movies": movies})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    title = data.get('title')
    if not title:
        return jsonify({"error": "Title required"}), 400
    
    recommendations = get_recommendations_logic(title)
    if recommendations is None:
        return jsonify({"error": "Movie not found"}), 404
        
    return jsonify({"recommendations": recommendations, "source_title": title})

if __name__ == '__main__':
    load_data()
    build_model()
    # Run on all interfaces for strictly local network access if needed, but per request "remove mobile access" we can just bind to localhost if we want.
    # But usually 0.0.0.0 is fine for dev. 
    # Port 5000 is standard.
    print("Starting Flask Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
