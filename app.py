import streamlit as st
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ============================================
# 1. Page Config & Layout
# ============================================
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title(" JASHWANTH Movie & Anime Recommendation System")
st.markdown("""
This app recommends movies and anime based on their content (Genre + Description). 
""")

# ============================================
# 2. Data Loading & Preprocessing
# ============================================

@st.cache_data
def load_data():
    """
    Loads and merges TMDB 5000 and Bollywood datasets.
    """
    # 1. Load TMDB Data
    try:
        df_tmdb = pd.read_csv('tmdb_5000_movies.csv')
        df_credits = pd.read_csv('tmdb_5000_credits.csv')
        
        # Merge Credits
        df_tmdb = df_tmdb.merge(df_credits, left_on='id', right_on='movie_id')
        
        # If title is duplicated (title_x, title_y), keep title_x and rename it to title
        if 'title_x' in df_tmdb.columns:
            df_tmdb = df_tmdb.rename(columns={'title_x': 'title'})
        
        # Extract Cast (Top 3)
        def get_top_cast(x):
            if pd.isnull(x): return []
            try:
                cast_list = json.loads(x)
                return [i['name'] for i in cast_list[:3]]
            except:
                return []
                
        df_tmdb['cast'] = df_tmdb['cast'].apply(get_top_cast)
        
    except FileNotFoundError:
        return None

    # TMDB Preprocessing
    # Convert to tuples for caching compatibility
    df_tmdb['genres'] = df_tmdb['genres'].apply(lambda x: tuple([i['name'] for i in json.loads(x)]) if pd.notnull(x) else tuple())
    df_tmdb['cast'] = df_tmdb['cast'].apply(lambda x: tuple(x)) # Convert to tuple
    df_tmdb['overview'] = df_tmdb['overview'].fillna('')
    df_tmdb['language'] = df_tmdb['original_language'].fillna('unknown')
    df_tmdb = df_tmdb[['title', 'genres', 'overview', 'language', 'cast']]

    # 2. Load Bollywood Data
    try:
        df_bolly = pd.read_csv('bollywood_movies.csv')
        df_bolly = df_bolly.rename(columns={'movie_name': 'title', 'genre': 'genres', 'cast': 'cast_raw'})
        
        # Process Cast
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
        df_anime['cast'] = [tuple() for _ in range(len(df_anime))] # No cast info
        
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
        df_ott['platforms'] = df_ott['platforms'].apply(lambda x: tuple(x)) # Convert to tuple for hashing
        
        # Preprocessing
        df_ott['genres'] = df_ott['genres'].apply(lambda x: tuple([g.strip() for g in str(x).split(',')]) if pd.notnull(x) else tuple())
        df_ott['cast'] = [tuple() for _ in range(len(df_ott))] # No cast info in this dataset
        
        df_ott = df_ott[['title', 'genres', 'overview', 'platforms', 'language', 'cast']]
        
    except FileNotFoundError:
        df_ott = pd.DataFrame(columns=['title', 'genres', 'overview', 'platforms', 'language', 'cast'])

    # 5. Load Expanded Indian Data (Kannada Focus)
    try:
        df_indian = pd.read_csv('indian_movies_all.csv')
        # Columns: movie, year, rating, director, starCast, crew, language
        
        # Filter for Kannada 2000-2025
        # Also clean year column if needed, but for now assuming it's roughly numeric
        def safe_year(x):
            try: return int(float(str(x).replace('(','').replace(')','')))
            except: return 0
            
        df_indian['year_clean'] = df_indian['year'].apply(safe_year)
        
        # Filter: Language=Kannada AND Year>=2000
        # You can expand 'languages' list here if you want more
        mask_kannada = (df_indian['language'].str.lower() == 'kannada') & (df_indian['year_clean'] >= 2000)
        df_kannada = df_indian[mask_kannada].copy()
        
        df_kannada = df_kannada.rename(columns={'movie': 'title', 'starCast': 'cast_raw'})
        
        # Process Cast
        df_kannada['cast'] = df_kannada['cast_raw'].apply(lambda x: tuple([c.strip() for c in str(x).split(',')[:3]]) if pd.notnull(x) else tuple())
        
        # Create mock genres/overview
        df_kannada['genres'] = [tuple(['Drama']) for _ in range(len(df_kannada))] # Default genre
        df_kannada['overview'] = df_kannada['title'] + " (Kannada Movie). Starring: " + df_kannada['cast_raw'].fillna('')
        
        df_kannada = df_kannada[['title', 'genres', 'overview', 'language', 'cast']]
        
    except FileNotFoundError:
         df_kannada = pd.DataFrame(columns=['title', 'genres', 'overview', 'language', 'cast'])


    # 6. Merge Datasets
    # Ensure all dfs have 'platforms' column
    for df in [df_tmdb, df_bolly, df_anime, df_kannada]:
        if 'platforms' not in df.columns: df['platforms'] = [tuple() for _ in range(len(df))]
    
    # Ensure all dfs have 'cast' column
    if 'cast' not in df_ott.columns: df_ott['cast'] = [tuple() for _ in range(len(df_ott))]

    df_final = pd.concat([df_tmdb, df_bolly, df_anime, df_ott, df_kannada], ignore_index=True)
    
    # Fill missing languages
    df_final['language'] = df_final['language'].fillna('unknown')
    
    # Create Display Title: Title (Language)
    df_final['display_title'] = df_final['title'] + " (" + df_final['language'] + ")"
    
    # Create content soup - Include Cast in soup!
    df_final['content_soup'] = df_final['overview'] + " " + df_final['genres'].apply(lambda x: ' '.join(x)) + " " + df_final['cast'].apply(lambda x: ' '.join(x))
    
    # Priority Sorting: Kannada movies first
    # Create a priority column: 0 for Kannada, 1 for everything else
    def get_priority(lang):
        if hasattr(lang, 'lower') and (lang.lower() == 'kannada' or lang.lower() == 'kn'):
            return 0
        return 1
        
    df_final['priority'] = df_final['language'].apply(get_priority)
    
    # Sort by priority (ascending) and then by title (ascending)
    df_final = df_final.sort_values(by=['priority', 'title'], ascending=[True, True]).reset_index(drop=True)
    
    # Drop the temporary priority column
    df_final = df_final.drop(columns=['priority'])
    
    return df_final

with st.spinner('Loading and merging datasets...'):
    df_media = load_data()

if df_media is None or df_media.empty:
    st.error("No data found! Please ensure 'tmdb_5000_movies.csv' is present.")
    st.stop()

# ============================================
# 3. Model Building (TF-IDF)
# ============================================

@st.cache_data
def build_tfidf_matrix(df):
    """
    Computes the TF-IDF matrix.
    """
    # Initialize TF-IDF Vectorizer
    # stop_words='english' removes common words (the, a, is...)
    # max_features=5000 limits the vocabulary to top 5k words to save memory
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

    # Construct the TF-IDF Matrix
    tfidf_matrix = tfidf.fit_transform(df['content_soup'])
    
    return tfidf_matrix

with st.spinner('Computing TF-IDF matrix...'):
    tfidf_matrix = build_tfidf_matrix(df_media)

# Create a reverse mapping of indices and movie titles
# Use display_title for indexing
indices = pd.Series(df_media.index, index=df_media['display_title']).drop_duplicates()


# ============================================
# 4. Recommendation Function
# ============================================

def get_content_recommendations(title):
    """
    Get top 10 recommendations based on cosine similarity.
    Computes similarity on-the-fly to save memory.
    """
    if title not in indices:
        return None
    
    # Get index
    idx = indices[title]
    
    # Handle duplicate titles
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    # Get the vector for the selected movie
    target_vector = tfidf_matrix[idx]
    
    # Compute cosine similarity between target and all other movies
    # linear_kernel(X, Y) computes X dot Y^T
    # This results in a (1, N) array of scores
    sim_scores = linear_kernel(target_vector, tfidf_matrix).flatten()

    # Get indices of top 11 scores (1st is the movie itself)
    # unexpected speed up: generic argsort is faster than sorting list of tuples
    # we use -sim_scores to sort descending
    movie_indices = sim_scores.argsort()[::-1][1:11]
    
    # Return the dataframe of recommendations
    return df_media[['title', 'display_title', 'genres', 'overview', 'platforms', 'cast']].iloc[movie_indices]


# ============================================
# 5. UI Interaction
# ============================================

# Sidebar: Mobile Sharing
st.sidebar.title("📱 Mobile Access")
import socket
try:
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    # If the detected IP is localhost loopback, try a different method
    if local_ip.startswith("127."):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
except:
    local_ip = "localhost"

# Use the current port (default 8501, but we should try to detect or user can verify)
# Since we can't easily detect the port from within the script reliably without Streamlit server object hacks,
# We will show the most likely URL.
url = f"http://{local_ip}:8508" # Assuming port 8508 based on usage, or 8501 default
st.sidebar.markdown(f"**Scan to open on mobile:**")
st.sidebar.image(f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={url}")
st.sidebar.markdown(f"[`{url}`]({url})")
st.sidebar.info("Ensure both devices are on the same Wi-Fi.")

st.subheader("Find a Movie")
# Use display_title for the list
movie_list = df_media['display_title'].values
selected_movie = st.selectbox("Select a movie you like:", movie_list)

if st.button("Recommended Movies"):
    recommendations = get_content_recommendations(selected_movie)
    
    if recommendations is not None:
        st.success(f"Because you liked **{selected_movie}**:")
        
        for i, (index, row) in enumerate(recommendations.iterrows()):
            with st.expander(f"{i+1}. {row['display_title']}"):
                # Display Genres
                st.write(f"**Genres:** {', '.join(row['genres'])}")
                
                # Display Cast
                if row['cast']:
                    st.write(f"**Cast:** {', '.join(row['cast'])}")
                
                # Display OTT Badges
                platforms = row['platforms']
                if platforms:
                    badges = []
                    for p in platforms:
                        if p == "Netflix": badges.append("🔴 Netflix")
                        elif p == "Prime Video": badges.append("🔵 Prime Video")
                        elif p == "Disney+": badges.append("🟣 Disney+")
                        elif p == "Hulu": badges.append("🟢 Hulu")
                        else: badges.append(p)
                    st.markdown(f"**Available on:** {'  '.join(badges)}")
                else:
                    st.caption("Streaming availability info not found.")
                
                st.write(f"**Overview:** {row['overview']}")
    else:
        st.error("Movie not found!")

# Sidebar Info
st.sidebar.title("About")
st.sidebar.info(
    "This recommender system uses **TF-IDF Vectorization** "
    "and **Cosine Similarity** to find movies with similar plot descriptions and genres."
)
st.sidebar.metric("Total Titles", len(df_media))
