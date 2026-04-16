import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="1f83ddc530734ff9a7eb6cef60234908",
    client_secret="1a88624bc0ee4e97be8a713cc9b4e72f"
))

from pymongo import MongoClient
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# MongoDB connection
client = MongoClient("mongodb+srv://zrps:zeerupreeshi@cluster0.zp4isjh.mongodb.net/neurotunes?retryWrites=true&w=majority")

db = client["ai_music_db"]
collection = db["processed_tracks"]

data = list(collection.find({}, {"_id": 0}).limit(5000))
df = pd.DataFrame(data)
df = df.dropna(subset=['track_name','danceability','energy','loudness'])
df = df.reset_index(drop=True)
print(df.columns)
# print(collection.count_documents({})) //228000

#to fetch songs data
def get_song_details(song_name):
    try:
        results = sp.search(q=song_name, limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            return track['album']['images'][0]['url'], None
    except:
        return None, None

#recommender
# Precompute similarity ONCE
features = df[['danceability','energy','loudness']]
similarity = cosine_similarity(features)

def recommend(song_name, df):
    match = df[df['track_name'] == song_name]
    
    if match.empty:
        return []
    
    idx = match.index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommended = [df.iloc[i[0]]['track_name'] for i in scores[1:6]]
    
    return recommended


#streamlit
st.title("🎵 NeuroTunes - Music Recommendation System")
# Search input
song = st.text_input("🔍 Enter song name")

# Button
if st.button("Recommend"):

    if song == "":
        st.warning("Please enter a song name")
    else:
        with st.spinner("Finding similar songs..."):

            #Handle case-insensitive search
            df['track_name'] = df['track_name'].astype(str).str.lower()
            song = song.lower()
            match = df[df['track_name'].str.contains(song, case=False, na=False)]

            if match.empty:
                st.error("Song not found in dataset")
            else:
                song = match.iloc[0]['track_name']

                # Get recommendations
                results = recommend(song, df)

                st.success("Recommendations ready!")

                #Spotify API (image + preview)
                try:
                    image, preview = get_song_details(song)
                    
                    if image:
                        st.image(image, width=250)
                    
                    if preview:
                        st.audio(preview)

                except:
                    st.warning("Spotify preview not available")

                #Song features
                selected = df[df['track_name'] == song]

                if not selected.empty:
                    st.subheader("🎧 Song Features")

                    st.write("Energy")
                    st.progress(int(selected['energy'].values[0] * 100))

                    st.write("Danceability")
                    st.progress(int(selected['danceability'].values[0] * 100))

                #Recommendations
                st.subheader("🎶 Recommended Songs:")

                for i, r in enumerate(results, 1):
                    st.write(f"{i}. 🎶 {r}")

                #Explanation
                st.info("These songs are recommended based on similarity in audio features like energy, loudness, and danceability using cosine similarity.")

