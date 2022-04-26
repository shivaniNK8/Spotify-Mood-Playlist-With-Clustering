import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from random import sample


pipeline = joblib.load('./song-cluster-model.joblib')
cid = '48a10dfefb294172a7053a196c799b7a'
secret = '2feb3ec9a9194416b50613cad389d733'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

df = pd.read_csv("spotify_songs.csv")
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
display_cols = ["track_name", "track_artist", "track_album_name","track_popularity", "playlist_subgenre"]
display_col_names = ["Track Name", "Artist", "Album","Track Popularity", "Genre"]

df['Labels']=pipeline.predict(df[audio_features])
tracks_df = df[audio_features]

#Create header
st.write("""# Spotify Mood Playlist""")
st.write("""## How it works""")
st.write("Get a mood playlist with songs similar to your favorite song. Just provide the song input."
         "Alternatively, play with the mood sliders to get a desired playlist.")

#image
image = Image.open('spotify.jpeg')
st.image(image)

#Bring in the data
data = pd.read_csv('spotify_songs.csv')
# st.write("## THE DATA BEING USED")
# data

#Create and name sidebar
st.sidebar.header('Choose Your Playlist Preferences')
artist_name = st.sidebar.text_input("Artist Name")
track_name = st.sidebar.text_input("Track Name")

attr_check_box = st.sidebar.checkbox("Provide audio features instead of artist and song name?")
obscurify = st.sidebar.checkbox("Do you want less popular songs?")
st.sidebar.write("Click Moodify multiple times to get different playlists")
submit_button = st.sidebar.button('Moodify!')

def user_input_features():

    if attr_check_box:
        #danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.746, step = 0.0001)
        danceability = st.sidebar.select_slider('Danceability', np.around(np.arange(0.0, 1.0, 0.001), decimals = 3), value = 0.726)
        energy = st.sidebar.select_slider('Energy', np.around(np.arange(0.0, 1.0, 0.001), decimals = 3), value = 0.815)
        key = st.sidebar.slider('Key', 0, 11, 11, 1)
        loudness = st.sidebar.slider('Loudness', -46.0, 2.0, -4.969, 0.01)
        mode = st.sidebar.slider('Mode', 0, 1, 1, 1)
        speechiness = st.sidebar.select_slider('Speechiness', np.around(np.arange(0.0, 1.0, 0.0001), decimals = 4), value = 0.0373)
        acousticness = st.sidebar.select_slider('Acousticness', np.around(np.arange(0.0, 1.0, 0.0001), decimals = 4), value = 0.0724)
        instrumentalness = st.sidebar.select_slider('Instrumentalness', np.around(np.arange(0.0, 1.0, 0.00001), decimals = 5), value = 0.00421)
        liveness = st.sidebar.select_slider('Liveness', np.around(np.arange(0.0, 1.0, 0.001), decimals = 3), value = 0.357)
        valence = st.sidebar.select_slider('Valence', np.around(np.arange(0.0, 1.0, 0.001), decimals = 3), value = 0.693)
        tempo = st.sidebar.select_slider('Tempo', np.around(np.arange(0.0, 240.0, 0.001), decimals = 3), value = 99.972)
        


        user_data = {'danceability': danceability,
                 'energy': energy,
                 'key': key,
                 'loudness': loudness,
                 'mode': mode,
                 'speechiness': speechiness,
                 'acousticness': acousticness,
                 'instrumentalness': instrumentalness,
                 'liveness': liveness,
                 'valence': valence,
                 'tempo': tempo}
        features = pd.DataFrame(user_data, index=[0])
        features = features[audio_features]

        closest_index = distance.cdist(np.array(features), np.array(tracks_df), metric = 'minkowski', p = 2).argmin()
        closest_song = df.iloc[[closest_index]][display_cols].reset_index(drop = True)
        closest_song.columns = display_col_names
        st.write("## CLOSEST SONG TO THESE AUDIO FEATURES:")        
        closest_song

        return features
    elif artist_name and track_name:
        song = sp.search(q="artist:" + artist_name + " track:" + track_name, type="track")
        song_id = song['tracks']['items'][0]['id']
        features = pd.DataFrame(sp.audio_features(song_id))
        features = features[audio_features]
        return features

def create_playlist(new_song, df, label, threshold = 0.1, obscurify = False):

    
    num_songs = 15
    song_label=pipeline.predict(new_song)[0]
    cluster_df=df[df['Labels']==song_label]
    song_dist=[]
    
    for i in range(len(cluster_df)):
        song_dist.append(distance.cdist(np.array(new_song.head(1)), np.array(cluster_df[cluster_df['Labels']==label].loc[:,cluster_df.columns.isin(audio_features)][i:i+1]), metric = 'minkowski', p = 2)[0])
      
    cluster_df['song_dist']=song_dist
    cluster_df['normal_dist']=MinMaxScaler().fit_transform(cluster_df[['song_dist']])
    playlist=cluster_df[cluster_df['normal_dist']<=threshold][display_cols]
    if obscurify == False:
        final_playlist=playlist.sample(min(len(playlist),num_songs)).reset_index(drop = True)
    else:
        popularity_quantile = playlist["track_popularity"].quantile(0.05)
        st.write("Maximum Track Popularity Percentage (5% Quantile):", popularity_quantile)
        quantile_playlist = playlist.loc[playlist["track_popularity"] <= popularity_quantile]
        final_playlist = quantile_playlist.sample(min(len(quantile_playlist),num_songs)).reset_index(drop = True)
    final_playlist.columns = display_col_names
    return(final_playlist)

def generate_playlist(df_user):
    cluster_num = predict_cluster(df_user)
    st.write(cluster_num)
    playlist = create_playlist(df_user, df, cluster_num, 0.08, obscurify = obscurify)
    playlist
    

def predict_cluster(df_user):
    cluster_num = pipeline.predict(df_user)
    return cluster_num[0]

df_user = user_input_features()

if df_user is not None:
    st.write("## YOUR CHOSEN MOOD ATTRIBUTES: ")
    df_user

if submit_button:
    st.write("## GENERATED PLAYLIST")
    generate_playlist(df_user)

with st.expander("More About Audio Features"):
     st.markdown("""
         **Danceability:** Danceability describes how suitable a track is for dancing based 
         on a combination of musical elements including tempo, rhythm stability, beat strength, 
         and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.

         **Energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. 
         Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a 
         Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, 
         perceived loudness, timbre, onset rate, and general entropy.

         **Key:** The key the track is in. Integers map to pitches using standard Pitch Class notation. 


         **Loudness:** The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track 
         and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary 
         psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db.

         **Mode:** Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. 
         Major is represented by 1 and minor is 0.

         **Speechiness:** Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording 
         (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably 
         made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in 
         sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like 
         tracks.

         **Acousticness:** A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence 
         the track is acoustic.

         **Instrumentalness:** Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental 
         in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the 
         greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, 
         but confidence is higher as the value approaches 1.0.

         **Liveness:**
         Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that 
         the track was performed live. A value above 0.8 provides strong likelihood that the track is live.

         **Valence:**
         A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive 
         (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

         **Tempo:**
         The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of 
         a given piece and derives directly from the average beat duration.

         _Credit: Spotify Web API Docs_

     """)
with st.expander("More About Data"):
    st.write("""
         This is a snapshot of data used to curate your playlist!
     """)
    st.dataframe(tracks_df.head(15))



