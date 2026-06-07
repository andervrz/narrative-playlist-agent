# spotify_client.py
# Wrapper de spotipy para interactuar con la API de Spotify

import spotipy
from spotipy.oauth2 import SpotifyOAuth

class SpotifyClient:
    def __init__(self, client_id, client_secret, redirect_uri, scope):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                                             client_secret=client_secret,
                                                             redirect_uri=redirect_uri,
                                                             scope=scope))

    def search_tracks(self, query, limit=10):
        results = self.sp.search(q=query, type='track', limit=limit)
        return results['tracks']['items']

    def get_track_features(self, track_id):
        return self.sp.audio_features([track_id])[0]

    def create_playlist(self, user_id, name, description=''):
        return self.sp.user_playlist_create(user=user_id, name=name, public=False, description=description)

    def add_tracks_to_playlist(self, playlist_id, track_uris):
        self.sp.playlist_add_items(playlist_id, track_uris)

    def get_user_id(self):
        return self.sp.current_user()['id']

# End of file
