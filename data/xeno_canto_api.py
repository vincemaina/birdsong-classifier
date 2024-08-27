import requests
import time
import json
import os
import librosa


BASE_API_ENDPOINT = "https://xeno-canto.org/api/2/recordings"
CACHE_PATH = "./data/api_responses/"
RECORDINGS_PATH = "./data/downloaded_recordings/"
SAMPLE_RATE = 16000


class XenoCantoApi:
    """Client for calling the XenoCanto api."""

    def __init__(self):
        """Construct new XenoCantoApi object."""

        self._last_call = None  # Used limit calls to 1 per second.
        print("Loaded new XenoCantoApi client.")

    def get_local_path(self, query_string: str) -> str:
        """Return properly formatted path name."""

        return CACHE_PATH + query_string + ".json"

    def load_from_cache(self, query_string: str) -> dict | None:
        """Load response from cache if exists, else None."""

        file_path = self.get_local_path(query_string)
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r") as f:
            return json.loads(f.read())

    def save_to_cache(self, data: dict, query_string: str) -> None:
        """Save api response to cache."""

        file_path = self.get_local_path(query_string)
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(json.dumps(data))

    def query(self, genus: str, subspecies: str):
        """Query xeno-canto API for recordings of the given birds."""

        query_string = f'?query=grp:"birds"gen:"{genus}"ssp:"{subspecies}"'

        # Check if cached already
        cached_res = self.load_from_cache(query_string)
        if cached_res is not None:
            return cached_res

        # Ensure has been longer than 1 second since last call
        if self._last_call is not None:
            time_since_last_call = time.time() - self._last_call
            if time_since_last_call < 1:
                time.sleep(1 - time_since_last_call)

        # Call the API
        print("Querying api")
        res = requests.get(BASE_API_ENDPOINT + query_string)
        self._last_call = time.time()  # Update last call time

        json_data = res.json()

        # Save to cache
        self.save_to_cache(data=json_data, query_string=query_string)

        return json_data

    def load_recording(self, id: int):
        """Downloads the audio file with the given id from xeno-canto.org"""

        # Construct the file path
        file_path = RECORDINGS_PATH + str(id)

        # Download file it not already saved
        if not os.path.exists(file_path):
            print(f"Downloading file with id {id}")
            res = requests.get(f"https://xeno-canto.org/{id}/download")
            if res.status_code == 200:
                # Save the file
                os.makedirs(RECORDINGS_PATH, exist_ok=True)
                with open(file_path, "wb") as file:
                    file.write(res.content)
                print(f"File saved successfully as {file_path}")
            else:
                print(f"Request failed with status code {res.status_code}")
                raise Exception(f"Failed to download file with id {id}")

        return librosa.load(path=file_path, sr=SAMPLE_RATE, mono=True)


client = XenoCantoApi()  # Singleton instance


if __name__ == "__main__":
    data = client.query(genus="fringilla", subspecies="coelebs")
    print("Number of recordings:", data["numRecordings"])

    first_recording_id = data["recordings"][0]["id"]
    print("First recording:", first_recording_id)

    y, sr = client.load_recording(first_recording_id)
    print("Successfully loaded recording.")

    print("Samples:", y)
