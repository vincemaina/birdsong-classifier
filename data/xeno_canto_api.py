import requests
import time
import json
import os


BASE_API_ENDPOINT = "https://xeno-canto.org/api/2/recordings"
CACHE_PATH = "./data/api_cache/"


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
        os.makedirs(CACHE_PATH)
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


client = XenoCantoApi()


if __name__ == "__main__":
    data = client.query(genus="fringilla", subspecies="coelebs")
    print("Number of recordings:", data["numRecordings"])
