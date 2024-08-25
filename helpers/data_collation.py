import json
import os
import time
import requests


BASE_URL = 'https://xeno-canto.org/api/2/recordings?query='


last_call = None  # Used to prevent calling the API too frequently. Limit is 1 per second.

def call_api(query: dict, last_call=last_call):
    """Call the xeno-canto API with the given query parameters."""
        
    # Build the query string
    params = ['grp:"birds"']
    for key, val in query.items():
        params.append(f'{key}:"{val}"')
    param_str = ' '.join(params)
    
    # Check if cached already
    cache_str = param_str.replace(" ", "").replace('"', "").replace(":", "")
    if os.path.exists(f"./data/api_cache/{cache_str}.json"):
        with open(f"./data/api_cache/{cache_str}.json", 'r') as f:
            cached_data = json.loads(f.read())
            return {
                'numRecordings': cached_data['numRecordings'],
                'recordings': cached_data['recordings']
            }
    
    # Ensure has been longer than 1 second since last call
    if last_call is not None:
        time_since_last_call = time.time() - last_call
        if time_since_last_call < 1:
            time.sleep(1 - time_since_last_call)
    
    # Call the API
    print("Calling the api")
    res = requests.get(BASE_URL + param_str)
    json_data = res.json()
    last_call = time.time()  # Update last call time
    
    # Cache response
    CACHE_PATH = './data/api_cache'
    with open(f"{CACHE_PATH}/{cache_str}.json", 'w') as f:
        f.write(json.dumps(json_data))
    
    return {
        'numRecordings': json_data['numRecordings'],
        'recordings': json_data['recordings']
    }


# Debug
if __name__ == "__main__":
    
    res = call_api({
        'gen': 'fringilla',
        'ssp': 'coelebs'
    })

    print("Number of recordings:", res['numRecordings'])

    print(json.dumps(res, indent=2))
