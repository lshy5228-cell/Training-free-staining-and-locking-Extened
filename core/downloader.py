import requests
import json
import logging
import os

class DictionaryDownloader:
    """
    Handles the synchronization between the local environment and the 
    upstream GitHub translation repository.
    """
    def __init__(self, remote_url, local_storage):
        self.url = remote_url
        self.storage = local_storage

    def sync(self):
        logging.info(f"Connecting to remote repository: {self.url}")
        try:
            response = requests.get(self.url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                os.makedirs(os.path.dirname(self.storage), exist_ok=True)
                with open(self.storage, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                logging.info(f"Successfully synchronized {len(data)} remote entries.")
                return data
            logging.warning(f"Remote server returned status code: {response.status_code}")
        except Exception as e:
            logging.error(f"Network synchronization failed: {e}")
        
        return self.load_local()

    def load_local(self):
        if os.path.exists(self.storage):
            with open(self.storage, 'r', encoding='utf-8') as f:
                return json.load(f)
        logging.info("No local dictionary found. Using built-in fallback.")
        return {}
