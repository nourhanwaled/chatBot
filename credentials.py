

import os

API_KEY = os.getenv('API_KEY')

if API_KEY:
    print("API Key found")
else:
    print("API Key not set")
    