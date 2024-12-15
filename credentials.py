

import os

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')


if GOOGLE_API_KEY:
    print("API Key found")
else:
    print("API Key not set")
    
if GROQ_API_KEY:
    print("API Key found")
else:
    print("API Key not set")


# QDRANT_HOST = "192.168.1.119"
    