import sys                                   # gives access to the Python version
import os                                    # lets us read environment variables
from importlib.metadata import version       # reads the installed version of a package

import requests                              # sends HTTP requests to web APIs
from dotenv import load_dotenv               # loads variables from the .env file

print(f"Python: {sys.version.split()[0]}")   # print the Python version, should be 3.12.x

packages = [                                 # list of key packages to check
    "pandas", "geopandas", "rasterio",       # data and geospatial packages
    "lightgbm", "scikit-learn", "mlflow",    # modeling and tracking packages
    "optuna", "shap", "fastapi", "streamlit" # tuning, explanation, API and app packages
]                                            # end of the package list

for pkg in packages:                         # loop through each package name
    print(f"{pkg}: {version(pkg)}")          # print the package name and its installed version

load_dotenv()                                # read the .env file into environment variables
key = os.getenv("FIRMS_MAP_KEY")             # get the FIRMS key, or None if it's missing

if not key or key == "paste_your_key_here":  # check if the key is missing or still the placeholder
    print("FIRMS key: MISSING - add it to .env")   # warn that the key needs to be added
else:                                        # otherwise the key exists
    url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={key}"  # FIRMS endpoint that checks a key
    response = requests.get(url, timeout=30) # call the endpoint, give up after 30 seconds
    print(f"FIRMS key status code: {response.status_code}")  # 200 means the request worked
    print(f"FIRMS response: {response.text[:200]}")          # print the first 200 characters of the reply