"""Download NASA FIRMS fire detections over Lebanon."""  # Short description of what this script does

import os  # Read environment variables (our API key)
import time  # Pause between requests so we don't hit the API rate limit
from datetime import date, timedelta  # Work with calendar dates and day steps
from io import StringIO  # Let pandas read CSV text that comes back from the API

import pandas as pd  # Tables: build, clean and save the fire data
import requests  # Send HTTP requests to the FIRMS API
from dotenv import load_dotenv  # Load variables from the .env file

load_dotenv()  # Read .env in the project root into environment variables

MAP_KEY = os.getenv("FIRMS_MAP_KEY")  # Get the key (change the name if yours is different in .env)
if not MAP_KEY:  # Stop early if the key wasn't found
    raise ValueError("FIRMS key not found. Check your .env file.")  # Clear error message instead of a confusing API error

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"  # FIRMS "area" endpoint that returns CSV
BBOX = "35.10,33.05,36.63,34.70"  # Lebanon box as west,south,east,north (longitude/latitude)

SOURCE_ARCHIVE = "VIIRS_SNPP_SP"  # VIIRS satellite, science-quality archive (older data)
SOURCE_RECENT = "VIIRS_SNPP_NRT"  # VIIRS satellite, near-real-time (last few months)

START_DATE = date(2012, 1, 20)  # First day VIIRS SNPP data exists
END_DATE = date.today() - timedelta(days=1)  # Yesterday (today's data isn't complete yet)
DAY_RANGE = 5  # The API returns at most 5 days per request, so we move in 5-day chunks

OUTPUT_PATH = "data/raw/firms_lebanon.parquet"  # Where the final cleaned file will be saved

def fetch_chunk(source: str, start: date) -> pd.DataFrame:  # Download up to 5 days of fires starting at 'start'
    url = f"{BASE_URL}/{MAP_KEY}/{source}/{BBOX}/{DAY_RANGE}/{start.isoformat()}"  # Build the request URL: key/source/box/days/date
    response = requests.get(url, timeout=60)  # Send the request, give up after 60 seconds
    response.raise_for_status()  # Raise an error if the server returned a failure code (e.g. 500)
    text = response.text.strip()  # The CSV body as text, without extra blank lines
    if text.lower().startswith("invalid"):  # FIRMS sends plain text like "Invalid MAP_KEY" on bad requests
        raise ValueError(f"FIRMS error: {text}")  # Stop and show the message from the API
    df = pd.read_csv(StringIO(text))  # Turn the CSV text into a table
    df["source"] = source  # Remember which product each row came from
    return df  # Hand back the table (can be empty if no fires those days)


def fetch_with_retry(source: str, start: date, tries: int = 3) -> pd.DataFrame:  # Retry a chunk if the network fails
    for attempt in range(1, tries + 1):  # Try up to 'tries' times (1, 2, 3)
        try:  # Attempt the download
            return fetch_chunk(source, start)  # Success: return the table straight away
        except requests.RequestException as err:  # Network/server problem (timeout, 503, etc.)
            print(f"  retry {attempt} for {start}: {err}")  # Tell us which chunk failed and why
            time.sleep(5 * attempt)  # Wait longer each time (5s, 10s, 15s) before retrying
    raise RuntimeError(f"Failed after {tries} tries: {source} {start}")  # Give up loudly if all tries fail


def fetch_range(source: str, start: date, end: date) -> pd.DataFrame:  # Download every 5-day chunk between two dates
    chunks = []  # Collect each chunk's table here
    current = start  # Begin at the start date
    n = 0  # Count requests for progress messages
    while current <= end:  # Keep going until we pass the end date
        df = fetch_with_retry(source, current)  # Download 5 days starting at 'current'
        if not df.empty:  # Only keep chunks that actually have fires
            chunks.append(df)  # Add it to the list
        current += timedelta(days=DAY_RANGE)  # Jump forward 5 days
        n += 1  # One more request done
        if n % 50 == 0:  # Every 50 requests...
            print(f"{source}: reached {current}")  # ...print where we are so you know it's alive
        time.sleep(0.5)  # Small pause to stay polite with the API
    if not chunks:  # Nothing found in the whole range
        return pd.DataFrame()  # Return an empty table
    return pd.concat(chunks, ignore_index=True)  # Stack all chunks into one table


if __name__ == "__main__":  # Only runs when you execute this file directly
    archive = fetch_range(SOURCE_ARCHIVE, START_DATE, END_DATE)  # Get all archive data (it stops having rows a few months ago)
    last_archive = pd.to_datetime(archive["acq_date"]).max().date()  # Last day the archive covers
    print(f"Archive rows: {len(archive)}, last date: {last_archive}")  # Report archive size and end date

    recent = fetch_range(SOURCE_RECENT, last_archive + timedelta(days=1), END_DATE)  # Fill the gap up to yesterday with near-real-time data
    print(f"Recent rows: {len(recent)}")  # Report how many recent rows we got

    raw = pd.concat([archive, recent], ignore_index=True)  # Combine both sources into one table
    raw.to_csv("data/raw/firms_lebanon_raw.csv", index=False)  # Save the untouched download so we never have to re-download
    print(f"Saved {len(raw)} rows to data/raw/firms_lebanon_raw.csv")  # Confirm the save