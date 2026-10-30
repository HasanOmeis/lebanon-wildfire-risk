from pathlib import Path  # Path builds file paths that work on Windows and Linux
import requests  # requests downloads data from the internet

API_URL = "https://www.geoboundaries.org/api/current/gbOpen/LBN/ADM0/"  # geoBoundaries API for Lebanon (LBN), country level (ADM0)
OUT_PATH = Path("data/raw/lebanon_boundary.geojson")  # where the border file will be saved


def main():  # main function that runs the download
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # create data/raw if it doesn't exist yet
    meta = requests.get(API_URL, timeout=30)  # ask the API for info about Lebanon's boundary file
    meta.raise_for_status()  # stop with an error if the request failed
    geojson_url = meta.json()["gjDownloadURL"]  # pull the actual GeoJSON download link from the response
    print(f"Downloading from: {geojson_url}")  # show the link so we know what we're fetching
    resp = requests.get(geojson_url, timeout=60)  # download the GeoJSON file itself
    resp.raise_for_status()  # stop with an error if the download failed
    OUT_PATH.write_bytes(resp.content)  # save the downloaded bytes to disk
    print(f"Saved {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")  # confirm the save and show the file size


if __name__ == "__main__":  # only run main() when this file is executed directly
    main()  # start the download