"""Clean raw FIRMS detections and save them as Parquet."""  # Short description of this script

import pandas as pd  # Tables: load, clean and save the data

RAW_PATH = "data/raw/firms_lebanon_raw.csv"  # The untouched download from Step 5
OUTPUT_PATH = "data/raw/firms_lebanon.parquet"  # Where the cleaned file goes

KEEP_COLS = [  # Only the columns we'll actually use later
    "latitude",  # Fire location: north-south
    "longitude",  # Fire location: east-west
    "acq_date",  # Day the satellite saw the fire
    "acq_time",  # Time of the pass (HHMM, UTC)
    "daynight",  # D = daytime pass, N = night pass
    "frp",  # Fire Radiative Power (MW): how intense the fire was
    "bright_ti4",  # Brightness temperature (K) in the fire-detection band
    "confidence",  # How sure the satellite is that it's a real fire
    "source",  # Archive (SP) or near-real-time (NRT)
]  # End of column list

df = pd.read_csv(RAW_PATH)  # Load the raw download
print(f"Raw rows: {len(df)}")  # Count before cleaning

if "type" in df.columns:  # The archive has a 'type' column, NRT does not
    df = df[df["type"].isna() | (df["type"] == 0)]  # Keep vegetation fires (0) and NRT rows (no type); drops factories/gas flares (2) and offshore (3)
print(f"After type filter: {len(df)}")  # Count after removing non-vegetation heat sources

df["confidence"] = df["confidence"].astype(str).str.lower().str[0]  # Normalize 'low'/'l', 'nominal'/'n', 'high'/'h' to one letter
df = df[df["confidence"] != "l"]  # Drop low-confidence detections (often false alarms)
print(f"After confidence filter: {len(df)}")  # Count after removing weak detections

df["acq_date"] = pd.to_datetime(df["acq_date"])  # Convert date text into real dates

df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])  # Remove the same detection appearing twice (e.g. overlapping chunks)
print(f"After dedupe: {len(df)}")  # Count after removing duplicates

df = df[KEEP_COLS].sort_values(["acq_date", "acq_time"]).reset_index(drop=True)  # Keep chosen columns, sort by time, renumber rows

df.to_parquet(OUTPUT_PATH, index=False)  # Save as Parquet: smaller and faster than CSV, keeps data types
print(f"Saved {len(df)} rows to {OUTPUT_PATH}")  # Confirm the save
print(df.dtypes)  # Show each column's type to check dates/numbers are correct