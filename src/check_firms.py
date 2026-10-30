"""Quick sanity checks on the cleaned FIRMS data."""  # Short description of this script

import pandas as pd  # Tables: load and summarize the data

df = pd.read_parquet("data/raw/firms_lebanon.parquet")  # Load the cleaned fire detections

print(f"Rows: {len(df)}")  # Total detections after cleaning
print(f"Date type: {df['acq_date'].dtype}")  # Should be datetime64, confirms dates parsed correctly
print(f"From {df['acq_date'].min().date()} to {df['acq_date'].max().date()}")  # First and last fire date in the data

print("\nDetections per year:")  # Header for yearly counts
print(df["acq_date"].dt.year.value_counts().sort_index())  # Count fires per year, oldest first

print("\nDetections per month (all years):")  # Header for monthly counts
print(df["acq_date"].dt.month.value_counts().sort_index())  # Count fires per calendar month, Jan to Dec

print("\nDay vs night:")  # Header for day/night split
print(df["daynight"].value_counts())  # How many detections came from day vs night passes

print("\nFire days (unique dates):", df["acq_date"].nunique())  # How many distinct days had at least one fire