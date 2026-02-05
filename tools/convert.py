import pandas as pd

# --- Define the specific dates you want to filter for ---
# List of dates in 'MMDD' format (e.g., '0808' for August 8th)
target_dates_mmdd = ['1015']

# Read the input CSV
df = pd.read_csv('NSE_NIFTY251020C25350_RE.csv')

# Convert 'time' to numeric (handles any stray strings)
df['time'] = pd.to_numeric(df['time'], errors='coerce')

# Convert to UTC datetime, then to IST. We keep the timezone info for now for accurate filtering.
df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')

# --- New Filtering Logic ---
# Filter the DataFrame to keep only rows where the date matches one of the target_dates_mmdd.
# We format the date part of the 'datetime' column as 'MMDD' and check if it's in our list.
df_filtered = df[df['datetime'].dt.strftime('%m%d').isin(target_dates_mmdd)].copy()

# --- Continue with original logic on the filtered data ---

# Now, remove timezone info from the filtered data
df_filtered['datetime'] = df_filtered['datetime'].dt.tz_localize(None)

# Format the datetime column to the desired string format
df_filtered['datetime'] = df_filtered['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Save the filtered data to a new CSV file
df_filtered.to_csv('NSE_NIFTY251020C25350_RE_OUT.csv', index=False)

# Optional: print first few rows of the filtered data to verify
print(f"Filtered data for dates: {target_dates_mmdd}")
print(df_filtered[['time', 'datetime']].head())