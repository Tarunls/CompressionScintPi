import zipfile
import os, glob, struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

def readv325(path):
    dt = np.dtype([
        ('week', np.int32), ('towe', np.float32),
        ('leap', np.uint8), ('cons', np.uint8),
        ('sats', np.uint8), ('svid', np.uint8),
        ('elev', np.int8), ('azim', np.int32),
        ('snr1', np.uint8), ('snr2', np.uint8),
        ('snr3', np.uint8), ('pst1', np.uint8),
        ('pst2', np.uint8), ('pst3', np.uint8),
        ('rst1', np.uint8), ('rst2', np.uint8),
        ('rst3', np.uint8), ('cph1', np.float64),
        ('cph2', np.float64), ('cph3', np.float64),
        ('rng1', np.float64), ('rng2', np.float64),
        ('rng3', np.float64), ('lon', np.float32),
        ('lat', np.float32), ('hei', np.float32),
    ], align=True)
    arr = np.fromfile(path, dtype=dt)
    return pd.DataFrame.from_records(arr)

# -- Reuse your readv326 logic
def readv326(path):
    rec_dt = np.dtype([
       ('towe', np.float32), ('cons', np.uint8),
       ('sats', np.uint8), ('svid', np.uint8),
       ('elev', np.int8), ('azim', np.int32),
       ('snr1', np.uint8), ('snr2', np.uint8),
       ('pst1', np.uint8), ('pst2', np.uint8),
       ('rst1', np.uint8), ('rst2', np.uint8),
       ('cph1', np.float64), ('cph2', np.float64),
       ('rng1', np.float64), ('rng2', np.float64),
       ('lck1', np.int32), ('lck2', np.int32),
    ], align=True)
    hdr_size = 60
    with open(path, 'rb') as f:
        buf = f.read(hdr_size)
    fmt = '@fBbiBBBBBBddddi'
    vals = struct.unpack(fmt, buf)
    week = vals[-1]
    rec = np.fromfile(path, dtype=rec_dt, offset=64*2)
    df = pd.DataFrame.from_records(rec)
    df['week'] = week
    return df

# -- Optimize numeric dtypes
def optimize_dtypes(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            if df[col].min() >= 0 and df[col].max() <= 255:
                df[col] = df[col].astype('uint8')
            elif df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
        elif pd.api.types.is_float_dtype(df[col]) and col not in ['cph1', 'cph2', 'rng1', 'rng2']:
            df[col] = df[col].astype('float32')
    return df

# -- Convert week + towe to datetime
def add_datetime(df):
    gps_epoch = datetime(1980, 1, 6)
    df['datetime'] = [gps_epoch + timedelta(weeks=w, seconds=t) for w, t in zip(df['week'], df['towe'])]
    return df.drop(columns=['week', 'towe'])

# -- Calculate S4 index
def compute_s4(df):
    df['timestamp'] = df['datetime'].astype('int64') // 10**9
    df['s4_bucket'] = df['timestamp'] // 60

    def s4_linear(x):
        lin = 10 ** (x / 10)  # convert dB to linear
        return np.std(lin) / np.mean(lin) if np.mean(lin) != 0 else 0

    s4_vals = df.groupby(['svid', 'cons', 's4_bucket'])['snr1'].agg(s4_linear)
    s4_df = s4_vals.reset_index().rename(columns={'snr1': 's4'})

    df = df.merge(s4_df, on=['svid', 'cons', 's4_bucket'], how='left')
    df = df.drop(columns=['timestamp', 's4_bucket'])
    return df

# -- Size helper
def get_file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0

# -- Plotting
def plot_compression_ratio(unmasked_size, masked_size, slim_size, elev_masked_1min_size, no_mask_1sec_size):
    labels = ['Unmasked', 'Masked', 'Slimmed', 'Elev Masked (1min)', 'No Mask (1sec)']
    sizes = [unmasked_size, masked_size, slim_size, elev_masked_1min_size, no_mask_1sec_size]

    if unmasked_size == 0:
        print("Cannot plot compression ratios: Unmasked size is 0.")
        return

    ratios = [(s / unmasked_size) * 100 for s in sizes]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(labels, ratios, color=['gray', 'steelblue', 'lightcoral', 'mediumseagreen', 'goldenrod'])
    ax.set_ylabel('% of Original Unmasked Size')
    ax.set_title('Parquet (brotli) Compression Ratio Comparison')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                f'{ratios[i]:.1f}%', ha='center', fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, -5, # Position for MB size
                f'{sizes[i]:.2f} MB', ha='center', color='black', fontsize=9)


    plt.ylim(0, max(100, max(ratios) + 15))
    plt.tight_layout()
    plt.savefig("compression_ratios_comparison.png", dpi=300)
    plt.show()
    print(f"\n📊 Saved graph to 'compression_ratios_comparison.png'")


# -- Main processing (MODIFIED TO PROCESS ONLY FIRST FILE WITH PRINTS AND NEW OUTPUTS)
def process_all_v326(folder):
    print("📂 Current working directory:", os.getcwd())
    print("📁 Looking inside folder:", folder)
    print("📍 Absolute path:", os.path.abspath(folder))

    files = glob.glob(os.path.join(folder, "*.bin.zip"))
    if not files:
        print("❌ No .bin.zip files found in the folder!")
        return

    print(f"📁 Found {len(files)} zipped v326 files")
    big_df = []

    # Process only the first file as per the user's previous request,
    # but the logic is set up for all if `files` wasn't sliced.
    for i, file in enumerate(files): # Removed `[:1]` for full processing
        print(f"\n🎯 Processing file {i+1}/{len(files)}: {os.path.basename(file)}")

        try:
            # --- Unzip ---
            with zipfile.ZipFile(file, 'r') as z:
                extracted = z.namelist()
                if len(extracted) != 1:
                    print(f"❌ Expected 1 file inside zip, found {len(extracted)}")
                    continue
                inner_file = extracted[0]
                temp_path = os.path.join(folder, "__temp_unzipped__.bin")
                with z.open(inner_file) as src, open(temp_path, 'wb') as dst:
                    dst.write(src.read())
            print("✅ File unzipped")

            # --- Read + process ---
            tempdf = readv326(temp_path)
            tempdf = optimize_dtypes(tempdf)
            tempdf = add_datetime(tempdf)
            tempdf = compute_s4(tempdf) # S4 computed for all subsequent operations

            big_df.append(tempdf)
            print(f"✅ Appended {len(tempdf)} records")

        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
                print("🧹 Cleaned up temporary file")

    if not big_df:
        print("❌ No valid data processed.")
        return

    print("\n🧩 Concatenating all DataFrames...")
    df = pd.concat(big_df, ignore_index=True)
    print(f"📊 Total merged record count: {len(df)}")

    # Calculate initial unmasked size (before any filtering/sampling)
    # Convert to PyArrow table first to estimate Parquet size without writing to disk
    # This is an approximation; actual file size might vary slightly.


    # -- Save full unmasked (for baseline comparison, if desired, but not strictly asked to save)
    # unmasked_file = "v326_unmasked_all.parquet"
    # df.to_parquet(unmasked_file, compression='brotli', index=False)
    # unmasked_size = get_file_size_mb(unmasked_file)
    # print(f"✅ Saved unmasked: {unmasked_file} ({unmasked_size:.2f} MB)")


    # -- Mask (original masked file)
    masked_df = df[(df['elev'] > 30) & (df['s4'] > 0.2)]
    masked_file = "v326_masked_all.parquet"
    masked_df.to_parquet(masked_file, compression='brotli', index=False)
    masked_size = get_file_size_mb(masked_file)
    print(f"✅ Saved masked (elev > 30, s4 > 0.2): {masked_file} ({masked_size:.2f} MB)")

    # -- Slim (original slimmed file)
    slim_cols = ['datetime', 'svid', 'cons', 'snr1', 's4']
    masked_slim_df = masked_df[slim_cols]
    slim_file = "v326_masked_slim_all.parquet"
    masked_slim_df.to_parquet(slim_file, compression='brotli', index=False)
    slim_size = get_file_size_mb(slim_file)
    print(f"✅ Saved slimmed (elev > 30, s4 > 0.2, selected cols): {slim_file} ({slim_size:.2f} MB)")

    # --- NEW FILE 1: Elevation-masked, no S4 mask, 1-minute downsample ---
    print("\nProcessing new file: Elevation-masked, 1-minute downsample...")
    elev_masked_df = df[(df['elev'] > 30) & (df['elev'] < 90)].copy() # Use .copy() to avoid SettingWithCopyWarning
    elev_masked_df['minbin'] = elev_masked_df['datetime'].dt.floor('min')
    
    # Group and take the first record in each minute bin for each constellation/svid
    elev_masked_1min_df = elev_masked_df.groupby(['cons','svid','minbin']).first().reset_index()
    # You might want to select specific columns here, otherwise it will save all columns from .first()
    # For consistency, let's keep it similar to the slimmed approach but with all relevant columns
    cols_to_keep_1min = ['datetime', 'minbin', 'svid', 'cons', 'elev', 'azim', 'snr1', 'snr2', 's4']
    elev_masked_1min_df = elev_masked_1min_df[cols_to_keep_1min]


    elev_masked_1min_file = "v326_elev_masked_1min.parquet"
    elev_masked_1min_df.to_parquet(elev_masked_1min_file, compression='brotli', index=False)
    elev_masked_1min_size = get_file_size_mb(elev_masked_1min_file)
    print(f"✅ Saved elev-masked (elev > 30, no s4 mask, 1-min downsample): {elev_masked_1min_file} ({elev_masked_1min_size:.2f} MB)")

    # --- NEW FILE 2: No elevation mask, no S4 mask, 1-second downsample ---
    print("\nProcessing new file: No mask, 1-second downsample...")
    no_mask_df = df.copy() # Start from the full DataFrame with S4 computed
    no_mask_df['secbin'] = no_mask_df['datetime'].dt.floor('s')

    # Group and take the first record in each second bin for each constellation/svid
    no_mask_1sec_df = no_mask_df.groupby(['cons','svid','secbin']).first().reset_index()
    # Similar to above, select relevant columns
    cols_to_keep_1sec = ['datetime', 'secbin', 'svid', 'cons', 'elev', 'azim', 'snr1', 'snr2', 's4']
    no_mask_1sec_df = no_mask_1sec_df[cols_to_keep_1sec]

    no_mask_1sec_file = "v326_no_mask_1sec.parquet"
    no_mask_1sec_df.to_parquet(no_mask_1sec_file, compression='brotli', index=False)
    no_mask_1sec_size = get_file_size_mb(no_mask_1sec_file)
    print(f"✅ Saved no-mask (1-sec downsample): {no_mask_1sec_file} ({no_mask_1sec_size:.2f} MB)")

    # --- Plotting ---
# --- Run it
if __name__ == '__main__':
    # Ensure the 'files/highconfig_s4' directory exists with your .bin.zip files
    # For testing, you might want to create a dummy directory and put a dummy zip file in it
    # if you don't have the actual data yet.
    if not os.path.exists("./files/highconfig_s4"):
        print("Creating dummy directory for demonstration: ./files/highconfig_s4")
        os.makedirs("./files/highconfig_s4")
        # You would typically place your .bin.zip files here.
        # For a truly runnable example, a dummy zip with a dummy bin file could be created.
        # However, for this task, we assume the user provides the data.

    process_all_v326("./files/highconfig_s4")