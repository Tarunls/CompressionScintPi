import zipfile
import os, glob, struct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gc 

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

def readv326(path, offset_bytes=None, num_records=None):
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

    record_size = rec_dt.itemsize 
    hdr_size = 60
    data_start_offset = 64 * 2 

    with open(path, 'rb') as f:

        buf = f.read(hdr_size)
        fmt = '@fBbiBBBBBBddddi'
        vals = struct.unpack(fmt, buf)
        week = vals[-1]

        if offset_bytes is None or num_records is None:

            f.seek(data_start_offset)
            rec = np.fromfile(f, dtype=rec_dt)
        else:

            effective_offset = data_start_offset + offset_bytes
            f.seek(effective_offset)

            rec = np.fromfile(f, dtype=rec_dt, count=num_records)

    df = pd.DataFrame.from_records(rec)
    df['week'] = week
    return df, record_size, data_start_offset 

def optimize_dtypes(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            if df[col].min() >= 0 and df[col].max() <= 255:
                df[col] = df[col].astype('uint8')
            elif df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            elif df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
        elif pd.api.types.is_float_dtype(df[col]) and col not in ['cph1', 'cph2', 'rng1', 'rng2']:
            df[col] = df[col].astype('float32')
    return df

def add_datetime(df):
    gps_epoch = datetime(1980, 1, 6)

    df['week'] = df['week'].astype(int)
    df['datetime'] = [gps_epoch + timedelta(weeks=w, seconds=t) for w, t in zip(df['week'], df['towe'])]
    return df.drop(columns=['week', 'towe'])

def compute_s4(df):
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])

    df['timestamp'] = df['datetime'].astype('int64') // 10**9
    df['s4_bucket'] = df['timestamp'] // 60

    def s4_linear(x):
        lin = 10 ** (x / 10)
        return np.std(lin) / np.mean(lin) if np.mean(lin) != 0 else 0

    s4_vals = df.groupby(['svid', 'cons', 's4_bucket'])['snr1'].agg(s4_linear)
    s4_df = s4_vals.reset_index().rename(columns={'snr1': 's4'})

    df = df.merge(s4_df, on=['svid', 'cons', 's4_bucket'], how='left')
    df = df.drop(columns=['timestamp', 's4_bucket'])
    return df

def get_file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0

def plot_compression_ratio(unmasked_size, masked_size, slim_size, elev_masked_1min_size, no_mask_1sec_size, filename_prefix=""):
    labels = ['Unmasked', 'Masked', 'Slimmed', 'Elev Masked (1min)', 'No Mask (1sec)']
    sizes = [unmasked_size, masked_size, slim_size, elev_masked_1min_size, no_mask_1sec_size]

    if unmasked_size == 0:
        print(f"Cannot plot compression ratios for {filename_prefix}: Unmasked size is 0.")
        return

    ratios = [(s / unmasked_size) * 100 for s in sizes]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(labels, ratios, color=['gray', 'steelblue', 'lightcoral', 'mediumseagreen', 'goldenrod'])
    ax.set_ylabel('% of Original Processed Data Size')
    ax.set_title(f'Parquet (brotli) Compression Ratio Comparison for {filename_prefix}')

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                f'{ratios[i]:.1f}%', ha='center', fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, -5,
                f'{sizes[i]:.2f} MB', ha='center', color='black', fontsize=9)

    plt.ylim(0, max(100, max(ratios) + 15))
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_filename = os.path.join(script_dir, f"compression_ratios_{filename_prefix}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig)
    print(f"Saved graph to '{plot_filename}'")

def process_single_v326_file(filepath, output_folder, chunk_size_records=500000):
    """
    Processes a single v326 .bin.zip file in chunks,
    applies initial transformations, saves temporary parquet files,
    then loads all temps for S4 calculation, and saves final outputs.
    """
    base_filename = os.path.basename(filepath).replace(".bin.zip", "")
    print(f"\n--- Processing file: {os.path.basename(filepath)} ---")

    temp_unzipped_path = None
    intermediate_chunk_files = [] 

    try:

        with zipfile.ZipFile(filepath, 'r') as z:
            extracted_files = z.namelist()
            if len(extracted_files) != 1:
                print(f"Expected 1 file inside zip, found {len(extracted_files)}. Skipping {filepath}.")
                return
            inner_file = extracted_files[0]
            temp_unzipped_path = os.path.join(output_folder, f"__temp_unzipped_{base_filename}__.bin")

            buffer_size = 4 * 1024 
            try:
                with z.open(inner_file) as src: 
                    with open(temp_unzipped_path, 'wb') as dst: 
                        while True:
                            data = src.read(buffer_size) 
                            if not data:
                                break 
                            dst.write(data) 
                print(f"'{inner_file}' unzipped to '{temp_unzipped_path}' by streaming.")
            except Exception as e:
                print(f"Error streaming unzipping '{inner_file}' from '{filepath}': {e}")

                if os.path.exists(temp_unzipped_path):
                    os.remove(temp_unzipped_path)
                return 

        dummy_df, record_size, data_start_offset = readv326(temp_unzipped_path, offset_bytes=0, num_records=1)
        del dummy_df 
        gc.collect()

        total_file_size = os.path.getsize(temp_unzipped_path)
        data_size_bytes = total_file_size - data_start_offset
        total_records = data_size_bytes // record_size
        print(f"Total data records in '{os.path.basename(temp_unzipped_path)}': {total_records}, Record size: {record_size} bytes")

        print(f"Processing in chunks of {chunk_size_records} records (pre-S4 transformations)...")
        for i in range(0, total_records, chunk_size_records):
            current_offset_records = i
            current_offset_bytes = current_offset_records * record_size
            num_records_to_read = min(chunk_size_records, total_records - current_offset_records)

            if num_records_to_read <= 0:
                break 

            print(f"  Reading chunk {i // chunk_size_records + 1}: Records {current_offset_records} to {current_offset_records + num_records_to_read - 1}")

            chunk_df, _, _ = readv326(temp_unzipped_path, offset_bytes=current_offset_bytes, num_records=num_records_to_read)

            chunk_df = optimize_dtypes(chunk_df)
            chunk_df = add_datetime(chunk_df)

            intermediate_file_name = f"__temp_{base_filename}_chunk_{i // chunk_size_records}.parquet"
            intermediate_file_path = os.path.join(output_folder, intermediate_file_name)
            chunk_df.to_parquet(intermediate_file_path, compression='brotli', index=False)
            intermediate_chunk_files.append(intermediate_file_path)
            print(f"  Saved intermediate chunk to {os.path.basename(intermediate_file_path)}")

            del chunk_df 
            gc.collect() 

        print(f"All {len(intermediate_chunk_files)} intermediate chunks for '{base_filename}' saved.")

        print(f"Loading all intermediate files ({len(intermediate_chunk_files)} files) for S4 computation...")

        try:
            df_for_s4 = pd.concat([pd.read_parquet(f) for f in intermediate_chunk_files], ignore_index=True)
            print(f"Successfully combined {len(df_for_s4)} records for S4 computation.")
        except MemoryError:
            print(f"MemoryError: Failed to load all intermediate files for S4 computation for {base_filename}.")
            print("This indicates the combined size of the processed data (before S4) is too large for your system's RAM.")
            print("Consider increasing RAM/swap space or moving to a more powerful machine.")
            return 

        df_final = compute_s4(df_for_s4)
        print(f"Computed S4 for all records for '{base_filename}'.")
        del df_for_s4 
        gc.collect()

        sizes_for_plot = {} 

        unmasked_file_name = f"{base_filename}_unmasked.parquet"
        unmasked_file_path = os.path.join(output_folder, unmasked_file_name)
        df_final.to_parquet(unmasked_file_path, compression='brotli', index=False)
        sizes_for_plot['unmasked'] = get_file_size_mb(unmasked_file_path)
        print(f"Saved unmasked processed data: {unmasked_file_path} ({sizes_for_plot['unmasked']:.2f} MB)")

        masked_df = df_final[(df_final['elev'] > 30) & (df_final['s4'] > 0.2)]
        masked_file_name = f"{base_filename}_masked.parquet"
        masked_file_path = os.path.join(output_folder, masked_file_name)
        masked_df.to_parquet(masked_file_path, compression='brotli', index=False)
        sizes_for_plot['masked'] = get_file_size_mb(masked_file_path)
        print(f"Saved masked (elev > 30, s4 > 0.2): {masked_file_path} ({sizes_for_plot['masked']:.2f} MB)")
        del masked_df
        gc.collect()

        slim_cols = ['datetime', 'svid', 'cons', 'snr1', 's4']
        actual_slim_cols = [col for col in slim_cols if col in df_final.columns]

        masked_slim_df = df_final[(df_final['elev'] > 30) & (df_final['s4'] > 0.2)][actual_slim_cols]
        slim_file_name = f"{base_filename}_masked_slim.parquet"
        slim_file_path = os.path.join(output_folder, slim_file_name)
        masked_slim_df.to_parquet(slim_file_path, compression='brotli', index=False)
        sizes_for_plot['slim'] = get_file_size_mb(slim_file_path)
        print(f"Saved slimmed (elev > 30, s4 > 0.2, selected cols): {slim_file_path} ({sizes_for_plot['slim']:.2f} MB)")
        del masked_slim_df
        gc.collect()

        print(f"Processing for {base_filename}: Elevation-masked, 1-minute downsample...")
        elev_masked_df = df_final[(df_final['elev'] > 30) & (df_final['elev'] < 90)].copy()
        elev_masked_df['minbin'] = elev_masked_df['datetime'].dt.floor('min')
        elev_masked_1min_df = elev_masked_df.groupby(['cons','svid','minbin']).first().reset_index()
        cols_to_keep_1min = ['datetime', 'minbin', 'svid', 'cons', 'elev', 'azim', 'snr1', 'snr2', 's4']
        actual_cols_1min = [col for col in cols_to_keep_1min if col in elev_masked_1min_df.columns]
        elev_masked_1min_df = elev_masked_1min_df[actual_cols_1min]
        elev_masked_1min_file_name = f"{base_filename}_elev_masked_1min.parquet"
        elev_masked_1min_file_path = os.path.join(output_folder, elev_masked_1min_file_name)
        elev_masked_1min_df.to_parquet(elev_masked_1min_file_path, compression='brotli', index=False)
        sizes_for_plot['elev_masked_1min'] = get_file_size_mb(elev_masked_1min_file_path)
        print(f"Saved elev-masked (elev > 30, no s4 mask, 1-min downsample): {elev_masked_1min_file_path} ({sizes_for_plot['elev_masked_1min']:.2f} MB)")
        del elev_masked_df, elev_masked_1min_df
        gc.collect()

        print(f"Processing for {base_filename}: No mask, 1-second downsample...")
        no_mask_df = df_final.copy()
        no_mask_df['secbin'] = no_mask_df['datetime'].dt.floor('s')
        no_mask_1sec_df = no_mask_df.groupby(['cons','svid','secbin']).first().reset_index()
        cols_to_keep_1sec = ['datetime', 'secbin', 'svid', 'cons', 'elev', 'azim', 'snr1', 'snr2', 's4']
        actual_cols_1sec = [col for col in cols_to_keep_1sec if col in no_mask_1sec_df.columns]
        no_mask_1sec_df = no_mask_1sec_df[actual_cols_1sec]
        no_mask_1sec_file_name = f"{base_filename}_no_mask_1sec.parquet"
        no_mask_1sec_file_path = os.path.join(output_folder, no_mask_1sec_file_name)
        no_mask_1sec_df.to_parquet(no_mask_1sec_file_path, compression='brotli', index=False)
        sizes_for_plot['no_mask_1sec'] = get_file_size_mb(no_mask_1sec_file_path)
        print(f"Saved no-mask (1-sec downsample): {no_mask_1sec_file_path} ({sizes_for_plot['no_mask_1sec']:.2f} MB)")
        del no_mask_df, no_mask_1sec_df
        gc.collect()

        plot_compression_ratio(
            sizes_for_plot.get('unmasked', 0),
            sizes_for_plot.get('masked', 0),
            sizes_for_plot.get('slim', 0),
            sizes_for_plot.get('elev_masked_1min', 0),
            sizes_for_plot.get('no_mask_1sec', 0),
            filename_prefix=base_filename
        )

    except Exception as e:
        print(f"Failed to process {filepath}: {e}")
        import traceback
        traceback.print_exc()

    finally:

        if temp_unzipped_path and os.path.exists(temp_unzipped_path):
            os.remove(temp_unzipped_path)
            print(f"Cleaned up temporary unzipped file: {temp_unzipped_path}")

        for f_path in intermediate_chunk_files:
            if os.path.exists(f_path):
                os.remove(f_path)
                print(f"Cleaned up intermediate chunk file: {f_path}")

        if 'df_final' in locals():
            del df_final
        gc.collect()
        print(f"Finished processing {os.path.basename(filepath)}. All related memory and temporary files released.")

def process_all_v326_files_in_folder(input_folder, output_folder="processed_v326_output", chunk_size_records=500000):
    """
    Reads all .bin.zip files from input_folder, processes them one by one,
    and saves derived Parquet files into output_folder.
    """
    print("Current working directory:", os.getcwd())
    print("Looking for .bin.zip files in input folder:", input_folder)
    print("Absolute input path:", os.path.abspath(input_folder))

    if not os.path.exists(output_folder):
        print(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder)
    else:
        print(f"Output directory already exists: {output_folder}")

    files = glob.glob(os.path.join(input_folder, "*.bin.zip"))
    if not files:
        print(f"No .bin.zip files found in the specified input folder: {input_folder}")
        print("Please ensure the folder path is correct and contains '.bin.zip' files.")
        return

    print(f"Found {len(files)} zipped v326 files to process.")

    for i, file_path in enumerate(files):

        process_single_v326_file(file_path, output_folder, chunk_size_records=chunk_size_records)

    print("\nAll specified files have been processed.")

if __name__ == '__main__':
    input_directory = "./files/highconfig_s4" 
    output_directory = "./processed_v326_output" 

    if not os.path.exists(input_directory):
        print(f"Creating input directory: {input_directory} (Please place your .bin.zip files here)")
        os.makedirs(input_directory)

    process_all_v326_files_in_folder(input_directory, output_directory, chunk_size_records=500000)