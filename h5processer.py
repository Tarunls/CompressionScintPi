import glob, pandas as pd, numpy as np, time, zipfile, os, re, struct
from multiprocessing import Pool
import duckdb
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.orc as orc

gnssdic = {0: 'GPS', 1: 'SBS', 2: 'GAL', 3: 'BDS', 6: 'GLO'}
workers = 10 

# Define usecols here as it's a global constant needed by process_file
usecols = ['cons', 'svid', 'week', 'towe', 'elev', 'azim',
             'snr1', 'snr2', 'cph1', 'cph2', 'rng1', 'rng2']

conversion_stats = {'parquet': [], 'duckdb': [], 'feather': [], 'hdf5': [], 'orc': []}

def readvscintpi1(path):
    colnames = ['hour', 'minute', 'second', 'year', 'month', 'day',
                'svid', 'elev', 'azim', 'snr1', 'lat', 'lon', 'alt']
    return pd.read_csv(path, sep=r'\s+', header=None, names=colnames)

def readv324(path):
    cols2read = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15]  
    colnames = ['week', 'towe', 'cons', 'svid', 'elev', 'azim',
                'snr1', 'snr2', 'cph1', 'cph2', 'rng1', 'rng2']
    df = pd.read_csv(path, sep=r'\s+', header=None, usecols=cols2read)
    df.columns = colnames
    return df

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

def readv326(path):
    rec_dt = np.dtype([
       ('towe', np.float32), ('cons', np.uint8), ('sats', np.uint8), ('svid', np.uint8),
       ('elev', np.int8), ('azim', np.int32), ('snr1', np.uint8), ('snr2', np.uint8),
       ('pst1', np.uint8), ('pst2', np.uint8), ('rst1', np.uint8), ('rst2', np.uint8),
       ('cph1', np.float64), ('cph2', np.float64), ('rng1', np.float64), ('rng2', np.float64),
       ('lck1', np.int32), ('lck2', np.int32),
    ], align=True)
    with open(path, 'rb') as f:
        f.read(60)  
        buf = f.read(60)
    week = struct.unpack('@fBbiBBBBBBddddi', buf)[-1]
    rec = np.fromfile(path, dtype=rec_dt, offset=64 * 2)
    df = pd.DataFrame.from_records(rec)
    df['week'] = week
    return df

def unzip_func(filepath, outpath):
    os.makedirs(outpath, exist_ok=True)
    with zipfile.ZipFile(filepath, 'r') as z:
        extracted_names = z.namelist()
        if len(extracted_names) != 1:
            raise ValueError(f"Zip file {filepath} should contain exactly one file, found: {extracted_names}")
        original_name = extracted_names[0]
        base_name = os.path.basename(original_name)
        name, ext = os.path.splitext(base_name)
        new_name = f"{name}_vscintpi1{ext}" if 'vscintpi1' in filepath.lower() else base_name
        extracted_path = os.path.join(outpath, new_name)
        with z.open(original_name) as src, open(extracted_path, 'wb') as dst:
            dst.write(src.read())
    return extracted_path

def getvers(path):
    basename = os.path.basename(path).lower()
    if re.search(r'_v(\d+)', basename):
        return re.search(r'_v(\d+)', basename).group(1)
    if '_vscintpi1' in basename:
        return 'vscintpi1'
    return None

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def optimize_dtypes(df): 
    """Optimize data types to reduce memory usage using explicit range checks."""
    for col in df.columns: 
        # Skip datetime column as it's already optimized by Pandas internally
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        if col in ['svid', 'elev', 'azim', 'cons']: # Added 'cons' here for explicit handling
            # If the column is initially a float (e.g., from read_csv), try converting to int first
            if pd.api.types.is_float_dtype(df[col]) and (df[col] == df[col].astype(int)).all():
                df[col] = df[col].astype(int)
            
            if pd.api.types.is_integer_dtype(df[col]):
                col_min, col_max = df[col].min(), df[col].max() 
                if col_min >= 0 and col_max <= 255: # Often for 'cons', it's 0-6
                    df[col] = df[col].astype('uint8') 
                elif col_min >= -128 and col_max <= 127: 
                    df[col] = df[col].astype('int8') 
                elif col_min >= -32768 and col_max <= 32767: 
                    df[col] = df[col].astype('int16') 
                elif col_min >= -2147483648 and col_max <= 2147447: 
                    df[col] = df[col].astype('int32')
                # If it exceeds int32, it will remain int64, which is appropriate.
        elif col in ['snr1', 'snr2']: 
            if pd.api.types.is_numeric_dtype(df[col]): 
                if df[col].min() >= 0 and df[col].max() <= 255: 
                    df[col] = df[col].astype('uint8') 
        elif col in ['towe', 'cph1', 'cph2', 'rng1', 'rng2']: 
            if df[col].dtype == 'float64': 
                df[col] = df[col].astype('float64')
    return df

def prepare_pyarrow_compatible_df(df):
    """Prepare dataframe for PyArrow-based formats (Feather, ORC) compatibility"""
    df_arrow = df.copy()
    
    for col in df_arrow.columns:
        # PyArrow does not directly support CategoricalDtype for non-string categories
        # and datetime tz-aware objects, or mixed types.
        # Simplest is to convert categoricals to strings, and let pyarrow handle datetime64[ns]
        if pd.api.types.is_categorical_dtype(df_arrow[col]):
            df_arrow[col] = df_arrow[col].astype(str)
        # Ensure that datetime columns are timezone-naive for best compatibility with Feather/Parquet/ORC
        if pd.api.types.is_datetime64_any_dtype(df_arrow[col]) and pd.api.types.is_datetime64tz_dtype(df_arrow[col]):
            df_arrow[col] = df_arrow[col].dt.tz_localize(None)
    
    return df_arrow

def save_to_formats(df, base_path, basename, file_version):
    """Save dataframe to Parquet, DuckDB, Feather (ZSTD/LZ4), ORC (ZSTD/Snappy), and HDF5 (BLOSC/NONE); record timing/size"""
    formats = {
        'parquet_zstd':   {'ext': '.parquet', 'folder': 'pq_zstd',   'compression': 'zstd', 'compression_level': 9}, # ZSTD default/normal level
        'parquet_lz4':    {'ext': '.parquet', 'folder': 'pq_lz4',    'compression': 'lz4'},
        'parquet_brotli': {'ext': '.parquet', 'folder': 'pq_brotli', 'compression': 'brotli'},
        'parquet_gzip':   {'ext': '.parquet', 'folder': 'pq_gzip',   'compression': 'gzip'},
        'parquet_snappy': {'ext': '.parquet', 'folder': 'pq_snappy', 'compression': 'snappy'},
        'duckdb_zstd':    {'ext': '.duckdb',   'folder': 'duckdb_zstd', 'compression': 'zstd', 'compression_level': 9}, # DuckDB uses Parquet's compression levels
        'duckdb_lz4':     {'ext': '.duckdb',   'folder': 'duckdb_lz4',  'compression': 'lz4'},
        'feather_zstd':   {'ext': '.feather', 'folder': 'feather_zstd', 'compression': 'zstd', 'compression_level': 9}, # Feather (PyArrow) default/normal ZSTD level
        'feather_lz4':    {'ext': '.feather', 'folder': 'feather_lz4',  'compression': 'lz4'},
        'orc_zstd':       {'ext': '.orc',     'folder': 'orc_zstd',    'compression': 'zstd', 'compression_level': 9}, # ORC with ZSTD compression
        'orc_snappy':     {'ext': '.orc',     'folder': 'orc_snappy',  'compression': 'snappy'}, # ORC with Snappy compression
        'hdf5_blosc':     {'ext': '.h5', 'folder': 'hdf5_blosc', 'compression': 'blosc', 'compression_level': 9}, # BLOSC max level is 9
        'hdf5_none':      {'ext': '.h5', 'folder': 'hdf5_none', 'compression': None}
    }

    results = {}
    saved_file_paths = {}

    for fmt, info in formats.items():
        try:
            start_time = time.time()

            base_dir = os.path.dirname(base_path)
            format_dir = os.path.join(base_dir, info['folder'])
            os.makedirs(format_dir, exist_ok=True)

            filename = os.path.basename(base_path).replace('.pq', info['ext'])
            filepath = os.path.join(format_dir, filename)
            compression = info['compression']
            compression_level = info.get('compression_level') # Get the compression level if specified

            if fmt.startswith('parquet'):
                if compression == 'zstd' and compression_level is not None:
                    # Pass compression_level for zstd
                    df.to_parquet(filepath, compression=compression, compression_level=compression_level, index=False)
                else:
                    df.to_parquet(filepath, compression=compression, index=False)

            elif fmt.startswith('duckdb'):
                # DuckDB typically works by creating a Parquet file for external data
                parquet_path = filepath.replace('.duckdb', '.parquet')
                conn = duckdb.connect(database=':memory:')
                df_duck = df.copy()
                for col in df_duck.columns:
                    if pd.api.types.is_categorical_dtype(df_duck[col]):
                        df_duck[col] = df_duck[col].astype(str)

                conn.register("df_duck", df_duck)
                # For DuckDB, the compression level is passed as an option in the COPY statement
                if compression == 'zstd' and compression_level is not None:
                    conn.execute(f"COPY df_duck TO '{parquet_path}' (FORMAT 'parquet', COMPRESSION '{compression}', COMPRESSION_LEVEL {compression_level})")
                else:
                    conn.execute(f"COPY df_duck TO '{parquet_path}' (FORMAT 'parquet', COMPRESSION '{compression}')")
                conn.close()
                filepath = parquet_path # Use the parquet path for duckdb for consistency in file ops

            elif fmt.startswith('feather'):
                df_arrow = prepare_pyarrow_compatible_df(df)
                try:
                    if compression == 'zstd' and compression_level is not None:
                        # Pass compression_level for zstd in feather
                        df_arrow.to_feather(filepath, compression=compression, compression_level=compression_level)
                    else:
                        df_arrow.to_feather(filepath, compression=compression)
                except Exception: # Fallback if specific dtypes are not supported by pyarrow feather
                    df_basic = df_arrow.copy()
                    for col in df_basic.columns:
                        if not pd.api.types.is_numeric_dtype(df_basic[col]) and not pd.api.types.is_string_dtype(df_basic[col]) and not pd.api.types.is_datetime64_any_dtype(df_basic[col]):
                            df_basic[col] = df_basic[col].astype(str)
                    if compression == 'zstd' and compression_level is not None:
                        df_basic.to_feather(filepath, compression=compression, compression_level=compression_level)
                    else:
                        df_basic.to_feather(filepath, compression=compression)

            elif fmt.startswith('orc'):
                df_arrow = prepare_pyarrow_compatible_df(df)
                table = pa.Table.from_pandas(df_arrow, preserve_index=False)
                if compression_level is not None:
                    orc.write_table(table, filepath, compression=compression, compression_level=compression_level)
                else:
                    orc.write_table(table, filepath, compression=compression)

            elif fmt.startswith('hdf5'):
                df_hdf5 = df.copy()
                for col in df_hdf5.columns:
                    if pd.api.types.is_categorical_dtype(df_hdf5[col]):
                        df_hdf5[col] = df_hdf5[col].astype(str) # Convert categorical to string for HDF5

                if compression is None:
                    df_hdf5.to_hdf(filepath, key='data', mode='w', format='table', complevel=0, complib=None)
                else:
                    # For HDF5, complib takes the compression name, and complevel takes the integer level
                    df_hdf5.to_hdf(filepath, key='data', mode='w', format='table', complevel=compression_level if compression_level is not None else 9, complib=compression)

            file_size = get_file_size_mb(filepath)
            write_time = time.time() - start_time
            
            results[fmt] = {
                'write_time': write_time,
                'size_mb': file_size,
                'filepath': filepath,
                'file_version': file_version
            }
            saved_file_paths[fmt] = filepath

            print(f"   {fmt.upper()} (Write): {write_time:.3f}s, {file_size:.2f}MB")

        except Exception as e:
            print(f"   {fmt.upper()} (Write): Failed - {e}")
            results[fmt] = None
            saved_file_paths[fmt] = None
            continue

    return results, saved_file_paths

def read_from_formats(saved_file_paths):
    """Read dataframes from various formats and measure read time and speed."""
    read_results = {}

    for fmt, filepath in saved_file_paths.items():
        if filepath is None or not os.path.exists(filepath):
            read_results[fmt] = {'read_time': None, 'read_speed': None}
            continue

        try:
            start_time = time.time()
            file_size_mb = get_file_size_mb(filepath) # Get size for speed calculation

            if fmt.startswith('parquet'):
                _ = pd.read_parquet(filepath)
            elif fmt.startswith('duckdb'):
                # For DuckDB, we're reading the generated Parquet file
                conn = duckdb.connect(database=':memory:')
                _ = conn.execute(f"SELECT * FROM '{filepath}'").fetchdf()
                conn.close()
            elif fmt.startswith('feather'):
                _ = pd.read_feather(filepath)
            elif fmt.startswith('orc'):
                _ = pd.read_orc(filepath)
            elif fmt.startswith('hdf5'):
                _ = pd.read_hdf(filepath, key='data')
            
            read_time = time.time() - start_time
            read_speed_mbps = file_size_mb / read_time if read_time > 0 else np.inf # MB/s

            read_results[fmt] = {
                'read_time': read_time,
                'read_speed': read_speed_mbps
            }
            print(f"   {fmt.upper()} (Read): {read_time:.3f}s, {read_speed_mbps:.2f} MB/s")

        except Exception as e:
            print(f"   {fmt.upper()} (Read): Failed - {e}")
            read_results[fmt] = {'read_time': None, 'read_speed': None}
            continue
    return read_results

def process_file(file, outpath, usecols_arg): 
    outfile = None
    try:
        start_overall = time.time()
        outfile = unzip_func(file, outpath=outpath)
        print(f"Unzipped {os.path.basename(file)} to {outfile} in {time.time() - start_overall:.2f}s")
        
        v = getvers(outfile)
        df = None

        if v == '326':
            df = readv326(outfile)
            is_std_format = True
        elif v == '325':
            df = readv325(outfile)
            is_std_format = True
        elif v == '324':
            df = readv324(outfile)
            is_std_format = True
        elif v == 'vscintpi1':
            df = readvscintpi1(outfile)
            is_std_format = False
        else:
            print(f"   Unknown file version for {os.path.basename(file)}. Skipping.")
            return None

        if df is None:
            print(f"   Could not load data from {os.path.basename(file)}. Skipping.")
            return None

        if is_std_format:
            missing_cols = [col for col in usecols_arg if col not in df.columns] 
            if missing_cols:
                print(f"   Missing columns for standard format: {missing_cols}. Skipping file processing.")
                return None
            df = df[usecols_arg] 
        
        # --- Modifications Start Here ---
        # Convert 'week' and 'towe' to 'datetime' and then delete original columns
        if 'week' in df.columns and 'towe' in df.columns:
            gps_epoch = datetime(1980, 1, 6) # GPS Epoch
            df['datetime'] = [gps_epoch + timedelta(weeks=w, seconds=t) for w, t in zip(df['week'], df['towe'])]
            df = df.drop(columns=['week', 'towe'])
        
        # Delete 'fname' column if it exists
        if 'fname' in df.columns:
            df = df.drop(columns=['fname'])
        # --- Modifications End Here ---

        #df = optimize_dtypes(df)
        
        base_path = outfile.replace('unzipped', 'processed')
        base_path = re.sub(r'\.dat$|\.bin$', '.pq', base_path)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        basename = os.path.basename(outfile)
        print(f"Converting {basename} to multiple formats:")
        print(f"   Data shape: {df.shape}, Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f}MB")

        if v in ['325', '326']:
            file_version = 'bin'
        elif v == '324':
            file_version = 'v324_dat'
        elif v == 'vscintpi1':
            file_version = 'vscintpi1_dat'
        else:
            file_version = 'unknown'

        # Save the data and get the paths and initial save results
        save_results, saved_file_paths = save_to_formats(df, base_path, basename, file_version)
        
        # Now, read the data back and measure read performance
        read_results = read_from_formats(saved_file_paths)

        # Combine save and read results
        combined_results = {}
        for fmt in save_results:
            if save_results[fmt]:
                combined_results[fmt] = save_results[fmt]
                if fmt in read_results:
                    combined_results[fmt]['read_time'] = read_results[fmt]['read_time']
                    combined_results[fmt]['read_speed'] = read_results[fmt]['read_speed']
                else:
                    combined_results[fmt]['read_time'] = None
                    combined_results[fmt]['read_speed'] = None

    except Exception as e:
        print(f"Skip {file} due to {e}")
        return None
    finally:
        if outfile and os.path.exists(outfile):
            os.remove(outfile)
        del df
        import gc; gc.collect()

    total_time = time.time() - start_overall
    print(f"Processed {file} in {total_time:.2f} secs total\n")
    return combined_results

# Define a partial function to pass the constant argument to pool.map
def wrapper_process_file(f):
    global usecols # Access the global usecols
    return process_file(f, outpath='./unzipped', usecols_arg=usecols)
    
def print_summary_stats(organized_data, zipped_original_sizes):
    if not organized_data:
        print("\nNo data processed for summary statistics.")
        return

    formats = [
        'parquet_zstd', 'parquet_lz4', 'parquet_brotli', 'parquet_gzip', 'parquet_snappy',
        'duckdb_zstd', 'duckdb_lz4',
        'feather_zstd', 'feather_lz4',
        'orc_zstd', 'orc_snappy', # Added ORC formats
        'hdf5_blosc', 'hdf5_none'
    ]

    file_types = ['bin', 'v324_dat', 'vscintpi1_dat']

    print("\n" + "="*60)
    print("CONVERSION SUMMARY STATISTICS")
    print("="*60)

    for fmt in formats:
        print(f"\n{fmt.upper()}:")
        for file_type in file_types:
            if fmt in organized_data and file_type in organized_data[fmt]:
                write_times = organized_data[fmt][file_type]['write_times']
                read_times = organized_data[fmt][file_type]['read_times']
                sizes = organized_data[fmt][file_type]['sizes']
                read_speeds = organized_data[fmt][file_type]['read_speeds']

                if write_times:
                    avg_write_time = sum(write_times) / len(write_times)
                    avg_read_time = sum(read_times) / len(read_times) if read_times else 0
                    avg_size = sum(sizes) / len(sizes)
                    avg_read_speed = sum(read_speeds) / len(read_speeds) if read_speeds else 0

                    print(f"   {file_type}: {len(write_times)} files - Avg Write: {avg_write_time:.3f}s, Avg Read: {avg_read_time:.3f}s, Avg Size: {avg_size:.2f}MB, Avg Read Speed: {avg_read_speed:.2f} MB/s")
                else:
                    print(f"   {file_type}: No data.")
            else:
                print(f"   {file_type}: No data.")

    create_conversion_plots(organized_data, formats, file_types, zipped_original_sizes)


def create_conversion_plots(organized_data, formats, file_types, zipped_original_sizes):
    plt.style.use('default')
    sns.set_palette("colorblind")

    fig = plt.figure(figsize=(24, 32), constrained_layout=True)
    gs = fig.add_gridspec(8, 2) # Increased rows to accommodate new plots
    
    format_colors = {
        'parquet_lz4': '#1f77b4',
        'feather_lz4': '#ff7f0e',
        'duckdb_lz4': '#2ca02c',
        'parquet_zstd': '#d62728',
        'parquet_brotli': '#9467bd',
        'parquet_gzip': '#8c564b',
        'parquet_snappy': '#e377c2',
        'feather_zstd': '#7f7f7f',
        'duckdb_zstd': '#bcbd22',
        'orc_zstd': '#ff00ff', # Added ORC ZSTD color
        'orc_snappy': '#00ffff', # Added ORC Snappy color
        'hdf5_blosc': '#17becf',
        'hdf5_none': '#aec7e8'
    }
    
    def format_size_label(value):
        return f'{value:.1f}MB'

    # Use zipped original sizes for baseline calculation
    avg_zipped_bin_size = np.mean(zipped_original_sizes.get('bin_zipped_sizes', [1])) if zipped_original_sizes.get('bin_zipped_sizes') else 1
    avg_zipped_v324_dat_size = np.mean(zipped_original_sizes.get('v324_dat_zipped_sizes', [1])) if zipped_original_sizes.get('v324_dat_zipped_sizes') else 1
    avg_zipped_vscintpi1_dat_size = np.mean(zipped_original_sizes.get('vscintpi1_dat_zipped_sizes', [1])) if zipped_original_sizes.get('vscintpi1_dat_zipped_sizes') else 1

    zipped_size_baselines = {
        'bin': avg_zipped_bin_size,
        'v324_dat': avg_zipped_v324_dat_size,
        'vscintpi1_dat': avg_zipped_vscintpi1_dat_size
    }

    # --- Graph 1: Percentage of Zipped Data Size for LZ4 Formats (Bin Files) ---
    ax1 = fig.add_subplot(gs[0, 0])
    methods_lz4 = ['parquet_lz4', 'feather_lz4', 'duckdb_lz4']
    x = np.arange(len(methods_lz4))
    
    size_ratios_lz4_bin = []
    size_labels_lz4_bin = []
    
    for fmt in methods_lz4:
        if fmt in organized_data and 'bin' in organized_data[fmt] and organized_data[fmt]['bin']['sizes']:
            avg_fmt_size = np.mean(organized_data[fmt]['bin']['sizes'])
            # Calculate ratio based on zipped original size
            ratio = (avg_fmt_size / zipped_size_baselines['bin']) * 100
            size_ratios_lz4_bin.append(ratio)
            size_labels_lz4_bin.append(format_size_label(avg_fmt_size))
        else:
            size_ratios_lz4_bin.append(0)
            size_labels_lz4_bin.append("0MB")

    bars1 = ax1.bar(x, size_ratios_lz4_bin, color=[format_colors[m] for m in methods_lz4])
    ax1.set_title('Compressed Size % of Zipped Data (v325+326 - LZ4)', fontweight='bold', fontsize=14)
    ax1.set_ylabel('% of Zipped Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').upper() for m in methods_lz4])
    ax1.set_ylim(0, max(size_ratios_lz4_bin) * 1.2 if size_ratios_lz4_bin and max(size_ratios_lz4_bin) > 0 else 100)
    
    for i, bar in enumerate(bars1):
        if size_ratios_lz4_bin[i] > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{size_ratios_lz4_bin[i]:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() - size_ratios_lz4_bin[i]*0.1,
                     size_labels_lz4_bin[i], ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # --- Graph 2: File Write Time for LZ4 Formats (Bin Files) ---
    ax2 = fig.add_subplot(gs[0, 1])
    write_times_lz4_bin = []
    for fmt in methods_lz4:
        if fmt in organized_data and 'bin' in organized_data[fmt] and organized_data[fmt]['bin']['write_times']:
            write_times_lz4_bin.append(np.mean(organized_data[fmt]['bin']['write_times']))
        else:
            write_times_lz4_bin.append(0)
    
    bars2 = ax2.bar(x, write_times_lz4_bin, color=[format_colors[m] for m in methods_lz4])
    ax2.set_title('File Write Time (v325+326 - LZ4)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Time (seconds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('_', ' ').upper() for m in methods_lz4])
    ax2.set_ylim(0, max(write_times_lz4_bin) * 1.2 if write_times_lz4_bin and max(write_times_lz4_bin) > 0 else 10)
    
    for i, bar in enumerate(bars2):
        if write_times_lz4_bin[i] > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(write_times_lz4_bin)*0.02,
                     f'{write_times_lz4_bin[i]:.3f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Graph 3: File Read Time for LZ4 Formats (Bin Files) ---
    ax3 = fig.add_subplot(gs[1, 0])
    read_times_lz4_bin = []
    for fmt in methods_lz4:
        if fmt in organized_data and 'bin' in organized_data[fmt] and organized_data[fmt]['bin']['read_times']:
            read_times_lz4_bin.append(np.mean(organized_data[fmt]['bin']['read_times']))
        else:
            read_times_lz4_bin.append(0)
    
    bars3 = ax3.bar(x, read_times_lz4_bin, color=[format_colors[m] for m in methods_lz4])
    ax3.set_title('File Read Time (v325+326 - LZ4)', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace('_', ' ').upper() for m in methods_lz4])
    ax3.set_ylim(0, max(read_times_lz4_bin) * 1.2 if read_times_lz4_bin and max(read_times_lz4_bin) > 0 else 1)
    
    for i, bar in enumerate(bars3):
        if read_times_lz4_bin[i] > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(read_times_lz4_bin)*0.02,
                     f'{read_times_lz4_bin[i]:.3f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Graph 4: Read Speed for LZ4 Formats (Bin Files) ---
    ax4 = fig.add_subplot(gs[1, 1])
    read_speeds_lz4_bin = []
    for fmt in methods_lz4:
        if fmt in organized_data and 'bin' in organized_data[fmt] and organized_data[fmt]['bin']['read_speeds']:
            read_speeds_lz4_bin.append(np.mean(organized_data[fmt]['bin']['read_speeds']))
        else:
            read_speeds_lz4_bin.append(0)
    
    bars4 = ax4.bar(x, read_speeds_lz4_bin, color=[format_colors[m] for m in methods_lz4])
    ax4.set_title('File Read Speed (v325+326 - LZ4)', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Speed (MB/s)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m.replace('_', ' ').upper() for m in methods_lz4])
    ax4.set_ylim(0, max(read_speeds_lz4_bin) * 1.2 if read_speeds_lz4_bin and max(read_speeds_lz4_bin) > 0 else 100)
    
    for i, bar in enumerate(bars4):
        if read_speeds_lz4_bin[i] > 0:
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(read_speeds_lz4_bin)*0.02,
                     f'{read_speeds_lz4_bin[i]:.2f}MB/s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Graph 5: Parquet Compression Comparison - Size (% of Zipped) (All Compressors, Bin Files) ---
    ax5 = fig.add_subplot(gs[2, 0])
    parquet_methods = ['parquet_lz4', 'parquet_brotli', 'parquet_gzip', 'parquet_snappy', 'parquet_zstd']
    x5 = np.arange(len(parquet_methods))
    
    parquet_ratios_bin = []
    parquet_labels_bin = []
    
    for fmt in parquet_methods:
        if fmt in organized_data and 'bin' in organized_data[fmt] and organized_data[fmt]['bin']['sizes']:
            avg_fmt_size = np.mean(organized_data[fmt]['bin']['sizes'])
            # Calculate ratio based on zipped original size
            ratio = (avg_fmt_size / zipped_size_baselines['bin']) * 100
            parquet_ratios_bin.append(ratio)
            parquet_labels_bin.append(format_size_label(avg_fmt_size))
        else:
            parquet_ratios_bin.append(0)
            parquet_labels_bin.append("0MB")

    bars5 = ax5.bar(x5, parquet_ratios_bin, color=[format_colors[m] for m in parquet_methods])
    ax5.set_title('Parquet Compression Comparison - Size (v325+326)', fontweight='bold', fontsize=14)
    ax5.set_ylabel('% of Zipped Size')
    ax5.set_xticks(x5)
    ax5.set_xticklabels([m.replace('parquet_', '').upper() for m in parquet_methods])
    ax5.set_ylim(0, max(parquet_ratios_bin) * 1.2 if parquet_ratios_bin and max(parquet_ratios_bin) > 0 else 100)
    
    for i, bar in enumerate(bars5):
        if parquet_ratios_bin[i] > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{parquet_ratios_bin[i]:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() - parquet_ratios_bin[i]*0.1,
                     parquet_labels_bin[i], ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # --- Graph 7: Parquet Read Times (All Compressors, Bin Files) ---
    ax7 = fig.add_subplot(gs[3, 0])
    parquet_read_times_bin = []
    for fmt in parquet_methods:
        if fmt in organized_data and 'bin' in organized_data[fmt] and organized_data[fmt]['bin']['read_times']:
            parquet_read_times_bin.append(np.mean(organized_data[fmt]['bin']['read_times']))
        else:
            parquet_read_times_bin.append(0)
    
    bars7 = ax7.bar(x5, parquet_read_times_bin, color=[format_colors[m] for m in parquet_methods])
    ax7.set_title('Parquet File Read Time (v325+326)', fontweight='bold', fontsize=14)
    ax7.set_ylabel('Time (seconds)')
    ax7.set_xticks(x5)
    ax7.set_xticklabels([m.replace('parquet_', '').upper() for m in parquet_methods])
    ax7.set_ylim(0, max(parquet_read_times_bin) * 1.2 if parquet_read_times_bin and max(parquet_read_times_bin) > 0 else 1)
    
    for i, bar in enumerate(bars7):
        if parquet_read_times_bin[i] > 0:
            ax7.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(parquet_read_times_bin)*0.02,
                     f'{parquet_read_times_bin[i]:.3f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Graph 8: Parquet Read Speeds (All Compressors, Bin Files) ---
    ax8 = fig.add_subplot(gs[3, 1])
    parquet_read_speeds_bin = []
    for fmt in parquet_methods:
        if fmt in organized_data and 'bin' in organized_data[fmt] and organized_data[fmt]['bin']['read_speeds']:
            parquet_read_speeds_bin.append(np.mean(organized_data[fmt]['bin']['read_speeds']))
        else:
            parquet_read_speeds_bin.append(0)
    
    bars8 = ax8.bar(x5, parquet_read_speeds_bin, color=[format_colors[m] for m in parquet_methods])
    ax8.set_title('Parquet File Read Speed (v325+326)', fontweight='bold', fontsize=14)
    ax8.set_ylabel('Speed (MB/s)')
    ax8.set_xticks(x5)
    ax8.set_xticklabels([m.replace('parquet_', '').upper() for m in parquet_methods])
    ax8.set_ylim(0, max(parquet_read_speeds_bin) * 1.2 if parquet_read_speeds_bin and max(parquet_read_speeds_bin) > 0 else 100)
    
    for i, bar in enumerate(bars8):
        if parquet_read_speeds_bin[i] > 0:
            ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(parquet_read_speeds_bin)*0.02,
                     f'{parquet_read_speeds_bin[i]:.2f}MB/s', ha='center', va='bottom', fontsize=11, fontweight='bold')


    # --- Graph 9: Average File Size by Type (Selected Formats - Absolute Sizes) ---
    ax9 = fig.add_subplot(gs[4, :]) # Changed to gs[4, :] to take full width
    selected_formats_for_type_comparison = ['parquet_zstd', 'duckdb_zstd', 'feather_lz4', 'orc_zstd'] # Added ORC ZSTD
    group_width = 0.8
    bar_width = group_width / len(selected_formats_for_type_comparison)
    x9 = np.arange(len(file_types))

    for i, fmt in enumerate(selected_formats_for_type_comparison):
        avg_sizes = []
        for ft in file_types:
            if fmt in organized_data and ft in organized_data[fmt] and organized_data[fmt][ft]['sizes']:
                avg_sizes.append(np.mean(organized_data[fmt][ft]['sizes']))
            else:
                avg_sizes.append(0)
        
        bar_positions = x9 - group_width/2 + i * bar_width + bar_width/2
        bars9 = ax9.bar(bar_positions, avg_sizes, bar_width,
                         label=fmt.replace('_', ' ').upper(),
                         color=format_colors.get(fmt, 'gray'))

        for bar, val in zip(bars9, avg_sizes):
            if val > 0:
                ax9.text(bar.get_x() + bar.get_width()/2., val + (max(avg_sizes)*0.01 if avg_sizes else 0.5),
                         f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax9.set_title('Average File Size by Type (Selected Formats)', fontweight='bold', fontsize=14)
    ax9.set_ylabel('Size (MB)')
    ax9.set_xticks(x9)
    ax9.set_xticklabels(['Binary (v325+326)', 'v324 DAT', 'vscintpi1 DAT'])
    ax9.legend(loc='upper left', fontsize=8, ncol=len(selected_formats_for_type_comparison)) # Adjusted ncol
    ax9.set_ylim(0, max([max(organized_data[fmt][ft]['sizes']) for fmt in selected_formats_for_type_comparison for ft in file_types if fmt in organized_data and ft in organized_data[fmt] and organized_data[fmt][ft]['sizes']] ) * 1.2 if any(organized_data[fmt][ft]['sizes'] for fmt in selected_formats_for_type_comparison for ft in file_types if fmt in organized_data and ft in organized_data[fmt]) else 100)


    plt.suptitle('GNSS Compression Performance Analysis (No Masks Applied)', fontsize=20, fontweight='bold')
    plt.savefig('compression_performance_no_mask.png', dpi=300, bbox_inches='tight')
    print("\n📊 Updated visualization saved as 'compression_performance_no_mask.png'")
    plt.show()

if __name__ == '__main__':
    base_dirs = ['./files/v324', './files/v325', './files/v326', './files/vscintpi2', './files/vscintpi1']
    zip_files = []
    for dir in base_dirs:
        zip_files.extend(glob.glob(os.path.join(dir, '**', '*.zip'), recursive=True))
    print(f"Found {len(zip_files)} zip files.")

    try:
        import pyarrow as pa
        print("pyarrow is available - Feather format will use PyArrow engine")
    except ImportError:
        print("pyarrow not found - Feather format will use fallback engine")
        print("    For better Feather performance, install with: pip install pyarrow")

    # Modified: Collect zipped file sizes instead of unzipped
    zipped_original_sizes_by_type = {'bin_zipped_sizes': [], 'v324_dat_zipped_sizes': [], 'vscintpi1_dat_zipped_sizes': []}

    all_results = []

    print("\nCalculating original zipped file sizes...")
    for f in zip_files:
        try:
            zipped_file_size_mb = get_file_size_mb(f)
            
            # Here, we're assuming the zip file name reflects the content type, similar to getvers.
            # You might need to adjust this logic based on your actual zip file naming conventions.
            if 'v325' in f.lower() or 'v326' in f.lower():
                zipped_original_sizes_by_type['bin_zipped_sizes'].append(zipped_file_size_mb)
            elif 'v324' in f.lower():
                zipped_original_sizes_by_type['v324_dat_zipped_sizes'].append(zipped_file_size_mb)
            elif 'vscintpi1' in f.lower():
                zipped_original_sizes_by_type['vscintpi1_dat_zipped_sizes'].append(zipped_file_size_mb)
            else:
                print(f"Warning: Could not determine type for zipped file {f}. Skipping for zipped size comparison.")

        except Exception as e:
            print(f"Could not get zipped size for {f}: {e}")
    print("Original zipped file size calculation complete.\n")


    if workers == 1:
        print("Processing files in single-threaded mode...")
        for f in zip_files:
            result = process_file(f, outpath='./unzipped', usecols_arg=usecols) 
            if result:
                all_results.append(result)
    else:
        print(f"Processing files with a pool of {workers} workers...")
        with Pool(workers) as pool:
            results_from_pool = pool.map(wrapper_process_file, zip_files)
            all_results = [res for res in results_from_pool if res is not None]
    
    final_organized_data = {}
    
    formats_list = [
        'parquet_zstd', 'parquet_lz4', 'parquet_brotli', 'parquet_gzip', 'parquet_snappy',
        'duckdb_zstd', 'duckdb_lz4',
        'feather_zstd', 'feather_lz4',
        'orc_zstd', 'orc_snappy', # Added ORC formats
        'hdf5_blosc', 'hdf5_none'
    ]
    file_types_list = ['bin', 'v324_dat', 'vscintpi1_dat']

    for fmt in formats_list:
        final_organized_data[fmt] = {}
        for f_type in file_types_list:
            # Now store separate lists for write times, read times, sizes, and read speeds
            final_organized_data[fmt][f_type] = {'write_times': [], 'read_times': [], 'sizes': [], 'read_speeds': []} 

    for result_set in all_results:
        if result_set:
            for fmt, data in result_set.items():
                if data and fmt in formats_list:
                    file_version = data.get('file_version', 'unknown')
                    if file_version in file_types_list:
                        final_organized_data[fmt][file_version]['write_times'].append(data['write_time'])
                        final_organized_data[fmt][file_version]['sizes'].append(data['size_mb'])
                        # Only append if read_time and read_speed are not None
                        if data['read_time'] is not None:
                            final_organized_data[fmt][file_version]['read_times'].append(data['read_time'])
                        if data['read_speed'] is not None:
                            final_organized_data[fmt][file_version]['read_speeds'].append(data['read_speed'])

    # Pass the collected zipped_original_sizes_by_type to the summary and plotting functions
    print_summary_stats(final_organized_data, zipped_original_sizes_by_type)