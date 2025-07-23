import glob, pandas as pd, numpy as np, time, zipfile, os, re, struct
from multiprocessing import Pool
import duckdb
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

gnssdic = {0: 'GPS', 1: 'SBS', 2: 'GAL', 3: 'BDS', 6: 'GLO'}
workers = 1 

usecols = ['cons', 'svid', 'week', 'towe', 'elev', 'azim',
           'snr1', 'snr2', 'cph1', 'cph2', 'rng1', 'rng2']

conversion_stats = {'parquet': [], 'duckdb': [], 'feather': [], 'hdf5': []}

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
        ('snr1', np.uint8), ('snr2', np.uint8), ('snr3', np.uint8),
        ('pst1', np.uint8), ('pst2', np.uint8), ('pst3', np.uint8),
        ('rst1', np.uint8), ('rst2', np.uint8), ('rst3', np.uint8),
        ('cph1', np.float64), ('cph2', np.float64), ('cph3', np.float64),
        ('rng1', np.float64), ('rng2', np.float64), ('rng3', np.float64),
        ('lon', np.float32), ('lat', np.float32), ('hei', np.float32),
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
    """Optimize data types to reduce memory usage"""
    for col in df.columns:
        if col in ['week', 'svid', 'elev', 'azim']:
            if df[col].dtype != 'int8' and df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            elif df[col].dtype != 'int16' and df[col].min() >= -32768 and df[col].max() <= 32767:
                df[col] = df[col].astype('int16')
            elif df[col].dtype != 'int32':
                df[col] = df[col].astype('int32')
        elif col in ['snr1', 'snr2'] and df[col].dtype != 'uint8':
            if df[col].min() >= 0 and df[col].max() <= 255:
                df[col] = df[col].astype('uint8')
        elif col in ['towe', 'cph1', 'cph2', 'rng1', 'rng2'] and df[col].dtype == 'float64':
            df[col] = df[col].astype('float64')
    return df

def prepare_feather_compatible_df(df):
    """Prepare dataframe for Feather format compatibility"""
    df_feather = df.copy()

    for col in df_feather.columns:
        if pd.api.types.is_categorical_dtype(df_feather[col]):
            df_feather[col] = df_feather[col].astype(str)

    for col in df_feather.columns:
        if pd.api.types.is_integer_dtype(df_feather[col]):

            pass
        elif pd.api.types.is_float_dtype(df_feather[col]):

            pass

    return df_feather

def save_to_formats(df, base_path, basename, file_version, elmask_val):
    """Save dataframe to Parquet (ZSTD); record timing/size"""
    formats = {
        'parquet_zstd':   {'ext': '.parquet', 'folder': 'pq_zstd',   'compression': 'zstd'},
    }

    results = {}

    for fmt, info in formats.items():
        try:
            start_time = time.time()

            base_dir = os.path.dirname(base_path)
            format_dir = os.path.join(base_dir, f"{info['folder']}_el{elmask_val}")
            os.makedirs(format_dir, exist_ok=True)

            filename = os.path.basename(base_path).replace('.pq', info['ext'])
            filepath = os.path.join(format_dir, filename)
            compression = info['compression']

            df.to_parquet(filepath, compression=compression, index=False)

            file_size = get_file_size_mb(filepath)
            time_taken = time.time() - start_time
            io_speed = file_size / time_taken if time_taken > 0 else 0

            results[fmt] = {
                'time': time_taken,
                'size_mb': file_size,
                'io_speed': io_speed,
                'filepath': filepath,
                'file_version': file_version,
                'elevation_mask': elmask_val
            }

            print(f"  {fmt.upper()} (Mask {elmask_val}): {time_taken:.3f}s, {file_size:.2f}MB, {io_speed:.1f}MB/s")

        except Exception as e:
            print(f"  {fmt.upper()} (Mask {elmask_val}): Failed - {e}")

            results[fmt] = None 
            continue

    return results

def process_file_with_masks(file, outpath, elevation_masks):
    all_mask_results = {}
    outfile = None 
    try:
        start_unzip = time.time()
        outfile = unzip_func(file, outpath=outpath)
        print(f"Unzipped {os.path.basename(file)} to {outfile} in {time.time() - start_unzip:.2f}s")

        v = getvers(outfile)
        df_raw = None 

        if v == '326':
            df_raw = readv326(outfile)
            is_std_format = True
        elif v == '325':
            df_raw = readv325(outfile)
            is_std_format = True
        elif v == '324':
            df_raw = readv324(outfile)
            is_std_format = True
        elif v == 'vscintpi1':
            df_raw = readvscintpi1(outfile)
            is_std_format = False
        else:
            print(f"  Unknown file version for {os.path.basename(file)}. Skipping.")
            if os.path.exists(outfile):
                os.remove(outfile)
            return None 

        if df_raw is None: 
            print(f"  Could not load data from {os.path.basename(file)}. Skipping.")
            if os.path.exists(outfile):
                os.remove(outfile)
            return None

        if is_std_format:
            df_raw = df_raw[usecols] 

            df_raw['fname'] = pd.Categorical([os.path.basename(outfile)] * len(df_raw))
            df_raw = optimize_dtypes(df_raw)
        else: 
            df_raw['fname'] = pd.Categorical([os.path.basename(outfile)] * len(df_raw))

            df_raw = optimize_dtypes(df_raw) 

        base_path = outfile.replace('unzipped', 'processed')
        base_path = re.sub(r'\.dat$|\.bin$', '.pq', base_path)
        os.makedirs(os.path.dirname(base_path), exist_ok=True)

        basename = os.path.basename(outfile)

        if v in ['325', '326']:
            file_version = 'bin'
        elif v == '324':
            file_version = 'v324_dat'
        elif v == 'vscintpi1':
            file_version = 'vscintpi1_dat'
        else:
            file_version = 'unknown'

        for elmask_val in elevation_masks:
            print(f"Processing {basename} with elevation mask: {elmask_val}")

            df_to_save = df_raw.copy() 

            if is_std_format:

                if 'elev' in df_to_save.columns:
                    df_to_save = df_to_save[df_to_save['elev'] > elmask_val]

                if df_to_save.empty:
                    print(f"  No data after applying elevation mask {elmask_val} for {basename}. Skipping save.")
                    all_mask_results[elmask_val] = None
                    continue 

                print(f"  Data shape (Mask {elmask_val}): {df_to_save.shape}, Memory usage: {df_to_save.memory_usage(deep=True).sum() / 1024**2:.2f}MB")
            else: 
                print(f"  vscintpi1 format, not applying elevation mask on 'elev'.")
                print(f"  Data shape (Mask {elmask_val}): {df_to_save.shape}, Memory usage: {df_to_save.memory_usage(deep=True).sum() / 1024**2:.2f}MB")

            format_results = save_to_formats(df_to_save, base_path, basename, file_version, elmask_val)
            all_mask_results[elmask_val] = format_results

            del df_to_save 
            import gc; gc.collect()

        if os.path.exists(outfile):
            os.remove(outfile)

        del df_raw
        import gc; gc.collect()

        print(f"Finished processing {file}\n")
        return all_mask_results

    except Exception as e:
        print(f"Skip {file} due to {e}")
        if outfile and os.path.exists(outfile): 
            os.remove(outfile)
        return None

def wrapper_process_file_with_masks(f, elevation_masks):
    return process_file_with_masks(f, outpath='./unzipped', elevation_masks=elevation_masks)

def aggregate_results_by_mask(all_processed_results):
    """Aggregates results from multiple files and masks."""
    aggregated = {}
    for file_results_by_mask in all_processed_results:
        if file_results_by_mask:
            for elmask_val, format_results in file_results_by_mask.items():
                if format_results: 
                    if elmask_val not in aggregated:
                        aggregated[elmask_val] = {}
                    for fmt, data in format_results.items():
                        if data: 
                            if fmt not in aggregated[elmask_val]:
                                aggregated[elmask_val][fmt] = {
                                    'times': [],
                                    'sizes': [],
                                    'io_speeds': [],
                                    'file_versions': []
                                }
                            aggregated[elmask_val][fmt]['times'].append(data['time'])
                            aggregated[elmask_val][fmt]['sizes'].append(data['size_mb'])
                            aggregated[elmask_val][fmt]['io_speeds'].append(data['io_speed'])
                            aggregated[elmask_val][fmt]['file_versions'].append(data['file_version'])
    return aggregated

def create_single_comparison_plot(aggregated_data):
    """
    Create a single plot comparing compressed sizes (relative to 0-degree mask)
    across elevation masks using only parquet_zstd, with a wider layout.
    """
    plt.style.use('default')
    sns.set_palette("colorblind")

    fig, ax = plt.subplots(1, 1, figsize=(14, 7), constrained_layout=True)

    mask_colors = {
        0: '#1f77b4', 
        5: '#ff7f0e', 
        10: '#2ca02c', 
        15: '#d62728', 
        20: '#9467bd', 
        25: '#8c564b', 
        30: '#e377c2', 
        35: '#7f7f7f', 
        40: '#bcbd22', 
    }

    selected_format = 'parquet_zstd'

    avg_size_el0 = 1.0 

    if 0 in aggregated_data and selected_format in aggregated_data[0] and aggregated_data[0][selected_format]['sizes']:
        sizes_el0 = aggregated_data[0][selected_format]['sizes']
        avg_size_el0 = np.mean(sizes_el0)
        print(f"Average size for 0-degree mask ({selected_format}): {avg_size_el0:.2f} MB")
    else:
        print(f"No valid data found for 0-degree mask ({selected_format}). Percentages will be based on an assumed 1MB baseline.")

    comparison_masks = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    mask_labels = [f'{mask}°' for mask in comparison_masks]
    size_percentages = []

    for mask in comparison_masks:
        if mask in aggregated_data and selected_format in aggregated_data[mask] and aggregated_data[mask][selected_format]['sizes']:
            sizes = aggregated_data[mask][selected_format]['sizes']

            avg_size = np.mean(sizes)
            percentage = (avg_size / avg_size_el0) * 100 if avg_size_el0 > 0 else 0 
            size_percentages.append(percentage)
            print(f"Average size for {mask}-degree mask ({selected_format}): {avg_size:.2f} MB, Percentage of 0-degree: {percentage:.1f}%")
        else:
            size_percentages.append(0)
            print(f"No valid data found for {mask}-degree mask ({selected_format}).")

    x = np.arange(len(comparison_masks))

    bars = ax.bar(x, size_percentages, color=[mask_colors[m] for m in comparison_masks])
    ax.set_title(f'Compressed Size % Relative to 0° Mask ({selected_format.replace("_", " ").upper()})', fontweight='bold', fontsize=14)
    ax.set_ylabel('% of 0° Mask Size')
    ax.set_xlabel('Elevation Mask Angle') 
    ax.set_xticks(x)
    ax.set_xticklabels(mask_labels)

    y_max = max(size_percentages) * 1.15 if size_percentages and max(size_percentages) > 0 else 100
    ax.set_ylim(0, y_max)

    for i, bar in enumerate(bars):
        if size_percentages[i] > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + (y_max * 0.01), 
                            f'{size_percentages[i]:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Impact of Elevation Mask on Parquet ZSTD Compressed Size', fontsize=16, fontweight='bold')
    plt.savefig('elevation_mask_parquet_zstd_size_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Updated visualization saved as 'elevation_mask_parquet_zstd_size_comparison.png'")
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
        print("      For better Feather performance, install with: pip install pyarrow")

    elevation_masks_to_process = [0, 5, 10, 15, 20, 25, 30, 35, 40]

    all_results_by_mask = []

    for f in zip_files:
        file_mask_results = wrapper_process_file_with_masks(f, elevation_masks_to_process)
        if file_mask_results:
            all_results_by_mask.append(file_mask_results)

    aggregated_data = aggregate_results_by_mask(all_results_by_mask)
    create_single_comparison_plot(aggregated_data)