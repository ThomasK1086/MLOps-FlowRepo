import pandas as pd

import hashlib
import base64
import cloudpickle

def main(
        working_dir,
        dataset_name,
        hash_function_string=None,
        split_function_string=None,
        seed=42,
        cutoff_year=2020,
):
    if hash_function_string is not None:
        try:
            hash_func_bytes = base64.b64decode(hash_function_string.encode("utf-8"))
            hash_fn = cloudpickle.loads(hash_func_bytes)
        except:
            raise ValueError("Could not load hash function from pickle file. Did you pass a cloudpickle.dumps object?")
    else:
        def hash_fn(datetime, seed=42):
            """Generates a hash from datetime and a seed, then returns a float between 0 and 1."""
            data_str = f"{seed}_{datetime}"
            hash_obj = hashlib.sha256(data_str.encode('utf-8'))
            hash_int = int(hash_obj.hexdigest(), 16)
            return hash_int % 1000 / 1000.0

    if split_function_string is not None:
        try:
            split_func_bytes = base64.b64decode(split_function_string.encode("utf-8"))
            split_fn = cloudpickle.loads(split_func_bytes)
        except:
            raise ValueError("Could not load splitter function from pickle file. Did you pass a cloudpickle.dumps object?")
    else:
        def split_fn(h, seed=42):
            if h < 0.33:
                return -1
            elif h < 0.66:
                return 0
            else:
                return 1


    input_path = working_dir / dataset_name
    dataset = pd.read_csv(input_path, index_col=False)
    dataset.release_date = pd.to_datetime(dataset.release_date)

    input_cols = ['release_date', 'price', 'positive_reviews', 'negative_reviews', 'metacritic_score', 'peak_ccu', 'recommendations', 'required_age', 'on_linux', 'on_mac', 'on_windows']
    input_cols.append('estimated_owners')
    ds = dataset[input_cols]

    mask = ds.release_date.dt.year >= cutoff_year

    ds = ds.loc[mask]

    def split_fn_seeded(x):
        return split_fn(x, seed=seed)
    def hash_fn_seeded(x):
        return hash_fn(x, seed=seed)

    ds['hash'] = ds.release_date.apply(hash_fn_seeded)
    ds['group'] = ds.hash.apply(split_fn_seeded)

    mask_A = ds['group'] == -1
    mask_B = ds['group'] == 1

    ds_A = ds.loc[mask_A]
    ds_B = ds.loc[mask_B]

    ds_A.drop(columns=['group', 'hash', 'release_date'], inplace=True)
    ds_B.drop(columns=['group', 'hash', 'release_date'], inplace=True)

    outpath_A = input_path.stem + "_A.csv"
    outpath_B = input_path.stem + "_B.csv"

    ds_A.to_csv(outpath_A, index=False)
    ds_B.to_csv(outpath_B, index=False)

    return outpath_A.name, outpath_B.name


if __name__ == "__main__":
    pass
