import mlflow.sklearn
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

import hashlib
import base64
import cloudpickle

def main(
        working_dir,
        dataset_name,
        A_modelpath,
        B_modelpath,
        hash_function_string=None,
        split_function_string=None,
        seed=42,
        cutoff_year=2020,
):
    if hash_function_string is not None:
        try:
            func_bytes = base64.b64decode(hash_function_string.encode("utf-8"))
            hash_fn = cloudpickle.loads(func_bytes)
        except:
            raise ValueError("Could not load hash function from pickle file. Did you pass a cloudpickle.dumps object?")
    else:
        def hash_fn(datetime, seed):
            """Generates a hash from datetime and a seed, then returns an float between 0 and 1."""
            data_str = f"{seed}_{datetime}"
            hash_obj = hashlib.sha256(data_str.encode('utf-8'))
            hash_int = int(hash_obj.hexdigest(), 16)
            return hash_int % 1000 / 1000.0

    if split_function_string is not None:
        try:
            func_bytes = base64.b64decode(split_function_string.encode("utf-8"))
            split_fn = cloudpickle.loads(func_bytes)
        except:
            raise ValueError("Could not load splitter function from pickle file. Did you pass a cloudpickle.dumps object?")
    else:
        def split_fn(h, seed):
            if h < 0.33:
                return -1
            elif h < 0.66:
                return 0
            else:
                return 1


    input_path = infile_dir / infile_name
    dataset = pd.read_csv(input_path, index_col=False)
    dataset.release_date = pd.to_datetime(dataset.release_date)

    input_cols = ['release_date', 'price', 'positive_reviews', 'negative_reviews', 'metacritic_score', 'peak_ccu', 'recommendations', 'required_age', 'on_linux', 'on_mac', 'on_windows']
    X, y = dataset[input_cols], dataset['estimated_owners']

    mask = dataset.release_date.dt.year >= cutoff_year

    X_new = X[mask]
    y_new = y[mask]

    X_new['hash'] = X_new.release_date.apply(hash_fn, seed)
    X_new['group'] = X_new.hash.apply(split_fn, seed)

    mask_A = X_new['group'] == -1
    mask_B = X_new['group'] == 1

    X_A = X_new[mask_A]
    y_A = y_new[mask_A]

    X_B = X_new[mask_B]
    y_B = y_new[mask_B]


    X_A.drop(columns=['group', 'hash', 'release_date'], inplace=True)
    X_B.drop(columns=['group', 'hash', 'release_date'], inplace=True)

    model_A = mlflow.sklearn.load_model(A_modelpath)
    model_B = mlflow.sklearn.load_model(B_modelpath)

    y_pred_A = model_A.predict(X_A)
    y_pred_B = model_B.predict(X_B)

    accuracy_A = accuracy_score(y_A, y_pred_A)
    f1_A = f1_score(y_A, y_pred_A, average='macro')
    balanced_accuracy_A = balanced_accuracy_score(y_A, y_pred_A)

    accuracy_B = accuracy_score(y_B, y_pred_B)
    f1_B = f1_score(y_B, y_pred_B, average='macro')
    balanced_accuracy_B = balanced_accuracy_score(y_B, y_pred_B)

    return {
        "results_A" : {
            "accuracy": accuracy_A,
            "balanced_accuracy": balanced_accuracy_A,
            "f1": f1_A,
            "groupsize": len(mask_A),
            "model": A_modelpath
        },
        "results_B" : {
            "accuracy": accuracy_B,
            "balanced_accuracy": balanced_accuracy_B,
            "f1": f1_B,
            "groupsize": len(mask_B),
            "model": B_modelpath
        }
    }

if __name__ == "__main__":
    pass


