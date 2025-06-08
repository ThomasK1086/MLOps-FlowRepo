import mlflow.sklearn
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score


def main(
        working_dir,
        dataset_name,
        modelpath
):
    input_path = working_dir / dataset_name
    dataset = pd.read_csv(input_path, index_col=False)

    input_cols = ['price', 'positive_reviews', 'negative_reviews', 'metacritic_score', 'peak_ccu', 'recommendations', 'required_age', 'on_linux', 'on_mac', 'on_windows']
    X, y = dataset[input_cols], dataset['estimated_owners']

    model = mlflow.sklearn.load_model(modelpath)

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    balanced_accuracy = balanced_accuracy_score(y, y_pred)

    return {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "groupsize": len(X),
            "model": modelpath
        },