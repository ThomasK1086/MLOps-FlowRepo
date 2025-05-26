import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

from evidently.test_suite import TestSuite
from evidently.tests import TestAccuracyScore, TestF1Score, TestRecallByClass
from evidently.test_preset import MulticlassClassificationTestPreset, DataDriftTestPreset, DataStabilityTestPreset

def main(
        infile_dir,
        infile_name,
        report_name,
        model_name,
        model_version=None,
        model_alias=None,
        model_path=None,
        cutoff_year=2020,
):
    input_path = infile_dir / infile_name
    dataset = pd.read_csv(input_path, index_col=False)
    dataset.release_date = pd.to_datetime(dataset.release_date)

    model_version = "latest" if model_version is None else model_version


    # Load the model from the Model Registry
    if model_path is not None:
        model_uri = model_path
    elif model_alias is not None:
        model_uri = f"models:/{model_name}@{model_alias}"
    else:
        model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.sklearn.load_model(model_uri)

    input_cols = ['price', 'positive_reviews', 'negative_reviews', 'metacritic_score', 'peak_ccu', 'recommendations', 'required_age', 'on_linux', 'on_mac', 'on_windows']
    X, y = dataset[input_cols], dataset['estimated_owners']

    mask = dataset.release_date.dt.year >= cutoff_year
    X_old = X[~mask]
    y_old = y[~mask]
    y_old_pred = model.predict(X_old)

    X_new = X[mask]
    y_new = y[mask]
    y_new_pred = model.predict(X_new)

    # Create a DataFrame with actual and predicted values
    df_old = pd.DataFrame({
        'target': y_old,
        'prediction': y_old_pred
    })
    df_new = pd.DataFrame({
        'target': y_new,
        'prediction': y_new_pred
    })


    # Define test suite
    test_suite = TestSuite(
        tests=[
            DataDriftTestPreset(),
            DataStabilityTestPreset()
        ]
    )

    # Run the test suite
    test_suite.run(reference_data=X_old, current_data=X_new)

    # Save the test results to HTML
    test_suite.save_html(str(infile_dir / report_name))

    return

if __name__ == "__main__":
    output_dir = Path('./data_flow1')
    outfile_name = "steam_games_dataset.csv"
    report_name = "datadrift_results.html"
    model_name = "RandomForestMulticlassifier"
    model_alias = "Backup"
    cutoff_year = 2020
    main(output_dir,
         outfile_name,
         report_name,
         model_name,
         model_alias=model_alias,
         cutoff_year=cutoff_year)


