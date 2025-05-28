import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

from evidently.test_suite import TestSuite
from evidently.tests import TestAccuracyScore, TestF1Score, TestRecallByClass
from evidently.test_preset import MulticlassClassificationTestPreset

def main(
        infile_dir,
         infile_name,
         model_name,
         model_version=None,
         model_alias=None,
         cutoff_year=2020,
         acc_threshold=0,
         f1_threshold=0
    ):

    input_path = infile_dir / infile_name
    dataset = pd.read_csv(input_path, index_col=False)
    dataset.release_date = pd.to_datetime(dataset.release_date)

    mask = dataset.release_date.dt.year >= cutoff_year
    dataset = dataset[mask]

    model_version = "latest" if model_version is None else model_version


    # Load the model from the Model Registry
    if model_alias is not None:
        model_uri = f"models:/{model_name}@{model_alias}"
    else:
        model_uri = f"models:/{model_name}/{model_version}"

    print(model_uri)

    model = mlflow.sklearn.load_model(model_uri)

    input_cols = ['price', 'positive_reviews', 'negative_reviews', 'metacritic_score', 'peak_ccu', 'recommendations', 'required_age', 'on_linux', 'on_mac', 'on_windows']
    X, y = dataset[input_cols], dataset['estimated_owners']
    y_pred = model.predict(X)

    # Create a DataFrame with actual and predicted values
    df = pd.DataFrame({
        'target': y,
        'prediction': y_pred
    })

    # Define test suite
    test_suite = TestSuite(
        tests=[
            TestAccuracyScore(gte=acc_threshold),
            TestF1Score(gte=f1_threshold),
            TestRecallByClass(label='0 - 20000')
        ]
    )

    # Run the test suite
    test_suite.run(reference_data=None, current_data=df)

    # Save the test results to HTML
    test_suite.save_html(str(infile_dir / "classifier_results.html"))

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    balanced_accuracy = balanced_accuracy_score(y, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    return accuracy, balanced_accuracy, f1

if __name__ == "__main__":
    output_dir = Path('./data')
    outfile_name = "steam_games_dataset.csv"
    report_name = "report.html"
    model_name = "RandomForestMulticlassifier"
    cutoff_year = 2020
    acc_threshold = 0.4
    f1_threshold = 0.4
    main(output_dir, outfile_name, model_name, cutoff_year=cutoff_year, acc_threshold=acc_threshold, f1_threshold=acc_threshold)
