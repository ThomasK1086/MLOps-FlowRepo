from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

from evidently.future.datasets import Dataset, DataDefinition
from evidently.test_suite import TestSuite
from evidently.tests import *

from pathlib import Path



def main(
        output_dir: Path,
        outfile_name: Path,
        report_name: Path,
):
    REPO_ID = "FronkonGames/steam-games-dataset"
    FILENAME = "games.csv"

    dataset = pd.read_csv(
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", revision="7e8915c96cd1a237d0655b8309dd1e8062ac841f"),
    )

    dataset['Release date'] = pd.to_datetime(dataset['Release date'], format="%b %d, %Y", errors="coerce")

    # Filter to only keep relevant columns, rename to make it easier to adress
    useful_columns_rename = {
        "Name" : "name",
        "Release date" : "release_date",
        "Estimated owners": "estimated_owners",
        "Price": "price",
        "Positive" : "positive_reviews",
        "Negative" : "negative_reviews",
        "Metacritic score" : "metacritic_score",
        "Peak CCU": "peak_ccu",
        "Recommendations": "recommendations",
        "Required age": "required_age",
        "Windows": "on_windows",
        "Linux": "on_linux",
        "Mac": "on_mac"
    }
    dataset = dataset[[col for col in useful_columns_rename.keys()]]
    dataset.rename(columns=useful_columns_rename, inplace=True)


    definition = DataDefinition(
        text_columns=["name"],
        numerical_columns=["positive_reviews", "negative_reviews", "metacritic_score", "peak_ccu", "recommendations", "price"],
        categorical_columns=["on_windows", "on_linux", "on_mac", "estimated_owners"],
        datetime_columns=["release_date"]
    )

    # Split into reference and current dataset, with everything before 2020 becoming reference
    mask = dataset.release_date.dt.year < 2020
    dataset_current = dataset[~mask]
    dataset_reference = dataset[mask]


    tests = [
        TestColumnNumberOfMissingValues(column_name=col) for col in dataset_current.columns
    ]
    tests.extend([
        TestColumnQuantile(column_name="price", quantile=quantile) for quantile in np.arange(0.1, 1, 0.2)
    ])

    tests.extend([
        TestColumnDrift(column_name="price"),
        TestColumnDrift(column_name="estimated_owners")
    ])

    def run_data_drift_tests(reference_data, current_data, tests, column_mapping=None):
        data_drift_test_suite = TestSuite(tests=tests)
        data_drift_test_suite.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
        return data_drift_test_suite

    result = run_data_drift_tests(reference_data=dataset_reference, current_data=dataset_current, tests=tests)


    output_dir.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(output_dir / outfile_name, index=False)
    result.save_html(str(output_dir / report_name))

    return

if __name__=="__main__":
    output_dir = Path('./data')
    outfile_name = "steam_games_dataset.csv"
    report_name = "report.html"

    main(output_dir, outfile_name, report_name)
