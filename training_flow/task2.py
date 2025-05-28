from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
import pandas as pd
from pathlib import Path
import mlflow
import json

def main(infile_dir,
         infile_name,
         model_name,
         cutoff_year=2020):
    input_path = infile_dir / infile_name
    dataset = pd.read_csv(input_path, index_col=False)
    dataset.release_date = pd.to_datetime(dataset.release_date)

    mask = dataset.release_date.dt.year < cutoff_year
    dataset = dataset[mask]

    input_cols = ['price', 'positive_reviews', 'negative_reviews', 'metacritic_score', 'peak_ccu', 'recommendations', 'required_age', 'on_linux', 'on_mac', 'on_windows']

    X, y = dataset[input_cols], dataset['estimated_owners']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    if len(X_train) < 1000:
        raise ValueError('Training set is too small to produce a good model')



    hp_path = Path("./flows_git/training_flow/model_hyperparameters.txt")
    try:
        with open(hp_path, "r", encoding="utf-8") as f:
            params = json.loads(f.read())
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find (or read) hyperparameter file at {hp_path} : {e}")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    metrics = {'accuracy': accuracy, 'balanced_accuracy':balanced_accuracy, "f1": f1}

    mlflow.set_experiment("MLOpsEx3")

    with mlflow.start_run(run_name="Model Training"):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score_macro", f1)
        mlflow.log_metric("balanced_accuracy", balanced_accuracy)

        mlflow.set_tag("Training Info", "Basic RF model for steam games dataset")

        signature = mlflow.models.infer_signature(X.loc[0].to_dict(), y.loc[0])

        mlflow.log_params(params)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            signature=signature,
            artifact_path="random_forest_model",
            input_example=X_train,
            registered_model_name=model_name
        )

        return model_info, metrics

if __name__=="__main__":
    input_dir = Path("./data")
    infile_name = "steam_games_dataset.csv"
    main(input_dir, infile_name, "RandomForestMulticlassifier", 2020)
