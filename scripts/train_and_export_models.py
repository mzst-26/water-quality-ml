import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def calculate_interquartile_capping_bounds(column_values: pd.Series) -> tuple[float, float]:
    # Find the clipping limits using the 1.5 x IQR rule.
    first_quartile = column_values.quantile(0.25)
    third_quartile = column_values.quantile(0.75)
    interquartile_range = third_quartile - first_quartile

    lower_bound = first_quartile - 1.5 * interquartile_range
    upper_bound = third_quartile + 1.5 * interquartile_range
    return float(lower_bound), float(upper_bound)


def apply_interquartile_capping(
    input_dataframe: pd.DataFrame,
    column_names_to_cap: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    # Clip outliers in each selected column and store the limits we used.
    capped_dataframe = input_dataframe.copy()
    capping_bounds_by_column: dict[str, dict[str, float]] = {}

    for column_name in column_names_to_cap:
        lower_bound, upper_bound = calculate_interquartile_capping_bounds(
            capped_dataframe[column_name]
        )
        capped_dataframe[column_name] = capped_dataframe[column_name].clip(
            lower=lower_bound,
            upper=upper_bound,
        )
        capping_bounds_by_column[column_name] = {
            "lower": lower_bound,
            "upper": upper_bound,
        }

    return capped_dataframe, capping_bounds_by_column


def compute_regression_metrics(
    actual_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[float, float]:
    # Calculate RMSE and R2 for a set of predictions.
    root_mean_squared_error = float(
        np.sqrt(mean_squared_error(actual_values, predicted_values))
    )
    coefficient_of_determination = float(r2_score(actual_values, predicted_values))
    return root_mean_squared_error, coefficient_of_determination


def main() -> None:
    # Train models, export artifacts, and save the summary plots.
    repository_root = Path(__file__).resolve().parents[1]

    dataset_file_path = repository_root / "data" / "e1_nutrients.csv"
    model_artifacts_directory = repository_root / "artifacts" / "models"
    metrics_output_directory = repository_root / "artifacts" / "results"
    analytics_output_directory = repository_root / "Analytics" / "3_cross_validator"

    # Make sure the output folders exist before saving files.
    model_artifacts_directory.mkdir(parents=True, exist_ok=True)
    metrics_output_directory.mkdir(parents=True, exist_ok=True)
    analytics_output_directory.mkdir(parents=True, exist_ok=True)

    # Load the raw dataset from the data folder.
    raw_dataset = pd.read_csv(dataset_file_path)

    # These columns get capped so outliers do not control the model too much.
    columns_for_outlier_capping = [
        "NITRITE",
        "AMMONIA",
        "SILICATE",
        "PHOSPHATE",
        "NITRATE+NITRITE",
    ]
    capped_dataset, capping_bounds = apply_interquartile_capping(
        raw_dataset,
        columns_for_outlier_capping,
    )

    # Split the dataset into features and target.
    feature_dataframe = capped_dataset.drop(columns=["NITRITE"])
    target_series = capped_dataset["NITRITE"]

    # Use a fixed seed so the same split happens every time.
    (
        training_features,
        testing_features,
        training_target,
        testing_target,
    ) = train_test_split(
        feature_dataframe,
        target_series,
        test_size=0.2,
        random_state=42,
    )

    # Scale the features so the models learn on the same range.
    feature_scaler = MinMaxScaler()
    training_features_scaled = feature_scaler.fit_transform(training_features)
    testing_features_scaled = feature_scaler.transform(testing_features)

    # Use a log transform for the tree model and the neural network.
    training_target_log = np.log1p(training_target)

    linear_regression_model = LinearRegression()

    # Random Forest settings chosen from the notebook tuning step.
    random_forest_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    # Neural Network layout chosen from the notebook tuning step.
    neural_model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )

    linear_regression_model.fit(training_features_scaled, training_target)
    random_forest_model.fit(training_features_scaled, training_target_log)
    neural_model.fit(training_features_scaled, training_target_log)

    # Make predictions on the test set.
    linear_predictions = linear_regression_model.predict(testing_features_scaled)

    # Convert the log predictions back to the original scale.
    random_forest_predictions = np.expm1(
        random_forest_model.predict(testing_features_scaled)
    )
    neural_network_predictions = np.expm1(neural_model.predict(testing_features_scaled))

    # Score each model with the same test data.
    linear_rmse, linear_r2 = compute_regression_metrics(
        testing_target,
        linear_predictions,
    )
    random_forest_rmse, random_forest_r2 = compute_regression_metrics(
        testing_target,
        random_forest_predictions,
    )
    neural_network_rmse, neural_network_r2 = compute_regression_metrics(
        testing_target,
        neural_network_predictions,
    )

    # Put the scores into one table so they are easy to save and print.
    model_metrics_dataframe = pd.DataFrame(
        [
            {
                "model": "Linear Regression",
                "rmse": linear_rmse,
                "r2": linear_r2,
                "target_space": "original",
            },
            {
                "model": "Random Forest",
                "rmse": random_forest_rmse,
                "r2": random_forest_r2,
                "target_space": "log1p-trained, inverse on predict",
            },
            {
                "model": "Neural Network",
                "rmse": neural_network_rmse,
                "r2": neural_network_r2,
                "target_space": "log1p-trained, inverse on predict",
            },
        ]
    ).sort_values("rmse", ascending=True)

    metrics_csv_path = metrics_output_directory / "latest_model_results.csv"
    metrics_json_path = metrics_output_directory / "latest_model_results.json"
    model_metrics_dataframe.to_csv(metrics_csv_path, index=False)
    model_metrics_dataframe.to_json(metrics_json_path, orient="records", indent=2)

    # Save the fitted models and the scaler for later prediction.
    with open(model_artifacts_directory / "linear_regression.pkl", "wb") as output_file:
        pickle.dump(linear_regression_model, output_file)
    with open(model_artifacts_directory / "random_forest_tuned.pkl", "wb") as output_file:
        pickle.dump(random_forest_model, output_file)
    with open(model_artifacts_directory / "neural_network_tuned.pkl", "wb") as output_file:
        pickle.dump(neural_model, output_file)
    with open(model_artifacts_directory / "minmax_scaler.pkl", "wb") as output_file:
        pickle.dump(feature_scaler, output_file)

    # Save the feature order and capping bounds used during training.
    preprocessing_metadata = {
        "feature_order": feature_dataframe.columns.tolist(),
        "target": "NITRITE",
        "cap_bounds": capping_bounds,
        "log_target_models": ["random_forest_tuned", "neural_network_tuned"],
    }
    with open(
        model_artifacts_directory / "preprocessing.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(preprocessing_metadata, output_file, indent=2)

    # Save a simple chart for the README.
    sorted_metrics_dataframe = model_metrics_dataframe.sort_values("rmse", ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        sorted_metrics_dataframe["model"],
        sorted_metrics_dataframe["rmse"],
        color=["#2a9d8f", "#457b9d", "#e76f51"],
    )
    axes[0].set_title("Model RMSE (lower is better)")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(
        sorted_metrics_dataframe["model"],
        sorted_metrics_dataframe["r2"],
        color=["#2a9d8f", "#457b9d", "#e76f51"],
    )
    axes[1].set_title("Model R2 (higher is better)")
    axes[1].set_ylabel("R2")
    axes[1].tick_params(axis="x", rotation=15)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Latest Notebook-Style Model Comparison")
    plt.tight_layout()
    plt.savefig(
        analytics_output_directory / "model_comparison_chart.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Print the final results so the user sees them in the terminal too.
    print("Saved model artifacts to artifacts/models")
    print(f"Saved metrics to {metrics_csv_path}")
    print("Saved model comparison chart to Analytics/3_cross_validator/model_comparison_chart.png")
    print("\nLatest metrics:")
    print(model_metrics_dataframe.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
