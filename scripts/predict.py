import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


MODEL_ARTIFACT_FILENAMES = {
    "linear": "linear_regression.pkl",
    "rf": "random_forest_tuned.pkl",
    "nn": "neural_network_tuned.pkl",
}

PREDICTION_DISCLAIMER = (
    "Disclaimer: This prediction is model-generated for educational/research use and "
    "should not replace laboratory measurements or expert environmental judgment."
)


def load_model_and_preprocessing_bundle(model_directory: Path, model_key: str):
    # Load the chosen model, the scaler, and the saved preprocessing data.
    with open(model_directory / MODEL_ARTIFACT_FILENAMES[model_key], "rb") as model_file:
        selected_model = pickle.load(model_file)
    with open(model_directory / "minmax_scaler.pkl", "rb") as scaler_file:
        feature_scaler = pickle.load(scaler_file)
    with open(model_directory / "preprocessing.json", "r", encoding="utf-8") as metadata_file:
        preprocessing_metadata = json.load(metadata_file)
    return selected_model, feature_scaler, preprocessing_metadata


def parse_float_input(prompt_text: str) -> float:
    # Keep asking until the user enters a valid number.
    while True:
        raw_value = input(prompt_text).strip()
        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a valid number.")


def apply_saved_capping_bounds(
    feature_dataframe: pd.DataFrame,
    preprocessing_metadata: dict,
) -> pd.DataFrame:
    # Use the same outlier limits that were used during training.
    capped_feature_dataframe = feature_dataframe.copy()
    capping_bounds_by_column = preprocessing_metadata.get("cap_bounds", {})

    for feature_name in preprocessing_metadata["feature_order"]:
        if feature_name in capping_bounds_by_column:
            lower_bound = capping_bounds_by_column[feature_name]["lower"]
            upper_bound = capping_bounds_by_column[feature_name]["upper"]
            capped_feature_dataframe[feature_name] = capped_feature_dataframe[
                feature_name
            ].clip(lower=lower_bound, upper=upper_bound)

    return capped_feature_dataframe


def run_inference(
    model_key: str,
    input_features: pd.DataFrame,
    model_directory: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    # Prepare the input data and run the selected model.
    selected_model, feature_scaler, preprocessing_metadata = (
        load_model_and_preprocessing_bundle(model_directory, model_key)
    )

    # Check that all required columns are present.
    required_feature_order = preprocessing_metadata["feature_order"]
    missing_required_columns = [
        feature_name
        for feature_name in required_feature_order
        if feature_name not in input_features.columns
    ]
    if missing_required_columns:
        raise ValueError(f"Missing required columns: {missing_required_columns}")

    feature_dataframe = input_features[required_feature_order].copy()

    # Apply the saved clipping values and then scale the features.
    capped_feature_dataframe = apply_saved_capping_bounds(
        feature_dataframe,
        preprocessing_metadata,
    )
    scaled_feature_matrix = feature_scaler.transform(capped_feature_dataframe)

    predicted_values = selected_model.predict(scaled_feature_matrix)
    if model_key in {"rf", "nn"}:
        # RF and NN were trained on log1p(target), so convert back now.
        predicted_values = np.expm1(predicted_values)

    return predicted_values, capped_feature_dataframe


def choose_mode_interactively() -> str:
    # Ask if the user wants testing or a single prediction.
    print("Choose an option:")
    print("1) Test model (predicted vs actual)")
    print("2) Predict with manual input")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "test"
        if choice == "2":
            return "predict"
        print("Invalid choice. Please enter 1 or 2.")


def choose_model_interactively() -> str:
    # Ask which model should be used.
    print("\nChoose model:")
    print("1) linear")
    print("2) rf")
    print("3) nn")

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "1":
            return "linear"
        if choice == "2":
            return "rf"
        if choice == "3":
            return "nn"
        print("Invalid choice. Please enter 1, 2, or 3.")


def run_test_mode(repository_root: Path, model_directory: Path) -> None:
    # Test the model using a CSV that includes the real NITRITE values.
    model_key = choose_model_interactively()
    default_input_path = "data/e1_nutrients.csv"
    entered_path = input(
        f"\nCSV path for testing (default: {default_input_path}): "
    ).strip()
    input_csv_path = Path(entered_path or default_input_path)
    if not input_csv_path.is_absolute():
        input_csv_path = repository_root / input_csv_path

    input_dataframe = pd.read_csv(input_csv_path)
    if "NITRITE" not in input_dataframe.columns:
        raise ValueError(
            "Testing requires an input file with 'NITRITE' column for actual values."
        )

    # Predict the values and compare them with the real answers.
    predicted_values, _ = run_inference(model_key, input_dataframe, model_directory)
    actual_values = input_dataframe["NITRITE"].to_numpy()

    root_mean_squared_error = float(np.sqrt(mean_squared_error(actual_values, predicted_values)))
    coefficient_of_determination = float(r2_score(actual_values, predicted_values))

    test_results_dataframe = input_dataframe.copy()
    test_results_dataframe["actual_NITRITE"] = input_dataframe["NITRITE"]
    test_results_dataframe["predicted_NITRITE"] = predicted_values
    test_results_dataframe["absolute_error"] = (
        test_results_dataframe["actual_NITRITE"] - test_results_dataframe["predicted_NITRITE"]
    ).abs()

    # Save the full test results so they can be checked later.
    output_path = (
        repository_root
        / "artifacts"
        / "results"
        / f"test_results_{model_key}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_results_dataframe.to_csv(output_path, index=False)

    print("\nTest complete.")
    print(f"Model: {model_key}")
    print(f"RMSE: {root_mean_squared_error:.6f}")
    print(f"R2: {coefficient_of_determination:.6f}")
    print(f"Saved detailed results to: {output_path}")
    print("\nSample rows (actual vs predicted):")
    print(
        test_results_dataframe[
            ["actual_NITRITE", "predicted_NITRITE", "absolute_error"]
        ]
        .head(10)
        .to_string(index=False)
    )


def run_manual_prediction_mode(repository_root: Path, model_directory: Path) -> None:
    # Ask the user for one row of values and predict a single result.
    model_key = choose_model_interactively()

    # Read the saved feature order from the preprocessing file.
    _, _, preprocessing_metadata = load_model_and_preprocessing_bundle(
        model_directory,
        model_key,
    )
    required_feature_order = preprocessing_metadata["feature_order"]

    print("\nEnter feature values for prediction:")
    user_row: dict[str, float] = {}
    for feature_name in required_feature_order:
        user_row[feature_name] = parse_float_input(f"{feature_name}: ")

    input_dataframe = pd.DataFrame([user_row])
    predicted_values, _ = run_inference(model_key, input_dataframe, model_directory)
    predicted_nitrite = float(predicted_values[0])

    # Save the single prediction for reference.
    output_path = repository_root / "artifacts" / "results" / "manual_prediction.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe = input_dataframe.copy()
    output_dataframe["predicted_NITRITE"] = predicted_nitrite
    output_dataframe.to_csv(output_path, index=False)

    print("\nPrediction complete.")
    print(f"Model: {model_key}")
    print(f"Predicted NITRITE: {predicted_nitrite:.6f}")
    print(PREDICTION_DISCLAIMER)
    print(f"Saved prediction to: {output_path}")


def run_non_interactive_mode(
    repository_root: Path,
    model_directory: Path,
    model_key: str,
    input_path: str,
    output_path: str,
) -> None:
    # Keep the old command-line mode for batch runs and scripts.
    input_csv_path = Path(input_path)
    if not input_csv_path.is_absolute():
        input_csv_path = repository_root / input_csv_path

    output_csv_path = Path(output_path)
    if not output_csv_path.is_absolute():
        output_csv_path = repository_root / output_csv_path
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Run predictions on every row in the input file.
    input_dataframe = pd.read_csv(input_csv_path)
    predicted_values, _ = run_inference(model_key, input_dataframe, model_directory)

    prediction_output_dataframe = input_dataframe.copy()
    prediction_output_dataframe["predicted_NITRITE"] = predicted_values
    prediction_output_dataframe.to_csv(output_csv_path, index=False)

    print(f"Predictions saved to {output_csv_path}")
    print(prediction_output_dataframe[["predicted_NITRITE"]].head().to_string(index=False))


def main() -> None:
    # If the user passes arguments, use the old command style.
    # If not, show the simple menu.
    parser = argparse.ArgumentParser(description="Run inference for NITRITE prediction.")
    parser.add_argument(
        "--model",
        choices=["linear", "rf", "nn"],
        default=None,
        help="Model to use: linear, rf, or nn (default: rf)",
    )
    parser.add_argument(
        "--input",
        required=False,
        help="Path to CSV containing feature columns only.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/results/predictions.csv",
        help="Where to write predictions CSV.",
    )
    command_arguments = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    model_directory = repository_root / "artifacts" / "models"

    if command_arguments.model and command_arguments.input:
        # Use command mode when model and input are already provided.
        run_non_interactive_mode(
            repository_root=repository_root,
            model_directory=model_directory,
            model_key=command_arguments.model,
            input_path=command_arguments.input,
            output_path=command_arguments.output,
        )
        return

    # Default behavior is the interactive menu.
    selected_mode = choose_mode_interactively()
    if selected_mode == "test":
        run_test_mode(repository_root, model_directory)
    else:
        run_manual_prediction_mode(repository_root, model_directory)


if __name__ == "__main__":
    main()
