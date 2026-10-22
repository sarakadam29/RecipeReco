import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.recipes import (  # noqa: E402
    CLEAN_RECIPE_COLUMNS,
    OPTIONAL_RECIPE_COLUMNS,
    RECIPE_COLUMNS,
    clean_recipe_frame,
)

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "archive (2)" / "RAW_recipes.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "recipes_clean.csv"
CHUNK_SIZE = 25_000


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Recipe dataset not found: {DATASET_PATH}")

    source_columns = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()
    missing_columns = [column for column in RECIPE_COLUMNS if column not in source_columns]
    if missing_columns:
        raise ValueError(f"Recipe dataset is missing required columns: {missing_columns}")
    selected_columns = [
        *RECIPE_COLUMNS,
        *(column for column in OPTIONAL_RECIPE_COLUMNS if column in source_columns),
    ]
    serving_columns = [
        column
        for column in source_columns
        if "serving" in column.casefold() or "yield" in column.casefold()
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    missing_values = pd.Series(0, index=selected_columns, dtype="int64")
    data_types = None
    cleaned_data_types = None
    sample_recipes = []
    row_count = 0
    invalid_nutrition_count = 0
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            dir=OUTPUT_PATH.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        for chunk in pd.read_csv(
            DATASET_PATH,
            usecols=selected_columns,
            chunksize=CHUNK_SIZE,
        ):
            chunk = chunk.loc[:, selected_columns]
            row_count += len(chunk)
            missing_values += chunk.isna().sum()
            data_types = chunk.dtypes.astype(str)

            cleaned = clean_recipe_frame(chunk)
            cleaned_data_types = cleaned.dtypes.astype(str)
            invalid_nutrition_count += int(cleaned["calories"].isna().sum())
            if len(sample_recipes) < 3:
                sample_recipes.extend(cleaned.head(3 - len(sample_recipes)).to_dict("records"))
            for column in ("ingredients", "steps", *OPTIONAL_RECIPE_COLUMNS):
                cleaned[column] = cleaned[column].map(
                    lambda items: json.dumps(items, ensure_ascii=False)
                )
            cleaned.to_csv(
                temporary_path,
                mode="a",
                index=False,
                header=row_count == len(chunk),
                encoding="utf-8",
            )

        if row_count == 0:
            pd.DataFrame(columns=RECIPE_COLUMNS).to_csv(temporary_path, index=False)

        os.replace(temporary_path, OUTPUT_PATH)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"Dataset dimensions: {row_count} rows x {len(selected_columns)} selected source columns")
    print(f"Cleaned columns ({len(CLEAN_RECIPE_COLUMNS)}):", CLEAN_RECIPE_COLUMNS)
    print("Serving/yield columns:", serving_columns or "none")
    print(f"Missing or invalid nutrition arrays: {invalid_nutrition_count}")
    print("\nMissing values:")
    print(missing_values.to_string())
    print("\nData types:")
    if data_types is None:
        print("No data rows.")
    else:
        print(data_types.to_string())
    print("\nCleaned data types:")
    if cleaned_data_types is None:
        print("No data rows.")
    else:
        print(cleaned_data_types.to_string())
    print("\nSample recipes:")
    if sample_recipes:
        print(pd.DataFrame(sample_recipes).to_string(index=False))
    else:
        print("No data rows.")
    print(f"\nCleaned recipes saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
