"""
Split preprocessed features and target into train/test sets.

This script loads `cleaned_X.csv` and `y.csv` from the same folder as this
script, performs a train/test split, and writes the resulting files back to
the same folder as `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.

Usage example:
    python data_split.py --test-size 0.2 --random-state 42

Outputs:
    - X_train.csv, X_test.csv, y_train.csv, y_test.csv (in the same folder)

Notes:
    - `test_size` should be a float in (0,1) or an int number of samples.
    - The script resolves paths relative to its own location, so run it from
      any working directory.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main(input_dir: Path):
    """Load cleaned_X and y, split, and save outputs.

    Args:
        test_size: proportion (0-1) or int of test set size.
        random_state: RNG seed for reproducibility.
        input_dir: directory containing `cleaned_X.csv` and `y.csv`.
    """

    # Resolve input file paths relative to this script's folder
    X_path = input_dir / "cleaned_X.csv"
    y_path = input_dir / "y.csv"

    # Load datasets
    # We use pandas read_csv so both single-column and multi-column targets work.
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)

    # Quick shape check for visibility
    print("Original Shapes -> X: {}, y: {}".format(X.shape, y.shape))

    # Perform train/test split using the required fixed parameters.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Print split sizes to help confirm everything went as expected
    print("After Split -> X_train: {}, X_test: {}, y_train: {}, y_test: {}".format(
        X_train.shape, X_test.shape, y_train.shape, y_test.shape
    ))

    # Output files (written to same folder)
    X_train.to_csv(input_dir / "X_train.csv", index=False)
    X_test.to_csv(input_dir / "X_test.csv", index=False)
    y_train.to_csv(input_dir / "y_train.csv", index=False)
    y_test.to_csv(input_dir / "y_test.csv", index=False)

    print("\nTrain-test split completed and files written to: {}".format(input_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split cleaned_X.csv and y.csv into train/test sets"
    )
    # Note: test size and random state are fixed per project requirement.
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing cleaned_X.csv and y.csv (defaults to script folder).",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()

    if not (input_dir / "cleaned_X.csv").exists():
        raise FileNotFoundError(f"cleaned_X.csv not found in {input_dir}")
    if not (input_dir / "y.csv").exists():
        raise FileNotFoundError(f"y.csv not found in {input_dir}")

    main(input_dir=input_dir)