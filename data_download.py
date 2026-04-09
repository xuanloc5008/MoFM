import argparse
import opendatasets as od
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset from Kaggle using opendatasets")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Kaggle dataset URL"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Directory to save dataset"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading dataset: {args.dataset}")
    print(f"Saving to: {args.output_dir}")

    od.download(args.dataset, data_dir=args.output_dir)


if __name__ == "__main__":
    main()

    