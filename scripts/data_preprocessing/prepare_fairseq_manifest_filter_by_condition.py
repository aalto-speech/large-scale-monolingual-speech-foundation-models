import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Filter TSV based on conditions")
    parser.add_argument("--input", required=True, help="Input TSV file")
    parser.add_argument("--output", required=True, help="Output TSV file")
    parser.add_argument(
        "--condition", 
        required=True,
        help=(
            "Filtering condition in pandas query syntax, e.g.: "
            "\"non_blank_log_prob_mean >= -0.1\""
        )
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep="\t")

    nsamples_before_filtering = len(df)
    dur_before_filtering = df['nframes'].sum() / 16000 / 3600

    try:
        df = df.query(args.condition)
    except Exception as e:
        raise ValueError(f"Failed to parse condition: {args.condition}\nError: {e}")

    nsamples_after_filtering = len(df)
    dur_after_filtering = df['nframes'].sum() / 16000 / 3600

    print(f"Original rows: {nsamples_before_filtering} ({dur_before_filtering}h), Filtered rows: {nsamples_after_filtering} ({dur_after_filtering}h)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df = df.sample(frac=1).reset_index(drop=True)

    with open(args.output, 'w') as f:
        f.write("/tar_vad_only/all\n")
        df[['file', 'nframes']].to_csv(f, sep="\t", index=False, header=False)

if __name__ == "__main__":
    main()
