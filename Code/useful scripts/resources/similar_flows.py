#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import ast

def parse_list_column(col):
    """
    Parse a Series of either real lists or their string repr into actual lists.
    """
    def _parse(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x
    return col.apply(_parse)

def compute_class_similarity_rates(df):
    """
    For each label, compute:
      - TRDL_similarity_pct = (N - unique_trdl) / N * 100
      - DL_similarity_pct   = (N - unique_dl)   / N * 100
    Returns a DataFrame with columns:
      label, TRDL_similarity_pct, DL_similarity_pct
    """
    rows = []
    for label, group in df.groupby('label', sort=False):
        N = len(group)
        # Convert each flow to a tuple-of-tuples for uniqueness counting
        trdl_keys = group['TRDL'].apply(lambda flow: tuple(tuple(pkt) for pkt in flow))
        dl_keys   = group['DL']  .apply(lambda flow: tuple(tuple(pkt) for pkt in flow))
        unique_trdl = trdl_keys.nunique()
        unique_dl   = dl_keys.  nunique()
        trdl_pct = (N - unique_trdl) / N * 100 if N else 0.0
        dl_pct   = (N - unique_dl)   / N * 100 if N else 0.0
        rows.append({
            'label': label,
            'TRDL_similarity_pct': round(trdl_pct, 2),
            'DL_similarity_pct':   round(dl_pct,   2)
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Compute percentage of identical TRDL and DL flows per class (label)."
    )
    parser.add_argument("input",  help="CSV file path with columns 'label','TRDL','DL'")
    parser.add_argument("output", help="Output CSV file path for class similarity rates")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    df = pd.read_csv(args.input)

    # Validate presence of required columns
    for col in ('label','TRDL','DL'):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in input.")

    # Parse TRDL and DL if they're strings
    df['TRDL'] = parse_list_column(df['TRDL'])
    df['DL']   = parse_list_column(df['DL'])

    # Compute rates
    summary = compute_class_similarity_rates(df)

    # Save result
    summary.to_csv(args.output, index=False)
    print(f"✅ Class similarity rates saved to: {args.output}")
    print(summary)

if __name__ == "__main__":
    main()
