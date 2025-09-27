# Updated full script: extract_trdl_dl.py

import pandas as pd
import argparse
import os
import ast

def load_trdl_and_label(file_path):
    """
    Load CSV and ensure TRDL is a list-of-lists of ints.
    """
    df = pd.read_csv(file_path)
    
    if 'TRDL' in df.columns:
        # Parse the TRDL column (string of list) into actual Python list
        df['TRDL'] = df['TRDL'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    else:
        # Build TRDL from all columns except first, 'label', and 'DL'
        data_cols = df.columns[1:]
        data_cols = [c for c in data_cols if c not in ['label', 'DL']]
        df['TRDL'] = df[data_cols].values.tolist()

    return df

def extract_dl_column(df):
    """
    From each packet [a, b, c] or string "[a, b, c]", keep only [a, b] → build DL.
    """
    def to_dl(pkt):
        # Parse string packet to list
        if isinstance(pkt, str):
            pkt_list = ast.literal_eval(pkt)
        else:
            pkt_list = pkt
        # Slice to first two elements
        if isinstance(pkt_list, list) and len(pkt_list) >= 2:
            return pkt_list[:2]
        return pkt_list  # fallback

    df['DL'] = df['TRDL'].apply(lambda packets: [to_dl(pkt) for pkt in packets])
    return df

def process_and_save(input_path, output_path):
    """
    Load CSV, extract TRDL & DL, keep label, and save to new CSV.
    """
    df = load_trdl_and_label(input_path)
    df = extract_dl_column(df)

    # Select columns to keep
    cols = ['TRDL', 'DL']
    if 'label' in df.columns:
        cols.append('label')

    result_df = df[cols]
    result_df.to_csv(output_path, index=False)
    return result_df

def main():
    parser = argparse.ArgumentParser(
        description="Extract TRDL & DL columns (first two values) and include label."
    )
    parser.add_argument("input",  help="Path to input CSV file")
    parser.add_argument("output", help="Path to output CSV file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return

    result_df = process_and_save(args.input, args.output)
    print(f"✅ Done! Saved to: {args.output}")
    print(result_df.head())

if __name__ == "__main__":
    main()

# Save as extract_trdl_dl.py and run:
# python extract_trdl_dl.py input.csv output.csv
