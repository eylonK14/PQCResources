#!/usr/bin/env python3
"""
compute_tdl_flow_stats.py
Compute mean, variance, and standard deviation of packet sizes and inter-arrival times (IAT)
for all datagrams in a CSV file produced by the NFPlugin TDL class.
Statistics are calculated separately for labels ending in 0 and labels ending in 1.

CSV format:
- First row: header
- 'label' column: flow labels (used for grouping)
- Remaining columns: datagrams represented as "[direction, size, time]" where:
    direction: packet direction (ignored)
    size: packet length in bytes
    time: timestamp relative to flow start (milliseconds)

Usage:
    python compute_tdl_flow_stats.py input.csv
"""
import csv
import re
import statistics
import argparse


def parse_datagram(cell):
    """
    Parse a datagram cell string "[direction, size, time]" and return (size_bytes, time_ms).
    """
    cell = cell.strip()
    if not cell:
        return None
    # match three comma-separated values inside brackets
    match = re.match(r"\[\s*([^,]+),\s*([^,]+),\s*([^\]]+)\s*\]", cell)
    if match:
        # group(1) = direction (ignored)
        size_bytes = float(match.group(2))
        time_ms = float(match.group(3))
        return size_bytes, time_ms
    return None


def compute_stats(datagrams):
    """
    Given a list of (size_bytes, time_ms) tuples, compute stats for sizes and IAT (in ms).
    Returns a dict with:
      - mean_size (bytes), var_size (bytes^2), std_size (bytes)
      - mean_iat (ms),   var_iat (ms^2),   std_iat (ms)
      - min_size, max_size, range_size (bytes)
      - min_iat, max_iat, range_iat (ms)
    """
    if not datagrams:
        return {
            'mean_size': 0,
            'var_size': 0,
            'std_size': 0,
            'min_size': 0,
            'max_size': 0,
            'range_size': 0,
            'mean_iat': 0,
            'var_iat': 0,
            'std_iat': 0,
            'min_iat': 0,
            'max_iat': 0,
            'range_iat': 0,
        }
    
    sizes = [d[0] for d in datagrams]
    # Ensure chronological order by time
    times = sorted(d[1] for d in datagrams)
    
    # Compute inter-arrival times in milliseconds
    if len(times) > 1:
        iats = [t2 - t1 for t1, t2 in zip(times, times[1:])]
    else:
        iats = [0.0]
    
    # Safe defaults
    if not sizes:
        sizes = [0]
    if not iats:
        iats = [0.0]
    
    # Calculate min, max, and range for sizes
    min_size = min(sizes)
    max_size = max(sizes)
    range_size = max_size - min_size
    
    # Calculate min, max, and range for IATs
    min_iat = min(iats)
    max_iat = max(iats)
    range_iat = max_iat - min_iat
    
    return {
        'mean_size': statistics.mean(sizes),
        'var_size': statistics.pvariance(sizes),
        'std_size': statistics.pstdev(sizes),
        'min_size': min_size,
        'max_size': max_size,
        'range_size': range_size,
        'mean_iat': statistics.mean(iats),
        'var_iat': statistics.pvariance(iats),
        'std_iat': statistics.pstdev(iats),
        'min_iat': min_iat,
        'max_iat': max_iat,
        'range_iat': range_iat,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute packet size & IAT stats (IAT in ms) from TDL CSV output, grouped by label suffix."
    )
    parser.add_argument('input_csv', help='Path to input CSV file')
    args = parser.parse_args()
    
    # Separate collections for labels ending in 0 and 1
    datagrams_ending_0 = []
    datagrams_ending_1 = []
    
    max_size_0 = 0
    max_size_1 = 0
    max_size_location_0 = (0, 0)  # (row, column)
    max_size_location_1 = (0, 0)  # (row, column)
    
    with open(args.input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # read header
        
        # Find the label column index
        try:
            label_col_idx = header.index('label')
        except ValueError:
            raise ValueError("'label' column not found in CSV header")
        
        for row_idx, row in enumerate(reader, start=2):  # start=2 because row 1 is header
            if len(row) <= label_col_idx:
                continue  # Skip rows that don't have enough columns
            
            # Get the label for this row
            label = row[label_col_idx].strip()
            
            # Determine if this label ends in 0 or 1
            label_ends_in_0 = label.endswith('0')
            label_ends_in_1 = label.endswith('1')
            
            # Skip if label doesn't end in 0 or 1
            if not (label_ends_in_0 or label_ends_in_1):
                continue
            
            # Process all datagram columns (skip label and any other metadata columns)
            for col_idx, cell in enumerate(row):
                if col_idx == label_col_idx:  # Skip the label column
                    continue
                    
                parsed = parse_datagram(cell)
                if parsed:
                    size_bytes, time_ms = parsed
                    
                    if label_ends_in_0:
                        datagrams_ending_0.append(parsed)
                        # Track maximum packet size and its location
                        if size_bytes > max_size_0:
                            max_size_0 = size_bytes
                            max_size_location_0 = (row_idx, col_idx + 1)  # +1 for 1-based indexing
                    
                    elif label_ends_in_1:
                        datagrams_ending_1.append(parsed)
                        # Track maximum packet size and its location
                        if size_bytes > max_size_1:
                            max_size_1 = size_bytes
                            max_size_location_1 = (row_idx, col_idx + 1)  # +1 for 1-based indexing
    
    # Compute stats for both groups
    stats_0 = compute_stats(datagrams_ending_0)
    stats_1 = compute_stats(datagrams_ending_1)
    
    # Print results
    print("=" * 80)
    print("STATISTICS FOR LABELS ENDING IN 0 (210, 220, etc.)")
    print("=" * 80)
    print(f"Number of datagrams: {len(datagrams_ending_0)}")
    print("\nPacket Size Statistics (bytes):")
    print(f"  Mean: {stats_0['mean_size']:.2f}")
    print(f"  Variance: {stats_0['var_size']:.2f}")
    print(f"  Std Dev: {stats_0['std_size']:.2f}")
    print(f"  Min: {stats_0['min_size']:.2f}")
    print(f"  Max: {stats_0['max_size']:.2f}")
    print(f"  Range: {stats_0['range_size']:.2f}")
    
    print("\nInter-Arrival Time Statistics (ms):")
    print(f"  Mean: {stats_0['mean_iat']:.2f}")
    print(f"  Variance: {stats_0['var_iat']:.2f}")
    print(f"  Std Dev: {stats_0['std_iat']:.2f}")
    print(f"  Min: {stats_0['min_iat']:.2f}")
    print(f"  Max: {stats_0['max_iat']:.2f}")
    print(f"  Range: {stats_0['range_iat']:.2f}")
    
    print(f"\nBiggest packet size: {max_size_0} bytes")
    print(f"Location: Row {max_size_location_0[0]}, Column {max_size_location_0[1]}")
    
    print("\n" + "=" * 80)
    print("STATISTICS FOR LABELS ENDING IN 1 (211, 221, etc.)")
    print("=" * 80)
    print(f"Number of datagrams: {len(datagrams_ending_1)}")
    print("\nPacket Size Statistics (bytes):")
    print(f"  Mean: {stats_1['mean_size']:.2f}")
    print(f"  Variance: {stats_1['var_size']:.2f}")
    print(f"  Std Dev: {stats_1['std_size']:.2f}")
    print(f"  Min: {stats_1['min_size']:.2f}")
    print(f"  Max: {stats_1['max_size']:.2f}")
    print(f"  Range: {stats_1['range_size']:.2f}")
    
    print("\nInter-Arrival Time Statistics (ms):")
    print(f"  Mean: {stats_1['mean_iat']:.2f}")
    print(f"  Variance: {stats_1['var_iat']:.2f}")
    print(f"  Std Dev: {stats_1['std_iat']:.2f}")
    print(f"  Min: {stats_1['min_iat']:.2f}")
    print(f"  Max: {stats_1['max_iat']:.2f}")
    print(f"  Range: {stats_1['range_iat']:.2f}")
    
    print(f"\nBiggest packet size: {max_size_1} bytes")
    print(f"Location: Row {max_size_location_1[0]}, Column {max_size_location_1[1]}")
    
    print("\n" + "=" * 80)
    print("COMBINED STATISTICS")
    print("=" * 80)
    all_datagrams = datagrams_ending_0 + datagrams_ending_1
    combined_stats = compute_stats(all_datagrams)
    max_size_overall = max(max_size_0, max_size_1)
    print(f"Total number of datagrams: {len(all_datagrams)}")
    print("\nPacket Size Statistics (bytes):")
    print(f"  Mean: {combined_stats['mean_size']:.2f}")
    print(f"  Variance: {combined_stats['var_size']:.2f}")
    print(f"  Std Dev: {combined_stats['std_size']:.2f}")
    print(f"  Min: {combined_stats['min_size']:.2f}")
    print(f"  Max: {combined_stats['max_size']:.2f}")
    print(f"  Range: {combined_stats['range_size']:.2f}")
    
    print("\nInter-Arrival Time Statistics (ms):")
    print(f"  Mean: {combined_stats['mean_iat']:.2f}")
    print(f"  Variance: {combined_stats['var_iat']:.2f}")
    print(f"  Std Dev: {combined_stats['std_iat']:.2f}")
    print(f"  Min: {combined_stats['min_iat']:.2f}")
    print(f"  Max: {combined_stats['max_iat']:.2f}")
    print(f"  Range: {combined_stats['range_iat']:.2f}")
    
    print(f"\nBiggest packet size overall: {max_size_overall} bytes")


if __name__ == "__main__":
    main()