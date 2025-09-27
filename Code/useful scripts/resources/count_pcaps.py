#!/usr/bin/env python3
import os
import argparse
from scapy.utils import RawPcapReader

# Define your packet‐count bins as (low_inclusive, high_exclusive)
BINS = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70)]
BIN_LABELS = [f"{low}-{high}" for low, high in BINS]

def count_pcaps_in_dir(dirpath):
    """
    For each .pcap/.pcapng in dirpath, count its packets and
    bucket it into one of the packet‐count bins.
    Returns a dict: { "20-30": count_of_pcaps_in_that_range, … }.
    """
    counts = {label: 0 for label in BIN_LABELS}

    try:
        for fn in os.listdir(dirpath):
            if not fn.lower().endswith(('.pcap', '.pcapng')):
                continue
            full = os.path.join(dirpath, fn)
            if not os.path.isfile(full):
                continue

            # Count packets
            pkt_count = 0
            try:
                reader = RawPcapReader(full)
                for _pkt_data, _pkt_meta in reader:
                    pkt_count += 1
                reader.close()
            except Exception as e:
                print(f"  [!] Failed to read {full}: {e}")
                continue

            # Bucket by packet count
            for (low, high), label in zip(BINS, BIN_LABELS):
                if low <= pkt_count < high:
                    counts[label] += 1
                    break

    except OSError as e:
        print(f"  [!] Could not access directory {dirpath}: {e}")

    return counts

def main():
    parser = argparse.ArgumentParser(
        description="For every directory under ROOT_DIR, count .pcap/.pcapng files by packet‐count bins."
    )
    parser.add_argument("root_dir",
                        help="Root directory to recurse through")
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Error: '{args.root_dir}' is not a directory.")
        return

    # Walk all directories under root_dir
    for dirpath, _dirnames, _filenames in os.walk(args.root_dir):
        counts = count_pcaps_in_dir(dirpath)
        total = sum(counts.values())
        if total == 0:
            continue

        # Print header with path relative to root
        rel = os.path.relpath(dirpath, args.root_dir)
        print(f"\nDirectory: {rel}")
        for label in BIN_LABELS:
            cnt = counts[label]
            print(f"  {label} packets: {cnt} file{'s' if cnt != 1 else ''}")

if __name__ == "__main__":
    main()
