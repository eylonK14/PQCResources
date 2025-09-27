import os
import numpy as np
from scapy.all import rdpcap
import argparse

# Function to compute stats for a single PCAP file
def compute_statistics(pcap_file):
    packets = rdpcap(pcap_file)
    num_packets = len(packets)
    packet_sizes = [len(pkt) for pkt in packets]
    iat = [float(packets[i].time) - float(packets[i-1].time)
           for i in range(1, num_packets)]

    def stats(values):
        return {
            'mean': np.mean(values) if values else 0,
            'std_dev': np.std(values) if values else 0
        }

    return stats([num_packets]), stats(packet_sizes), stats(iat)

# Helper to aggregate stats lists
def aggregate(stats_list):
    means = np.mean([s['mean'] for s in stats_list]) if stats_list else 0
    stds = np.mean([s['std_dev'] for s in stats_list]) if stats_list else 0
    return means, stds

# Main entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Aggregate PCAP stats (packet count, size, IAT) across subdirectories'
    )
    parser.add_argument(
        'root_directory', help='Path to root directory containing PCAP subdirectories'
    )
    args = parser.parse_args()
    root_directory = args.root_directory

    # Prepare accumulators for overall stats
    overall = {
        'num_packets': [],
        'packet_sizes': [],
        'iat': []
    }

    # Iterate through each subdirectory
    for subdir in os.listdir(root_directory):
        dir_path = os.path.join(root_directory, subdir)
        if not os.path.isdir(dir_path):
            continue

        print(f"Processing {subdir}...")
        per_dir = {
            'num_packets': [],
            'packet_sizes': [],
            'iat': []
        }

        for fname in os.listdir(dir_path):
            if not fname.endswith('.pcap'):
                continue
            path = os.path.join(dir_path, fname)
            np_s, ps_s, iat_s = compute_statistics(path)
            per_dir['num_packets'].append(np_s)
            per_dir['packet_sizes'].append(ps_s)
            per_dir['iat'].append(iat_s)
            overall['num_packets'].append(np_s)
            overall['packet_sizes'].append(ps_s)
            overall['iat'].append(iat_s)

        # Aggregate per-directory stats
        np_mean, np_std = aggregate(per_dir['num_packets'])
        ps_mean, ps_std = aggregate(per_dir['packet_sizes'])
        iat_mean, iat_std = aggregate(per_dir['iat'])

        print(f"\nStats for {subdir}:")
        print(f"  Packets — Mean: {np_mean:.2f}, Std: {np_std:.2f}")
        print(f"  Packet Size — Mean: {ps_mean:.2f}, Std: {ps_std:.2f}")
        print(f"  IAT — Mean: {iat_mean:.2f}, Std: {iat_std:.2f}\n")
        print('-' * 50)

    # Aggregate overall across all subdirectories
    print("Overall Stats Across All Subdirectories:")
    np_mean, np_std = aggregate(overall['num_packets'])
    ps_mean, ps_std = aggregate(overall['packet_sizes'])
    iat_mean, iat_std = aggregate(overall['iat'])

    print(f"Overall Packets — Mean: {np_mean:.2f}, Std: {np_std:.2f}")
    print(f"Overall Packet Size — Mean: {ps_mean:.2f}, Std: {ps_std:.2f}")
    print(f"Overall IAT — Mean: {iat_mean:.2f}, Std: {iat_std:.2f}")
