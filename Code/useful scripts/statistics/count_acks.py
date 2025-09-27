#!/usr/bin/env python3

import os
import statistics
from scapy.all import rdpcap, TCP
from pathlib import Path

def count_ack_packets(pcap_file):
    """
    Count the number of ACK packets in a single PCAP file.
    
    Args:
        pcap_file (str): Path to the PCAP file
        
    Returns:
        int: Number of ACK packets found
    """
    try:
        packets = rdpcap(pcap_file)
        ack_count = 0
        
        for packet in packets:
            # Check if packet has TCP layer and ACK flag is set
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                # Check if ACK flag is set (flag value 16 or 0x10)
                if (tcp_layer.flags & 0x10) and len(tcp_layer.payload) == 0:  # ACK flag
                    ack_count += 1
                    
        return ack_count
    
    except Exception as e:
        print(f"Error processing {pcap_file}: {e}")
        return 0

def analyze_pcap_directory(root_directory):
    """
    Analyze all PCAP files in a directory and calculate ACK packet statistics.
    
    Args:
        root_directory (str): Path to the root directory containing PCAP files
        
    Returns:
        dict: Dictionary containing mean, standard deviation, and variance
    """
    # Find all PCAP files in the directory (including subdirectories)
    pcap_extensions = ['.pcap', '.pcapng', '.cap']
    pcap_files = []
    
    root_path = Path(root_directory)
    
    for ext in pcap_extensions:
        pcap_files.extend(root_path.rglob(f'*{ext}'))
    
    if not pcap_files:
        print(f"No PCAP files found in {root_directory}")
        return None
    
    print(f"Found {len(pcap_files)} PCAP files")
    
    # Count ACK packets in each file
    ack_counts = []
    
    for pcap_file in pcap_files:
        ack_count = count_ack_packets(str(pcap_file))
        ack_counts.append(ack_count)
    
    # Calculate statistics
    if len(ack_counts) < 2:
        print("Warning: Need at least 2 files for standard deviation calculation")
        if len(ack_counts) == 1:
            return {
                'mean': ack_counts[0],
                'std_dev': 0,
                'variance': 0,
                'total_files': 1,
                'total_acks': ack_counts[0]
            }
    
    mean_acks = statistics.mean(ack_counts)
    std_dev_acks = statistics.stdev(ack_counts) if len(ack_counts) > 1 else 0
    var_acks = statistics.variance(ack_counts) if len(ack_counts) > 1 else 0
    
    return {
        'mean': mean_acks,
        'std_dev': std_dev_acks,
        'variance': var_acks,
        'total_files': len(ack_counts),
        'total_acks': sum(ack_counts),
        'ack_counts': ack_counts
    }

def main():
    # Set your root directory path here
    root_directory = input("Enter the root directory path containing PCAP files: ").strip()
    
    if not os.path.exists(root_directory):
        print(f"Directory {root_directory} does not exist!")
        return
    
    print(f"\nAnalyzing PCAP files in: {root_directory}")
    print("-" * 50)
    
    results = analyze_pcap_directory(root_directory)
    
    if results:
        print("\n" + "="*50)
        print("ACK PACKET STATISTICS")
        print("="*50)
        print(f"Total PCAP files processed: {results['total_files']}")
        print(f"Total ACK packets found: {results['total_acks']}")
        print(f"Mean ACK packets per file: {results['mean']:.2f}")
        print(f"Standard Deviation: {results['std_dev']:.2f}")
        print(f"Variance: {results['variance']:.2f}")
        
        # Optional: Show individual file counts
        print(f"\nIndividual file ACK counts: {results['ack_counts']}")

if __name__ == "__main__":
    main()