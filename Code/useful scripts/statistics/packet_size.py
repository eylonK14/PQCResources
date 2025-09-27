import os
import numpy as np
from scapy.all import rdpcap
import statistics

# Directories containing pcap files
directories = ['221']

# Function to compute packet statistics
def compute_statistics(pcap_file):
    # Read pcap file
    packets = rdpcap(pcap_file)
    
    t = (packets[13].time - packets[0].time) * 1000
    
    return float(t)

# Iterate over all directories and compute statistics for each pcap file
for directory in directories:
    print(f"time of 14th packet - CHROME/NECH/PQC")
    pcap_files = [f for f in os.listdir(directory) if f.endswith('.pcap')]

    packet_sizes_list = []

    for pcap_file in pcap_files:
        pcap_path = os.path.join(directory, pcap_file)

        packet_size_stats = compute_statistics(pcap_path)

        #Collect statistics
        packet_sizes_list.append(packet_size_stats)

    # Aggregate stats for all pcap files in the directory
    def aggregate_stats(stats_list):
        means = np.mean(stats_list)
        std_devs = np.std(stats_list)
        variances = np.var(stats_list)
        return means, std_devs, variances
    #def aggregate_stats(stats_list):
    #    means = np.mean(stats_list) if stats_list else 0

    packet_size_mean, packet_size_std, packet_size_var = aggregate_stats(packet_sizes_list)
    
    print(f"Avg packet time (in ms): {packet_size_mean:.2f} | Std Dev: {packet_size_std:.2f} | Variance: {packet_size_var:.2f}")

