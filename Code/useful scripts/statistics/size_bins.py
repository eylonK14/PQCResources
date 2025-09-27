import os
import matplotlib.pyplot as plt
import numpy as np
from scapy.all import rdpcap, PcapReader
from collections import defaultdict
import glob

def analyze_pcap_packet_sizes(root_directory):
    """
    Analyze packet sizes from all PCAP files in subdirectories
    
    Args:
        root_directory (str): Path to the root directory containing subdirectories with PCAP files
    
    Returns:
        list: List of packet sizes
    """
    packet_sizes = []
    
    # Find all PCAP files in subdirectories
    pcap_patterns = ['*.pcap', '*.pcapng', '*.cap']
    pcap_files = []
    
    for pattern in pcap_patterns:
        pcap_files.extend(glob.glob(os.path.join(root_directory, '**', pattern), recursive=True))
    
    if not pcap_files:
        return []
    
    # Process each PCAP file
    for pcap_file in pcap_files:
        try:
            # Use PcapReader for memory efficiency with large files
            with PcapReader(pcap_file) as pcap_reader:
                for packet in pcap_reader:
                    packet_sizes.append(len(packet))
                    
        except Exception as e:
            continue
    
    return packet_sizes

def create_packet_size_histogram(packet_sizes, bin_size=500, max_size=16000):
    """
    Create histogram of packet sizes with custom bins
    
    Args:
        packet_sizes (list): List of packet sizes
        bin_size (int): Size of each bin (default: 500)
        max_size (int): Maximum packet size to include (default: 16000)
    """
    if not packet_sizes:
        return
    
    # Filter packets within the specified range
    filtered_sizes = [size for size in packet_sizes if size <= max_size]
    
    # Create bins: 0-500, 500-1000, 1000-1500, ..., up to max_size
    bins = list(range(0, max_size + bin_size, bin_size))
    
    # Create the histogram
    plt.figure(figsize=(15, 8))
    
    counts, bin_edges, patches = plt.hist(filtered_sizes, bins=bins, 
                                         alpha=0.7, color='skyblue', 
                                         edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    plt.xlabel('Packet Size (bytes)', fontsize=12)
    plt.ylabel('Number of Packets', fontsize=12)
    plt.title('Packet Size Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show every bin
    plt.xticks(bins, rotation=45)
    
    # Add some statistics as text
    total_packets = len(filtered_sizes)
    avg_size = np.mean(filtered_sizes) if filtered_sizes else 0
    median_size = np.median(filtered_sizes) if filtered_sizes else 0
    
    stats_text = f'Total Packets: {total_packets:,}\n'
    stats_text += f'Average Size: {avg_size:.1f} bytes\n'
    stats_text += f'Median Size: {median_size:.1f} bytes'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    # Get directory path from user
    root_dir = input("Enter the path to the directory containing subdirectories with PCAP files: ").strip()
    
    # Validate directory
    if not os.path.exists(root_dir):
        return
    
    if not os.path.isdir(root_dir):
        return
    
    # Analyze packet sizes
    packet_sizes = analyze_pcap_packet_sizes(root_dir)
    
    if not packet_sizes:
        return
    
    # Create histogram
    create_packet_size_histogram(packet_sizes)

if __name__ == "__main__":
    main()
