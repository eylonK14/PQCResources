#!/usr/bin/env python3
"""
PCAP Large Packet Analyzer
Analyzes PCAP files to find packets larger than 15KB and provides statistics.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

try:
    from scapy.all import rdpcap, Packet
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.inet6 import IPv6
    from scapy.layers.dns import DNS
    from scapy.layers.http import HTTP
except ImportError:
    print("Error: scapy is not installed. Please install it using:")
    print("pip install scapy")
    sys.exit(1)


def get_packet_type(packet: Packet) -> str:
    """
    Determine the type/protocol of a packet.
    
    Args:
        packet: A scapy packet object
        
    Returns:
        String describing the packet type
    """
    packet_type = []
    
    # Check for IP layer
    if IP in packet:
        packet_type.append("IPv4")
    elif IPv6 in packet:
        packet_type.append("IPv6")
    
    # Check for transport layer
    if TCP in packet:
        packet_type.append("TCP")
        tcp_layer = packet[TCP]
        
        # Check for common application layer protocols based on ports
        if tcp_layer.dport == 80 or tcp_layer.sport == 80:
            packet_type.append("HTTP")
        elif tcp_layer.dport == 443 or tcp_layer.sport == 443:
            packet_type.append("HTTPS/TLS")
        elif tcp_layer.dport == 22 or tcp_layer.sport == 22:
            packet_type.append("SSH")
        elif tcp_layer.dport == 21 or tcp_layer.sport == 21:
            packet_type.append("FTP-Control")
        elif tcp_layer.dport == 20 or tcp_layer.sport == 20:
            packet_type.append("FTP-Data")
        elif tcp_layer.dport == 25 or tcp_layer.sport == 25:
            packet_type.append("SMTP")
        elif tcp_layer.dport == 110 or tcp_layer.sport == 110:
            packet_type.append("POP3")
        elif tcp_layer.dport == 143 or tcp_layer.sport == 143:
            packet_type.append("IMAP")
        elif tcp_layer.dport == 3306 or tcp_layer.sport == 3306:
            packet_type.append("MySQL")
        elif tcp_layer.dport == 5432 or tcp_layer.sport == 5432:
            packet_type.append("PostgreSQL")
        elif tcp_layer.dport == 3389 or tcp_layer.sport == 3389:
            packet_type.append("RDP")
        else:
            # If payload exists, it's application data
            if len(tcp_layer.payload) > 0:
                packet_type.append("Application Data")
                
    elif UDP in packet:
        packet_type.append("UDP")
        udp_layer = packet[UDP]
        
        # Check for common UDP protocols
        if udp_layer.dport == 53 or udp_layer.sport == 53:
            packet_type.append("DNS")
        elif udp_layer.dport == 67 or udp_layer.dport == 68:
            packet_type.append("DHCP")
        elif udp_layer.dport == 123 or udp_layer.sport == 123:
            packet_type.append("NTP")
        elif udp_layer.dport == 161 or udp_layer.dport == 162:
            packet_type.append("SNMP")
        elif udp_layer.dport == 500 or udp_layer.sport == 500:
            packet_type.append("IKE/IPSec")
        elif 5060 <= udp_layer.dport <= 5061 or 5060 <= udp_layer.sport <= 5061:
            packet_type.append("SIP")
        else:
            if len(udp_layer.payload) > 0:
                packet_type.append("Application Data")
                
    elif ICMP in packet:
        packet_type.append("ICMP")
    
    # Check if DNS layer is present
    if DNS in packet:
        if "DNS" not in packet_type:
            packet_type.append("DNS")
    
    # If we couldn't determine the type
    if not packet_type:
        packet_type.append("Unknown")
    
    return "/".join(packet_type)


def analyze_pcap_file(pcap_path: str, size_threshold: int = 15000) -> Tuple[int, int, Dict[str, int]]:
    """
    Analyze a single PCAP file for packets larger than the threshold.
    
    Args:
        pcap_path: Path to the PCAP file
        size_threshold: Size threshold in bytes (default 15000 for 15KB)
        
    Returns:
        Tuple of (total_packets, large_packets, packet_types_dict)
    """
    try:
        packets = rdpcap(pcap_path)
        total_packets = len(packets)
        large_packets = 0
        packet_types = defaultdict(int)
        
        for packet in packets:
            packet_size = len(packet)
            if packet_size > size_threshold:
                large_packets += 1
                packet_type = get_packet_type(packet)
                packet_types[packet_type] += 1
        
        return total_packets, large_packets, dict(packet_types)
        
    except Exception as e:
        print(f"Error processing {pcap_path}: {e}")
        return 0, 0, {}


def find_pcap_files(directory: str) -> List[str]:
    """
    Recursively find all PCAP files in a directory tree.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of paths to PCAP files
    """
    pcap_files = []
    path = Path(directory)
    
    # Common PCAP file extensions
    pcap_extensions = {'.pcap', '.pcapng', '.cap', '.dmp'}
    
    for file_path in path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in pcap_extensions:
            pcap_files.append(str(file_path))
    
    return pcap_files


def main():
    parser = argparse.ArgumentParser(
        description='Analyze PCAP files for packets larger than 15KB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/pcap/directory
  %(prog)s /path/to/pcap/directory --threshold 20000
  %(prog)s /path/to/pcap/directory --verbose
        """
    )
    
    parser.add_argument(
        'directory',
        help='Directory containing PCAP files (searches recursively)'
    )
    
    parser.add_argument(
        '--threshold',
        type=int,
        default=15000,
        help='Size threshold in bytes (default: 15000 for 15KB)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output for each PCAP file'
    )
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory")
        sys.exit(1)
    
    # Find all PCAP files
    print(f"Searching for PCAP files in: {args.directory}")
    pcap_files = find_pcap_files(args.directory)
    
    if not pcap_files:
        print("No PCAP files found in the specified directory tree")
        sys.exit(0)
    
    print(f"Found {len(pcap_files)} PCAP file(s)")
    print(f"Analyzing packets larger than {args.threshold} bytes ({args.threshold/1000:.1f}KB)...")
    print("-" * 60)
    
    # Initialize counters
    total_packets_all = 0
    large_packets_all = 0
    packet_types_all = defaultdict(int)
    
    # Process each PCAP file
    for i, pcap_file in enumerate(pcap_files, 1):
        if args.verbose:
            print(f"\n[{i}/{len(pcap_files)}] Processing: {pcap_file}")
        
        total, large, types = analyze_pcap_file(pcap_file, args.threshold)
        
        if args.verbose and total > 0:
            percentage = (large / total * 100) if total > 0 else 0
            print(f"  Total packets: {total}")
            print(f"  Large packets: {large} ({percentage:.2f}%)")
            if types:
                print("  Packet types:")
                for ptype, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {ptype}: {count}")
        
        # Aggregate results
        total_packets_all += total
        large_packets_all += large
        for ptype, count in types.items():
            packet_types_all[ptype] += count
    
    # Calculate overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    
    if total_packets_all == 0:
        print("No packets found in the PCAP files")
        return
    
    percentage_large = (large_packets_all / total_packets_all * 100)
    
    print(f"\n1. Total packets analyzed: {total_packets_all:,}")
    print(f"\n2. Packets larger than {args.threshold} bytes ({args.threshold/1000:.1f}KB):")
    print(f"   - Count: {large_packets_all:,}")
    print(f"   - Percentage: {percentage_large:.2f}%")
    
    if packet_types_all:
        print(f"\n3. Types of packets larger than {args.threshold} bytes:")
        print("   " + "-" * 50)
        print(f"   {'Packet Type':<35} {'Count':<10} {'Percentage'}")
        print("   " + "-" * 50)
        
        for ptype, count in sorted(packet_types_all.items(), key=lambda x: x[1], reverse=True):
            pct = (count / large_packets_all * 100) if large_packets_all > 0 else 0
            print(f"   {ptype:<35} {count:<10,} {pct:>6.2f}%")
        
        print("   " + "-" * 50)
    else:
        print("\n3. No packets larger than the threshold were found")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()