#!/usr/bin/env python3

import sys
import os
import pyshark

def analyze_clienthello_split(pcap_file):
    """
    Analyzes a pcap file using PyShark to check whether
    the TLS ClientHello is split into multiple TCP segments
    or contained in one segment.
    
    Returns:
        bool: True if the pcap has at least one ClientHello
              split across multiple segments, False otherwise.
    """
    # Display filter to pick only TLS handshake packets where handshake.type == 1 (ClientHello).
    # Adjust 'tshark_path' to your environment as needed or remove if tshark is in your PATH.
    capture = pyshark.FileCapture(
        pcap_file,
        display_filter="tls.handshake.type == 1",
        tshark_path=r"D:\Wireshark\tshark.exe"  # Example Windows path
    )

    # If we find at least one ClientHello that is split, we'll set this flag.
    clienthello_split_found = False

    # Check each ClientHello in the file.
    for packet in capture:
        try:
            handshake_len = int(packet.tls.handshake_length)

            # Some versions of PyShark / TShark store length in 'packet.tcp.len'
            tcp_payload_len = int(packet.tcp.len) if hasattr(packet.tcp, 'len') else None

            if tcp_payload_len is not None:
                # If handshake_len > tcp_payload_len, the ClientHello is likely split
                if handshake_len > tcp_payload_len:
                    clienthello_split_found = True
                    # We can break after finding the first split, 
                    # but you might choose to keep analyzing. 
                    # For performance, we'll break here.
                    break

        except AttributeError:
            # Means some TLS or TCP fields were missing
            pass

    capture.close()
    return clienthello_split_found


def main():
    """
    Main entry point: given a top-level directory path,
    walk all subdirectories, find .pcap files, analyze each,
    and compute the percentage of PCAPs that have a split
    ClientHello.
    """
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory_path>")
        sys.exit(1)

    top_level_dir = sys.argv[1]

    # Counters to track how many PCAPs we analyzed
    # and how many of those contain a split ClientHello.
    total_pcaps = 0
    split_pcaps = 0

    # Walk the directory tree
    for root, dirs, files in os.walk(top_level_dir):
        for f in files:
            if f.lower().endswith(".pcap") or f.lower().endswith(".pcapng"):
                pcap_path = os.path.join(root, f)
                total_pcaps += 1

                # Check if the pcap has a split ClientHello
                if analyze_clienthello_split(pcap_path):
                    split_pcaps += 1
                    print(f"[SPLIT]  {pcap_path}")
                else:
                    print(f"[SINGLE] {pcap_path}")

    # Compute and display the percentage of PCAPs
    # that have a split ClientHello
    if total_pcaps > 0:
        split_percentage = (split_pcaps / total_pcaps) * 100
        print(f"\nAnalyzed {total_pcaps} PCAP(s)")
        print(f"ClientHello split was found in {split_pcaps} PCAP(s)")
        print(f"Percentage: {split_percentage:.2f}%")
    else:
        print("No PCAP files were found in the specified directory.")

if __name__ == "__main__":
    main()
