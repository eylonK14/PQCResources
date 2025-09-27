import os
import sys
import statistics
from scapy.all import rdpcap, TCP

def tls_client_hello_handshake_length(packet):
    if not packet.haslayer(TCP):
        return None

    payload = bytes(packet[TCP].payload)
    if len(payload) < 9:
        return None

    # TLS Handshake record: ContentType 0x16, HandshakeType 0x01
    if payload[0] == 0x16 and payload[5] == 0x01:
        return (payload[6] << 16) | (payload[7] << 8) | payload[8]
    return None

def get_client_hello_lengths(pcap_paths):
    """
    Given a list of PCAP file paths, return a tuple (lengths_list, pcap_count).
    """
    lengths = []
    for path in pcap_paths:
        try:
            packets = rdpcap(path)
        except Exception:
            continue
        for pkt in packets:
            hl = tls_client_hello_handshake_length(pkt)
            if hl is not None:
                lengths.append(hl)
    return lengths, len(pcap_paths)

def print_stats(name, lengths, pcap_count):
    count = len(lengths)
    mean_len = statistics.mean(lengths) if count else 0.0
    stdev_len = statistics.stdev(lengths) if count > 1 else 0.0
    avg_per_pcap = count / pcap_count if pcap_count else 0.0

    print(f"--- {name} ---")
    print(f"  PCAP files:              {pcap_count}")
    print(f"  ClientHello messages:    {count}")
    print(f"  Mean handshake length:   {mean_len:.2f}")
    print(f"  Std deviation:           {stdev_len:.2f}")
    print(f"  Avg ClientHello/PCAP:    {avg_per_pcap:.2f}")
    print()

def main(root_directory):
    # Collect all files by directory
    dir_to_pcaps = {}
    for dirpath, _, filenames in os.walk(root_directory):
        pcaps = [os.path.join(dirpath, f)
                 for f in filenames
                 if f.lower().endswith('.pcap')]
        if pcaps:
            dir_to_pcaps[dirpath] = pcaps

    # Global aggregation
    all_pcaps = []
    for pcaps in dir_to_pcaps.values():
        all_pcaps.extend(pcaps)

    # Print per-subdirectory stats
    for dirpath in sorted(dir_to_pcaps):
        lengths, count_pcaps = get_client_hello_lengths(dir_to_pcaps[dirpath])
        print_stats(dirpath, lengths, count_pcaps)

    # Print overall stats
    lengths_all, total_pcaps = get_client_hello_lengths(all_pcaps)
    print_stats("Overall", lengths_all, total_pcaps)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <root_directory>")
        sys.exit(1)
    main(sys.argv[1])
