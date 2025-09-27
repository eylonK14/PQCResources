import os
import sys
import statistics
from scapy.all import rdpcap, TCP

def tls_client_hello_handshake_length(packet):
    """
    Attempt to extract the Wireshark-like 'Handshake Length' field (3 bytes) 
    for a TLS ClientHello (handshake_type == 0x01).
    
    Returns:
      An integer representing the handshake_length (the value Wireshark shows 
      in 'Handshake Protocol: Client Hello (length: X)'),
      or None if this packet is not a ClientHello or is malformed.
    """
    if not packet.haslayer(TCP):
        return None
    
    tcp_payload = bytes(packet[TCP].payload)
    # Minimum 9 bytes: 
    #   5 for the TLS record header (record_type + version + length),
    #   plus at least 4 for the handshake header (type + 3-byte length).
    if len(tcp_payload) < 9:
        return None

    record_type = tcp_payload[0]       # 0x16 (22) => TLS Handshake
    handshake_type = tcp_payload[5]    # 0x01 => ClientHello
    if record_type == 0x16 and handshake_type == 0x01:
        # Extract the 3-byte handshake length field (bytes [6..8])
        handshake_len = (tcp_payload[6] << 16) | (tcp_payload[7] << 8) | tcp_payload[8]
        # This 'handshake_len' is exactly what Wireshark calls 
        # "Handshake Protocol: Client Hello (Length: X)" 
        return handshake_len
    
    return None

def get_client_hello_lengths(pcap_path):
    """
    Read a PCAP file and return a list of the TLS ClientHello handshake lengths.
    These lengths match the 'Length' field you see in Wireshark under
    'Handshake Protocol: Client Hello'.
    """
    lengths = []
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"[-] Could not read {pcap_path}: {e}")
        return lengths

    for pkt in packets:
        hello_len = tls_client_hello_handshake_length(pkt)
        if hello_len is not None:
            lengths.append(hello_len)

    return lengths


def main(directory):
    """
    Iterates over all .pcap files in the specified directory,
    collects the 'Wireshark-like' ClientHello handshake lengths,
    then prints out the total count, mean length, and std deviation.
    Also prints per-PCAP info on how many ClientHello messages are in each file,
    and each one's handshake length.
    """
    all_hello_lengths = []

    # Process each .pcap in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pcap"):
            pcap_path = os.path.join(directory, filename)
            c_hello_lengths = get_client_hello_lengths(pcap_path)

            if c_hello_lengths:
                print(f"[+] {pcap_path} has {len(c_hello_lengths)} ClientHello handshake(s):")
                for length in c_hello_lengths:
                    print(f"    - Handshake Length: {length} bytes")
            else:
                print(f"[!] {pcap_path} has no ClientHello messages.")

            # Accumulate for overall stats
            all_hello_lengths.extend(c_hello_lengths)

    # Compute overall statistics
    total_client_hellos = len(all_hello_lengths)
    if total_client_hellos > 0:
        mean_len = statistics.mean(all_hello_lengths)
        stdev_len = statistics.stdev(all_hello_lengths) if total_client_hellos > 1 else 0.0
    else:
        mean_len = 0.0
        stdev_len = 0.0

    print("\n=== ClientHello Handshake Length Statistics (Wireshark-like) ===")
    print(f"Total ClientHello messages:   {total_client_hellos}")
    print(f"Mean handshake length:        {mean_len:.2f}")
    print(f"Std Deviation:                {stdev_len:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <directory_of_pcaps>")
        sys.exit(1)

    main(sys.argv[1])
