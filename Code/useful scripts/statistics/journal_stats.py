#!/usr/bin/env python3
import os
import sys
import statistics
from scapy.all import rdpcap, TCP

def safe_stats(data):
    """Calculate range, mean, and standard deviation for the data."""
    if not data:
        return (0, 0), 0, 0
    
    return (
        (min(data), max(data)),  # range (min, max)
        statistics.mean(data),
        statistics.pstdev(data),
    )

def process_directory(root_dir):
    packet_sizes_bytes        = []
    interarrival_times_ms     = []
    client_hello_sizes        = []
    server_hello_sizes        = []
    
    # Track max ClientHello and ServerHello with their files
    max_client_hello = {'size': 0, 'file': 'N/A'}
    max_server_hello = {'size': 0, 'file': 'N/A'}

    for dirpath, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.lower().endswith(('.pcap', '.pcapng')):
                continue
            path = os.path.join(dirpath, fn)
            try:
                packets = rdpcap(path)
            except Exception as e:
                print(f"   ✗ failed to read {path}: {e}")
                continue

            if not packets:
                continue

            # Packet sizes
            sizes = [len(pkt) for pkt in packets]
            packet_sizes_bytes.extend(sizes)

            # Inter-arrival times → milliseconds
            times = sorted(float(pkt.time) for pkt in packets)
            interarrival_times_ms.extend(
                (times[i] - times[i-1]) * 1000.0
                for i in range(1, len(times))
            )

            # TLS handshake metrics (ClientHello and ServerHello)
            for pkt in packets:
                if not pkt.haslayer(TCP):
                    continue
                data = bytes(pkt[TCP].payload)
                
                # Skip empty payloads
                if len(data) < 9:
                    continue
                
                # Check for TLS Handshake record (0x16) and valid TLS version
                if data[0] != 0x16:
                    continue
                
                # Check for valid TLS/SSL version (0x0301-0x0304 for TLS 1.0-1.3)
                version = (data[1] << 8) | data[2]
                if version < 0x0301 or version > 0x0304:
                    continue
                
                # Get the TLS record length
                record_len = (data[3] << 8) | data[4]
                
                # Sanity check: record length shouldn't be huge
                if record_len > 16384:  # TLS max record size
                    continue
                
                # Check we have enough data for the handshake header
                if len(data) < 9:
                    continue
                
                # Get handshake type
                hs_type = data[5]
                
                # Get handshake length (3 bytes, big-endian)
                hs_len = (data[6] << 16) | (data[7] << 8) | data[8]
                
                # Sanity check handshake length
                if hs_len > record_len - 4:  # Can't be larger than record payload minus handshake header
                    continue
                
                # Use the entire packet length to match Wireshark's display
                # This includes Ethernet + IP + TCP headers + TCP payload
                packet_size = len(pkt)

                if hs_type == 0x01:  # ClientHello
                    client_hello_sizes.append(packet_size)
                    if packet_size > max_client_hello['size']:
                        max_client_hello['size'] = packet_size
                        max_client_hello['file'] = path
                elif hs_type == 0x02:  # ServerHello  
                    server_hello_sizes.append(packet_size)
                    if packet_size > max_server_hello['size']:
                        max_server_hello['size'] = packet_size
                        max_server_hello['file'] = path

    # Print statistics
    print("\n" + "="*60)
    print(" "*20 + "Dataset Statistics")
    print("="*60 + "\n")

    stats_metrics = [
        ("Packet Size (bytes)", packet_sizes_bytes),
        ("Inter-arrival Time (ms)", interarrival_times_ms),
        ("ClientHello Size (bytes)", client_hello_sizes),
        ("ServerHello Size (bytes)", server_hello_sizes),
    ]

    print(f"{'Metric':<30} {'Range (min-max)':<20} {'Mean':<12} {'STD':<12}")
    print("-"*60)
    
    for name, data in stats_metrics:
        range_vals, mean_val, std_val = safe_stats(data)
        if data:  # Only print if we have data
            range_str = f"[{range_vals[0]:.1f}, {range_vals[1]:.1f}]"
            print(f"{name:<30} {range_str:<20} {mean_val:<12.3f} {std_val:<12.3f}")
        else:
            print(f"{name:<30} {'No data':<20} {'N/A':<12} {'N/A':<12}")
    
    print("\n" + "="*60)
    print(f"Total packets analyzed: {len(packet_sizes_bytes)}")
    print(f"Total IAT samples: {len(interarrival_times_ms)}")
    print(f"ClientHello packets found: {len(client_hello_sizes)}")
    print(f"ServerHello packets found: {len(server_hello_sizes)}")
    print("="*60)
    
    # Print max packet information
    if max_client_hello['size'] > 0:
        print(f"\nMaximum ClientHello: {max_client_hello['size']} bytes")
        print(f"  → From file: {max_client_hello['file']}")
    
    if max_server_hello['size'] > 0:
        print(f"\nMaximum ServerHello: {max_server_hello['size']} bytes")
        print(f"  → From file: {max_server_hello['file']}")
    
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_dir>")
        sys.exit(1)
    process_directory(sys.argv[1])