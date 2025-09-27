import os
import sys  # <-- Import sys so we can call sys.exit()
import pyshark
import matplotlib.pyplot as plt
import numpy as np

def round_down_to_hundred(x: int) -> int:
    """Round x down to the nearest hundred."""
    return (x // 100) * 100

def round_up_to_hundred(x: int) -> int:
    """Round x up to the nearest hundred."""
    return ((x + 99) // 100) * 100

def collect_clienthello_packet_sizes(pcap_path: str) -> list[int]:
    """
    Given a path to a pcap/pcapng file, collect the length of the ClientHello
    portion of the TLS handshake, rather than the entire frame.

    If you get no packets, try using 'ssl.handshake.type == 1' instead 
    of 'tls.handshake.type == 1', depending on your Wireshark/TShark version.
    """
    sizes = []
    
    # Filter for TLS ClientHello (handshake type == 1).
    display_filter = "tls.handshake.type == 1"

    try:
        cap = pyshark.FileCapture(
            pcap_path,
            display_filter=display_filter,
            only_summaries=False,
            # If tshark is not in your PATH, uncomment and adjust the path below:
            tshark_path=r"D:\Wireshark\tshark.exe"
        )

        for packet in cap:
            # Depending on your version, the ClientHello length may appear as
            # 'packet.tls.handshake_length' or something similar.
            if hasattr(packet.tls, 'handshake_length'):
                clienthello_length = int(packet.tls.handshake_length)
                sizes.append(clienthello_length)
            else:
                # If no handshake_length is found, you can skip or fall back to packet.frame_info.len
                pass

        cap.close()
    except Exception as e:
        print(f"Error reading {pcap_path}: {e}")

    return sizes

def main(primary_directory: str):
    """
    1. Recursively walk through 'primary_directory'.
    2. For each .pcap / .pcapng file, gather the lengths of the ClientHello part
       (TLS handshake length) in each packet that carries a TLS ClientHello.
    3. Combine sizes, find the min and max, and round the boundaries to the
       nearest hundred (down for min, up for max).
    4. Plot a histogram with bin width = 100.
    """

    all_sizes = []

    # Walk the directory tree
    for root, dirs, files in os.walk(primary_directory):
        for file in files:
            if file.endswith(".pcap") or file.endswith(".pcapng"):
                pcap_file_path = os.path.join(root, file)
                print(f"Processing: {pcap_file_path}")

                # Collect the sizes (ClientHello handshake length only)
                clienthello_sizes = collect_clienthello_packet_sizes(pcap_file_path)
                all_sizes.extend(clienthello_sizes)

    if not all_sizes:
        print("No packets containing TLS ClientHello found (or no handshake length field).")
        return

    # Find minimum and maximum sizes
    min_size = min(all_sizes)
    max_size = max(all_sizes)

    # Round down the min, round up the max
    bin_start = round_down_to_hundred(min_size)
    bin_end   = round_up_to_hundred(max_size)

    # Create bins from bin_start to bin_end, stepping by 100
    bins = np.arange(bin_start, bin_end + 100, 100)

    # Build histogram data
    counts, edges = np.histogram(all_sizes, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts, width=100, align='center', edgecolor='black')
    plt.title("TLS ClientHello Message Size Distribution (Handshake Length) - ECH/NPQC")
    plt.xlabel("ClientHello Size (bytes) - Binned by 100")
    plt.ylabel("Count")
    plt.xticks(bin_centers, [str(int(x)) for x in bin_centers], rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # On Windows, specify your directory (raw string if you prefer):
    primary_directory_path = r"C:\Users\Eylon\PQC\windows\ech\npqc"
    main(primary_directory_path)
