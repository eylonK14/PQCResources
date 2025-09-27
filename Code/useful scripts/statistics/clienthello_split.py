#!/usr/bin/env python3

import sys
import pyshark

def analyze_clienthello_split(pcap_file):
    """
    Analyzes a pcap file using PyShark to check whether
    the TLS ClientHello is split into multiple TCP segments
    or contained in one segment.
    """

    # Display filter to pick only TLS handshake packets where handshake.type == 1 (ClientHello)
    # Note: "tls.handshake.type == 1" is recognized by newer Wireshark/TShark versions
    capture = pyshark.FileCapture(
        pcap_file,
        display_filter="tls.handshake.type == 1",
        tshark_path=r"D:\Wireshark\tshark.exe"
    )

    print(f"Analyzing file: {pcap_file}")
    print("--------------------------------------------------")

    # If there are multiple ClientHellos in the capture, we'll look at each.
    for i, packet in enumerate(capture, start=1):
        print(f"\n--- ClientHello #{i} ---")

        # Safely try to read relevant fields
        try:
            # The handshake_length is the length of the ClientHello portion of the handshake
            handshake_len = int(packet.tls.handshake_length)

            # The record_length is the total TLS record length (often includes handshake + header)
            # If available, you can see how big the entire TLS record is
            # record_len = int(packet.tls.record_length)  # (Optional usage)

            # TCP segment payload length
            # Depending on your Wireshark version, this might be packet.tcp.len or packet.tcp.segment_len
            # We'll use packet.tcp.len if it exists
            tcp_payload_len = int(packet.tcp.len) if hasattr(packet.tcp, 'len') else None

            # Print the captured fields
            print(f"  TLS handshake (ClientHello) length: {handshake_len} bytes")
            if tcp_payload_len is not None:
                print(f"  TCP segment data length:           {tcp_payload_len} bytes")

                # Very rough check:
                # If handshake_len <= tcp_payload_len, then the entire ClientHello likely fits in one segment
                # If handshake_len > tcp_payload_len, it likely means multiple segments were used.
                if handshake_len <= tcp_payload_len:
                    print("  => This ClientHello is likely NOT split across multiple TCP segments.")
                else:
                    print("  => This ClientHello is likely SPLIT across multiple TCP segments.")
            else:
                print("  (No TCP length field found; cannot determine split vs. single segment.)")

        except AttributeError as e:
            # Means some TLS or TCP fields were missing
            print(f"  [Error reading packet fields: {e}]")

    capture.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <capture.pcap>")
        sys.exit(1)

    pcap_file = sys.argv[1]
    analyze_clienthello_split(pcap_file)
