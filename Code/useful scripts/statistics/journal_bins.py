#!/usr/bin/env python3
import os
import sys
import statistics
from scapy.all import rdpcap, TCP
import matplotlib.pyplot as plt
import numpy as np

# Fixed metric order and names
METRIC_NAMES = [
    "Packet Size (bytes)",
    "TLS Handshake Packets (count)",
    "Inter-arrival Time (s)",
    "ClientHello Record Length (bytes)",
    "ServerHello Record Length (bytes)",
    "Handshake Duration (s)",
]


def safe_mean(data):
    """Return the mean of a list or 0.0 if empty."""
    return statistics.mean(data) if data else 0.0


def extract_metrics_from_pcaps(pcap_files):
    """
    Parse a list of PCAP files and return raw metric lists.
    """
    pkt_sizes, hs_counts, iat, ch_sizes, sh_sizes, hs_durs = [], [], [], [], [], []

    for path in pcap_files:
        try:
            packets = rdpcap(path)
        except Exception as e:
            print(f"✗ can't read {path}: {e}")
            continue

        # Packet sizes
        sizes = [len(p) for p in packets]
        pkt_sizes.extend(sizes)

        # Inter-arrival times
        times = sorted(p.time for p in packets)
        iat.extend(t2 - t1 for t1, t2 in zip(times, times[1:]))

        # TLS handshake metrics
        hs_times = []
        count = 0
        for p in packets:
            if not p.haslayer(TCP):
                continue
            data = bytes(p[TCP].payload)
            if len(data) < 9 or data[0] != 0x16:
                continue
            rec_len = (data[3] << 8) | data[4]
            hs_type = data[5]
            hs_times.append(p.time)
            count += 1
            if hs_type == 0x01:
                ch_sizes.append(rec_len)
            elif hs_type == 0x02:
                sh_sizes.append(rec_len)
        hs_counts.append(count)
        hs_durs.append(max(hs_times) - min(hs_times) if hs_times else 0)

    return {
        METRIC_NAMES[0]: pkt_sizes,
        METRIC_NAMES[1]: hs_counts,
        METRIC_NAMES[2]: iat,
        METRIC_NAMES[3]: ch_sizes,
        METRIC_NAMES[4]: sh_sizes,
        METRIC_NAMES[5]: hs_durs,
    }


def gather_subdir_means(root_dir):
    """
    Compute the mean of each metric for each immediate subdirectory.
    """
    metrics_means = {m: {} for m in METRIC_NAMES}

    subdirs = [d for d in sorted(os.listdir(root_dir))
               if os.path.isdir(os.path.join(root_dir, d))]

    for name in subdirs:
        subdir = os.path.join(root_dir, name)
        # collect PCAP paths
        pcaps = []
        for dp, _, files in os.walk(subdir):
            for fn in files:
                if fn.lower().endswith((".pcap", ".pcapng")):
                    pcaps.append(os.path.join(dp, fn))
        print(f"→ {name}: {len(pcaps)} PCAP files")

        raw = extract_metrics_from_pcaps(pcaps)
        for m in METRIC_NAMES:
            metrics_means[m][name] = safe_mean(raw.get(m, []))

    return subdirs, metrics_means


def plot_grouped_bar(subdirs, metrics_means, out_path):
    """
    Plot a grouped bar chart:
    - X-axis: subdirectories
    - Each group has one bar per metric
    """
    n_sub = len(subdirs)
    n_met = len(METRIC_NAMES)
    x = np.arange(n_sub)
    total_width = 0.8
    width = total_width / n_met

    plt.figure(figsize=(14, 7))
    for i, metric in enumerate(METRIC_NAMES):
        # bar positions offset
        offsets = x - total_width/2 + i*width + width/2
        values = [metrics_means[metric][sd] for sd in subdirs]
        plt.bar(offsets, values, width=width, label=metric)

    plt.xticks(x, subdirs, rotation=45, ha='right')
    plt.xlabel('Subdirectory')
    plt.ylabel('Mean Value')
    plt.title('Grouped Metrics by Subdirectory')
    plt.legend(loc='upper right', ncol=2)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved grouped bar chart: {out_path}")


def main(root_dir):
    subdirs, metrics_means = gather_subdir_means(root_dir)
    out_file = os.path.abspath('output_barcharts/grouped_metrics.png')
    plot_grouped_bar(subdirs, metrics_means, out_file)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_dir>")
        sys.exit(1)
    main(sys.argv[1])
