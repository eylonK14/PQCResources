#!/usr/bin/env python3

from nfstream import NFStreamer
import TDL
import os
import pandas as pd
import ast
import re


def tdl(dir: str, num_of_pcaps: int, num_of_packets: int) -> None:
    all_files = []
    with os.scandir(dir) as it:
        for i, entry in enumerate(it):
            if i == num_of_pcaps:
                break
            streamer = NFStreamer(source=entry.path, udps=TDL.TDL())
            for _ in streamer:
                streamer.to_csv(path=f'{entry.name}.csv', flows_per_file=1)
                df = pd.read_csv(f'{entry.name}.0.csv')
                my_list = ast.literal_eval(df.iloc[0]["udps.ip_TDL"])
                my_list = [str(i) for i in my_list]
                # Change here: start from the 10th packet (index 9) and get the next num_of_packets packets.
                all_files.append(my_list[9:9+num_of_packets])
    new_df = pd.DataFrame(all_files)
    new_df["label"] = dir[-3:]
    new_df.to_csv(f'all_files_{dir[-3:]}.csv')


def delete_files(cwd: str, ext1: str) -> None:
    with os.scandir(cwd) as it:
        for entry in it:
            if entry.is_file() and re.search(ext1, entry.name):
                os.remove(entry.path)


def merge_csv_files(cwd, num_of_pcaps: int, num_of_packets: int) -> None:
    merged_df = pd.concat(
        [pd.read_csv(f)
         for f in os.listdir(cwd)
         if f.startswith('all_files_')]
    )
    merged_df = merged_df.drop(columns=['Unnamed: 0'])
    merged_df = merged_df.reset_index(drop=True)
    merged_df.to_csv(f'all-files-ech-noam-10.csv')


def runner(num_of_pcaps: int, num_of_packets: int) -> None:
    cwd = os.getcwd()
    # operating_systems = ['windows', 'linux', 'macos'] #  , 'ios']
    operating_systems = ['/DataAndModels/NoamData/Raw/ECH']
    for os_name in operating_systems:
        with os.scandir(f'../{os_name}') as it:
            for entry in it:
                print(entry)
                if entry.is_dir() and entry.name.isdigit():
                    tdl(entry.path, num_of_pcaps, num_of_packets)
    # dirs = ['122', 'chrome_true_new', 'firefox_true_new']
    # for i in dirs:
    #     tdl(f'/home/noam/Desktop/cs/PQC/oqs/{i}', 100, 25)
    delete_files(cwd, r'\.0\.csv')
    merge_csv_files(cwd, num_of_pcaps, num_of_packets)
    delete_files(cwd, r'[0-9]{3}\.csv')


def main() -> None:
    # for pcaps in range(10, 101, 10):
    # for packets in range(2, 5):
    runner(200, 15)


if __name__ == "__main__":
    main()
