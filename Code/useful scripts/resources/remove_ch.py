import os
import pandas as pd
import ast
import re


def tdl(dir: str, num_of_pcaps: int, num_of_packets: int) -> None:
    all_files = []
    print("hello ya cunt")
    with os.scandir(dir) as it:
        for i, entry in enumerate(it):
            if i == num_of_pcaps:
                break
                print("hello ya cunt")
                df = pd.read_csv(f'{entry.name}.0.csv')
                my_list = ast.literal_eval(df.iloc[0]["udps.ip_TDL"])
                print(my_list)
                my_list = [str(i) for i in my_list]
                all_files.append(my_list[0:num_of_packets])
                print(all_files)
    new_df = pd.DataFrame(all_files)
    print(new_df)
    print("hi")
    #new_df["label"] = dir[-3:]
    #new_df.to_csv(f'all_files_{dir[-3:]}.csv')
    
    
def runner(num_of_pcaps: int, num_of_packets: int) -> None:
    cwd = os.getcwd()
    # operating_systems = ['windows', 'linux', 'macos'] #  , 'ios']
    operating_systems = ['windows/test']
    for os_name in operating_systems:
        with os.scandir(f'../{os_name}') as it:
            for entry in it:
                print(entry)
                if entry.is_dir() and entry.name.isdigit():
                    tdl(entry.path, num_of_pcaps, num_of_packets)
    # dirs = ['122', 'chrome_true_new', 'firefox_true_new']
    # for i in dirs:
    #     tdl(f'/home/noam/Desktop/cs/PQC/oqs/{i}', 100, 25)
    #delete_files(cwd, r'\.0\.csv')
    #merge_csv_files(cwd, num_of_pcaps, num_of_packets)
    #delete_files(cwd, r'[0-9]{3}\.csv')


def main() -> None:
    # for pcaps in range(10, 101, 10):
    # for packets in range(2, 5):
    runner(100, 30)


if __name__ == "__main__":
    main()