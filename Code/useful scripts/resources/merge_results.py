#!/usr/bin/env python3

import os
import pandas as pd

'''
below is the first column of the csv file - not looking very good
0
210
211
220
221
310
311
320
321
410
411
420
421
430
620
630
accuracy
macro
weighted
0
0
1
accuracy
macro
weighted
0
10
20
accuracy
macro
weighted
0
200
300
400
accuracy
macro
weighted
'''

'''
README format specification - how the first column should look
PQC:
0 - Non-PQC
1 - PQC

Browsers:
10 - FireFox
20 - Chrome
30 - Safari
40 - iMessage

OS:
200 - Windows
300 - Linux
400 - MacOS
500 - Android
600 - iOS

Examples:
531 - Android/Safari/PQC
420 - Mac/Chrome/Non-PQC
'''


def main():
    format_spec = {
        "0": "Non-PQC",
        "1": "PQC",
        "10": "FireFox",
        "20": "Chrome",
        "30": "Safari",
        "40": "iMessage",
        "200": "Windows",
        "300": "Linux",
        "400": "MacOS",
        "500": "Android",
        "600": "iOS"
    }
    df = pd.DataFrame()
    with os.scandir(os.getcwd()) as it:
        for entry in it:
            if entry.name.endswith(".csv"):
                file_name = entry.name.split('_')[3]
                temp_df = pd.DataFrame([file_name], columns=["file_name"])
                modified_df = pd.read_csv(entry.name, index_col=0)
                # make the firsty column follow the format of the file name specified in the README
                # change it only if the first column is numeric
                if modified_df.index[0].isnumeric():
                    modified_df.index = modified_df.index.map(format_spec)
                    # make it compatible with combination of multiple values
                    modified_df.index = modified_df.index.str.replace(' ', '/')
                    modified_df.index = modified_df.index.str.replace('PQC', 'PQC/')
                    modified_df.index = modified_df.index.str.replace('Non-PQC', 'Non-PQC/')
                temp_df = pd.concat([temp_df, modified_df])
                df = pd.concat([df, temp_df])
    df.to_csv("merged-results.csv")


if __name__ == "__main__":
    main()
