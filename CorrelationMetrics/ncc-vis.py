#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Libs
import pandas as pd;
import matplotlib.pyplot as plt

# Own modules
from utils import *
from matplotlib_vis_utils import *


# Read all the data files
input_df_set = loadDataFiles();

output_df_set = {};
for file in data_files:
    number_of_columns = len(input_df_set[file].columns);

    columns = ['X', 'Y'];
    for filter in window_set:
        columns.append(filter + "_ZNCC");

    output_df_set[file] = pd.DataFrame(columns = columns);

    for i in range(number_of_columns):
        for j in range(number_of_columns):

            new_row = {
                'X': input_df_set[file].columns[i],
                'Y': input_df_set[file].columns[j]
            };

            for filter in window_set:
                if filter == 'none':
                    series1 = input_df_set[file].iloc[:, i];
                    series2 = input_df_set[file].iloc[:, j];
                else:
                    series1, series2 = removeMissingData(input_df_set[file].iloc[:, i], input_df_set[file].iloc[:, j]);

                    # 11 is the default window size
                    if series1.shape[0] >= 11 and series2.shape[0] >= 11:
                        series1 = smooth(x = series1.astype(float), window=filter);
                        series2 = smooth(x = series2.astype(float), window=filter);

                if series1.shape[0] == 0 or series2.shape[0] == 0:
                    new_row[filter + "_ZNCC"] = 0;
                else:
                    new_row[filter + "_ZNCC"] = getZNCC(series1, series2);

            output_df_set[file] = output_df_set[file].append(new_row, ignore_index=True)




    output_df_set[file].to_csv("zncc_" + file);
    # np.savetxt("raw_zncc_" + file, output_df_set[file]);

    for filter in window_set:
        visualiseCorrelationMatrix(output_df_set[file], file, filter);
plt.show()


#
#
#
# print(input_df_set["cumulative-cases.csv"]["NHS Ayrshire & Arran"]);
#
# print(getZNCC(input_df_set["cumulative-cases.csv"]["NHS Ayrshire & Arran"], input_df_set["cumulative-cases.csv"]["NHS Ayrshire & Arran"]))
