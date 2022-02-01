# Copyright (c) Microsoft Corporation
# Licensed under the MIT License.

import pandas as pd


def undummify(df, prefix_sep="_", col=None):
    if col == None:
        cols_to_collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols_to_collapse.items():
            if needs_to_collapse:
                undummified = (
                    df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)
    else:
        series_list = []
        collapse_df = df
        # df = df.rename(columns={'gender-f': 'gender-0', 'gender-m': 'gender-1'})
        undummified = (
            df.filter(like=col)
            .idxmax(axis=1)
            .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
            .rename(col)
        )
        series_list.append(undummified)
        collapse_df = collapse_df.loc[:, ~df.columns.str.startswith(col)]
        undummified_df = pd.concat(series_list, axis=1)
        undummified_df = pd.concat([undummified_df, collapse_df], axis=1)    
        undummified_df[col] = pd.Categorical(undummified_df[col], categories=undummified_df[col].unique()).codes  
    return undummified_df

# def undummify(df, prefix_sep="_", col_list=None):
#     if col_list == None:
#         cols_to_collapse = {
#             item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
#         }
#         series_list = []
#         for col, needs_to_collapse in cols_to_collapse.items():
#             if needs_to_collapse:
#                 undummified = (
#                     df.filter(like=col)
#                     .idxmax(axis=1)
#                     .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
#                     .rename(col)
#                 )
#                 series_list.append(undummified)
#             else:
#                 series_list.append(df[col])
#         undummified_df = pd.concat(series_list, axis=1)
#     else:
#         series_list = []
#         collapse_df = df
#         for col in col_list:
#             undummified = (
#                 df.filter(like=col)
#                 .idxmax(axis=1)
#                 .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
#                 .rename(col)
#             )
#             series_list.append(undummified)
#             collapse_df = collapse_df.loc[:, ~df.columns.str.startswith(col)]
#         undummified_df = pd.concat(series_list, axis=1)
#         undummified_df = pd.concat([undummified_df, collapse_df], axis=1)
#     return undummified_df
