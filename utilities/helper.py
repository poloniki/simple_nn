import numpy as np
import pandas as pd
def add_number_to_duplicates(df, column_name):
    value_counts = {}

    for i, value in enumerate(df[column_name]):
        if value in value_counts:
            value_counts[value] += 1
            df.loc[i, column_name] = f"{value}_{value_counts[value]}"
        else:
            value_counts[value] = 1

    return df

def highlight(data):
    is_min = data == data.nsmallest(1).iloc[-1]
    is_max = data == data.nlargest(1).iloc[0]
    styles = [''] * len(data)
    min_index = np.flatnonzero(is_min.to_numpy())
    max_index = np.flatnonzero(is_max.to_numpy())
    for i in min_index:
        styles[i] = 'background-color: rgba(255,0,0,0.3)'
    for i in max_index:
        styles[i] = 'background-color: rgba(0,255,0,0.3)'
    return styles


def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
