# Calculate each AU weights.
import numpy as np
import pandas as pd

def calculate_AU_weight(occurence_df):
    """
    Calculates the AU weight according to a occurence dataframe 
    inputs: 
        occurence_df: a pandas dataframe containing occurence of each AU. See BP4D+
    """
    #occurence_df = occurence_df.rename(columns = {'two':'new_name'})
    weight_mtrx = np.zeros((occurence_df.shape[1], 1))
    for i in range(occurence_df.shape[1]):
        weight_mtrx[i] = np.sum(occurence_df.iloc[:, i]
                                > 0) / float(occurence_df.shape[0])
    weight_mtrx = 1.0/weight_mtrx

    print(weight_mtrx)
    weight_mtrx[weight_mtrx == np.inf] = 0
    print(np.sum(weight_mtrx)*len(weight_mtrx))
    weight_mtrx = weight_mtrx / (np.sum(weight_mtrx)*len(weight_mtrx))

    return(weight_mtrx)
