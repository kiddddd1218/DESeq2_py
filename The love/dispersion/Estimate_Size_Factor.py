'''
*** Median-of-ratios method to normlize counts and calculate size factors.
*** df as input should take the gene names as the index and the samples as columns.
*** estimateSizeFactor() will return data frame of Normalized read counts and data frame of size_factors.
'''
import pandas as pd
import numpy as np
def estimateSizeFactor(df):
    # Step 1: take natural log of the read counts
    log_count_dic = {}
    for i in df.columns:
        log_count_dic[i] = np.log(df[i])
    log_count_df = pd.DataFrame(log_count_dic, index=df.index)
    # Step 2: compute the mean of each gene
    log_count_df["avgLogVal"] = log_count_df.mean(axis=1)
    # Step 3: filter out the gene with -inf log mean
    filtered_log_count_df = log_count_df[log_count_df["avgLogVal"]!=-np.inf]
    # Step 4: Substract the average log value from the log counts
    log_subtract_dic = {}
    for i in filtered_log_count_df.columns[:-1]:
        log_subtract_dic[i] = filtered_log_count_df.loc[:,i]- filtered_log_count_df.loc[:,"avgLogVal"]
    log_subtract_df = pd.DataFrame(log_subtract_dic, index=filtered_log_count_df.index)
    # Step 5 & 6: Calculate the median of the ratios for each sample
    # Convert the medians to "normal numbers" to get the final size factors for each sample
    size_factors = pd.DataFrame({"sizeFactor":np.exp(log_subtract_df.median())}, index=log_subtract_df.columns)
    # Step 7: Normalize the raw read counts by the size factors
    Normalized_dic = {}
    for i in size_factors.index:
        Normalized_dic[i] = df.loc[:,i]/size_factors.loc[i, "sizeFactor"]
    Normalized_df = pd.DataFrame(Normalized_dic, index=df.index)

    return Normalized_df, size_factors


if __name__=='__main__':
	print("*** Median-of-ratios method to normlize counts and calculate size factors.\n"+
    "*** df as input should take the gene names as the index and the samples as columns."+
    "\n*** estimateSizeFactor() will return data frame of Normalized read counts and data frame of size_factors.")