from __future__ import print_function

from time import time
import pandas
import networkx
import scipy.stats as stats

from pyspark.sql import  SQLContext
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import LongType

from pyspark.mllib.stat import Statistics


#this function computes the correlation matrix necessary to generate the graph
#any correlation supported by DataFrame.corr can be passed in
#Once MIC is implemented it will be added

def calculate_features(rdd):
    #print(rdd.collect())
    features = rdd.map(lambda row: compute_correlation(row[0:]))
    #print(features)
    #corr_mat=Statistics.corr(features, method="pearson")
    #df_r = pandas.DataFrame(corr_mat)
        #df_r = df.corr(corr_type)
    #if isinstance(df_r, pandas.DataFrame):
    #    df_r.fillna(0,inplace=True)  # ugly hack to make the NAs go away, should work for sampling but not advisable
    #    df_r = df_r[(df_r != 0).any(axis=1)]
    #    df_r = df_r.loc[:, (df_r != 0).any(axis=0)]

    return df_r


def compute_correlation(features):
    corr_mat=Statistics.corr(features, method="pearson")
    
    return corr_mat




def compute_correlation_pandas(df, corr_type=''):
    df_r = 0
    

processing_start_time = time()

sc = SparkContext(appName = "correlation")
sqlContext = SQLContext(sc)


working_rdd = (sqlContext.createDataFrame(working_df)).rdd


features = working_rdd.map(lambda row: row[0:])
#correla_Matrix = find_correlation(working_rdd)
print(features)
#print(correla_Matrix)



processing_end_time = time() - processing_start_time

print("SUCCESS: Pearson Correlation calculated in {} seconds".format(round(processing_end_time, 3)))
sc.stop()





