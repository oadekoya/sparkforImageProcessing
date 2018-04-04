
from __future__ import print_function

from time import time
import cProfile
import pandas
import networkx
import scipy.stats as stats

from pyspark.sql import  SQLContext
from pyspark import SparkContext
from pyspark.sql import SparkSession
#from pyspark.sql.functions import col
#from pyspark.sql.types import LongType

#from pyspark.mllib.stat import Statistics


#this function removes data that is all zeros in a column
#best used once merging has taken place to get rid of all OTUs that are zero in both conditions
def remove_zero_data(df):
    return (df.loc[:, (df > min_count).any(axis=0)])




#Add one smoothing adds one to every cell then divides by the total, normalizing everything
#this is equivilent to having a uniform prior on the distriubtion of variables
#necessary for KL-Divergence calculation
def add_one_smoothing(df):
    temp  = df + 1
    temp = temp/(df.sum() + (len(df.index)))
    return temp



#this function computes the correlation matrix necessary to generate the graph
#any correlation supported by DataFrame.corr can be passed in
#Once MIC is implemented it will be added

def find_correlation(rdd):
    corr_mat=Statistics.corr(rdd, method="pearson")
    df_r = pandas.DataFrame(corr_mat)
   
    if isinstance(df_r, pandas.DataFrame):
        df_r.fillna(0,inplace=True)  # ugly hack to make the NAs go away, should work for sampling but not advisable
        df_r = df_r[(df_r != 0).any(axis=1)]
        df_r = df_r.loc[:, (df_r != 0).any(axis=0)]

    return df_r


#this function returns the sorted centrality for a given centrality
#given a dataframe organized as an adjacency matrix, build a graph and compute the centrality
#return sorted centrality and the graph in networkx format
def find_centrality(df, cent_type='betweenness', keep_thresh=0.5):
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    df_b = df.copy()
    df_b[(df.abs() < keep_thresh)] = 0 #eliminate edges that are too weak
    labels = list(df_b.index)
    temp = abs(df_b.copy())
    temp.insert(0, 'var1', labels)
    df_b = pandas.melt(temp, 'var1', var_name='var2', value_name='edge')
    df_b = df_b.loc[(df_b['edge'] > 0), :]  # take only those edge pairs that made the cut
    df_g = networkx.from_pandas_dataframe(df_b, 'var1', 'var2', 'edge')  # takes a list of valid edges
    if cent_type == 'betweenness':
        centrality = networkx.betweenness_centrality(df_g)
    elif cent_type == 'degree':
        centrality = networkx.degree_centrality(df_g)
    elif cent_type == 'closeness':
        centrality = networkx.closeness_centrality(df_g)
    elif cent_type == 'eigenvector':
        centrality = networkx.eigenvector_centrality(df_g)
    else:
        print('error, unknown centrality')
        return -1
    try:
        centrality_df = pandas.DataFrame.from_dict(centrality, orient='index')
        centrality_df.sort_values(0, axis=0, ascending=False, inplace=True)
        centrality_df = centrality_df.transpose()
    except (KeyError) as e:
        pass

    return centrality_df, df_g

#this function returns the KL divergence of two matched dataframes
#if the dataframes have more than one dimension they are flattened
#dataframes cannot contain zeros
def find_kl_divergence(df_A, df_B):
    if len(df_A.shape) > 1: #if there is more than one dimension, flatten
        tempA = df_A.values.flatten()
    else:
        tempA = df_A.values

    if len(df_B.shape) > 1: #if there is more than one dimension, flatten
        tempB = df_B.values.flatten()
    else:
        tempB = df_B.values

    kl_diverge = stats.entropy(tempA, tempB, 2.0) #find the KL-Divergence base 2
    if kl_diverge > 1e50:
        kl_diverge = 1e50

    return kl_diverge




def pipeline(num_2_pull):
    w_corr_df = (find_correlation(working_rdd)) #find the correlation
    cent_df,corr_graph = find_centrality(w_corr_df) #find the centrality
    drop_list = list(cent_df.columns[0:num_2_pull]) #find the top N OTU names
    downsampled_df = cent_df.loc[:,drop_list] #append them to the solution
    working_df.drop(drop_list, inplace=True, axis=1) #drop them from the inputs
    working_df_A.drop(drop_list, inplace=True)
    working_df_B.drop(drop_list, inplace=True)
    diverge_d = find_kl_divergence(working_df_A, working_df_B)

    return num_2_pull, diverge_d
        




if __name__ == "__main__":
    spark = SparkSession\
		.builder\
		.appName("spearman_winnowing")\
		.getOrCreate()

    num_to_pull = 10
    corr_thresh = 0.4
    min_count = 3
#num_required = 1

    count = 0

#pd_df_1 = pandas.DataFrame.from_csv('/data/mounted_hdfs_path/user/hduser/bromeData/brome1A.csv')
#pd_df_2 = pandas.DataFrame.from_csv('/data/mounted_hdfs_path/user/hduser/bromeData/brome2A.csv')

# KL divergence requires that histograms are the same size, so sum to remove differences in number of samples
#working_df_A = pd_df_1.sum(axis=0).transpose()
#working_df_B = pd_df_2.sum(axis=0).transpose()

#brome1_2_pd = pd_df_1.append(pd_df_2)

#zero_data_a = cProfile.run('remove_zero_data(brome1_2_pd)')

#working_df = add_one_smoothing(remove_zero_data(brome1_2_pd))
#print(working_df)

#working_df_A = add_one_smoothing(working_df_A.loc[working_df.columns])
#working_df_B = add_one_smoothing(working_df_B.loc[working_df.columns])

#processing_start_time = time()


#working_df.to_csv('/data/funmi/working_df.csv')
#working_df_A.to_csv('/data/funmi/working_df_A.csv')
#working_df_B.to_csv('/data/funmi/working_df_B.csv')
#sc = SparkContext(appName = "spearmann")
#sqlContext = SQLContext(sc)
#    a = spark.read.csv('hdfs://discus-p2irc-master:54310/user/hduser/bromeData/working_df_A.csv', header = True, inferSchema = True)
#a = sqlContext.read.load('file:///home/hduser/Downloads/ow_temp.csv', format='com.databricks.spark.csv', header='true', inferSchema='true')
    a = spark.read.csv('file:///home/hduser/ow_temp_2.csv', header = True, inferSchema = True)
    working_rdd = a.rdd.map(lambda r : r[0:]).cache()
    

#   processing_start_time = time()

#    pd_df_1 = pandas.DataFrame.from_csv('file:///data/funmi/working_df.csv')
#    working_rdd = pd_df_1.rdd.map(lambda r : r[0:]).cache()
    #working_df = sc.textFile('/data/mounted_hdfs_path/user/hduser/bromeData/working_df')

#data = sc.parallelize(working_df).collect()
#print(type(data))
#df_brome1 = sqlContext.createDataFrame(pd_df_1)
#df_brome2 = sqlContext.createDataFrame(pd_df_2)


#sqlContext = SQLContext(sc)

# Convert pandas working_df to spark dataframe an
    #working_rdd = sc.parallelize(working_df.values)

#working_rdd.collect()

#w_corr_df = find_correlation(working_rdd)


    diverge_vals = sc.parallelize(xrange(num_to_pull)).map(pipeline).collect()

#print(diverge_vals)
#print('downsampling complete!')

    #processing_end_time = time() - processing_start_time

    #print("SUCCESS: KL-Divergence calculated in {} seconds".format(round(processing_end_time, 3)))
    spark.stop()



