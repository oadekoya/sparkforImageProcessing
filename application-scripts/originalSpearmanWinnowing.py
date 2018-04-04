import pandas
import networkx
import scipy.stats as stats

#this function removes data that is all zeros in a column
#best used once merging has taken place to get rid of all OTUs that are zero in both conditions
def remove_zero_data(df):
    return (df.loc[:, (df > min_count).any(axis=0)])

#Add one smoothing adds one to every cell then divides by the total, normalizing everything
#this is equivilent to having a uniform prior on the distriubtion of variables
#necessary for KL-Divergence calculation
def add_one_smoothing(df):
    temp  = df.copy() + 1
    temp = temp/df.sum()
    return temp

#this function computes the correlation matrix necessary to generate the graph
#any correlation supported by DataFrame.corr can be passed in
#Once MIC is implemented it will be added
def find_correlation(df, corr_type='spearman'):
    df_r = 0
    if corr_type == 'MIC':
        print('not yet implemented')
    else:
        df_r = df.corr(corr_type)
    if isinstance(df_r, pandas.DataFrame):
        df_r.fillna(0,inplace=True)  # ugly hack to make the NAs go away, should work for sampling but not advisable
        df_r = df_r[(df_r != 0).any(axis=1)]
        df_r = df_r.loc[:, (df_r != 0).any(axis=0)]

    return df_r

#this function returns the sorted centrality for a given centrality
#given a dataframe organized as an adjacency matrix, build a graph and compute the centrality
#return sorted centrality and the graph in networkx format
def find_centrality(df, cent_type='betweenness', keep_thresh=0.5):
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
    centrality_df = pandas.DataFrame.from_dict(centrality, orient='index')
    centrality_df.sort_values(0, axis=0, ascending=False, inplace=True)
    centrality_df = centrality_df.transpose()

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

#mainline program
#parameters necessary for the pipeline
#should be refactored as command line arguements in the future
in_file_path = '/Users/kgs325/dropbox/Soildata/bromedata/brome_input/'
out_file_path = '/Users/kgs325/dropbox/Soildata/bromedata/'
num_to_pull = 1
corr_thresh = 0.4
min_count = 3
num_required = 50

#open the files
brome1 = pandas.DataFrame.from_csv(in_file_path+'brome1A.csv') #read in input files
brome2 = pandas.DataFrame.from_csv(in_file_path+'brome2A.csv')
brome1_2 = brome1.append(brome2)
print("files read")

#Do some initial data formatting and cleaning
working_df_A = brome1.sum(axis=0).transpose() #KL divergence requires that histograms are the same size so sum to remove differences in number of samples
working_df_B = brome2.sum(axis=0).transpose()
working_df = add_one_smoothing(remove_zero_data(brome1_2.copy()))
working_df_A = add_one_smoothing(working_df_A.loc[working_df.columns]) #make sure that everything has the same number of columns
working_df_B = add_one_smoothing(working_df_B.loc[working_df.columns])

#loop working variable declaration
max_length = len(brome1_2.columns) #if you iterate over all OTUs you must be done
i = 0
downsampled_df = pandas.DataFrame()
diverge_vals = []
max_iter = len(working_df.columns)
#this is the select N bit, needs to be refactored into a function, but it also controls the looping and
#calls all the rest of the functions, so might wait until more components exist to do it right
while num_required > i*num_to_pull and i < max_iter:
    w_corr_df = find_correlation(working_df) #find the correlation
    cent_df,corr_graph = find_centrality(w_corr_df) #find the centrality
    drop_list = list(cent_df.columns[0:num_to_pull]) #find the top N OTU names
    downsampled_df = downsampled_df.append(cent_df.loc[:,drop_list]) #append them to the solution
    working_df.drop(drop_list, inplace=True, axis=1) #drop them from the inputs
    working_df_A.drop(drop_list, inplace=True)
    working_df_B.drop(drop_list, inplace=True)
    diverge_vals.append(find_kl_divergence(working_df_A, working_df_B)) #determine if the remaining histograms are more alike after elimination
    print(i) #output progress, should eventually refactor a silent mode
    print(diverge_vals[i])
    print(drop_list)
    i = i + 1


print('downsampling complete!')
#output results
#more results are possible from intermediate steps, should eventually refactor a verbose mode
working_df_A.to_csv(out_file_path+'brome1A_pruned.csv')
working_df_B.to_csv(out_file_path+'brome2A_pruned.csv')
working_df.to_csv(out_file_path+'brome12A_pruned.csv')
downsampled_df.to_csv(out_file_path+'brome12_down.csv')
temp = pandas.Series(diverge_vals)
temp.to_csv(out_file_path+'brome12_converge.csv')
