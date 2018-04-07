import paramiko
import os
import io
import errno
import cv2
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('notebook')
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext, Row
from pyspark.mllib.clustering import KMeans
from scipy.spatial import distance
from pyspark.mllib.clustering import KMeansModel
import functools
from skimage import *
from skimage import color

spark = SparkSession.builder.appName("img_pipe.py").getOrCreate()


IMG_PATH ='/images/b2'
MODEL_PATH='/k_model'
FEATURE_PATH = '/xtract_feature'
SERVER = "127.0.0.1"
U_NAME = "***********"
PASSWORD = "***********"

LOCAL_PATH = "/home/amit/A1/b2"
sc =  spark.sparkContext
sqlContext = SQLContext(sc)
#data=images.frompng('/home/amit/A1',npartitions=8, engine=sc)


from thunder.readers import get_parallel_reader, FileNotFoundError
reader = get_parallel_reader(IMG_PATH)(sc)
#data = reader.read(IMG_PATH, recursive=True, npartitions=8)
from scipy.misc import imread

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(SERVER, username=U_NAME, password=PASSWORD)
ftp = ssh.open_sftp()

def readlocal(path, offset=None, size=-1):
    """
    Wrapper around open(path, 'rb') that returns the contents of the file as a string.
    Will rethrow FileNotFoundError if it receives an IOError.
    """
    #print(path)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SERVER, username=U_NAME, password=PASSWORD)


    try:
        ftp = ssh.open_sftp()

        file = ftp.file(path, "r", -1)
        buf = file.read()
        imagbuf = imread(BytesIO(buf))

        ftp.close()
    except IOError as e:
        if e.errno == errno.ENOENT:
            raise FileNotFoundError(e)
        else:
            raise
    #ftp.close()
    ssh.close()
    return path,imagbuf

def listrecursive(path, ext=None):
    """
    List files recurisvely
    """

    apath = path
    apattern = '"*"'
    rawcommand = 'find {path} -name {pattern}'
    command = rawcommand.format(path=apath, pattern=apattern)
    stdin, stdout, stderr = ssh.exec_command(command)
    filelist = stdout.read().splitlines()
    return filelist


files = listrecursive(LOCAL_PATH)
ftp.close()
ssh.close()
filenames = set()
ii=0
for file in files:
    if(ii!=0):
        filenames.add(file)

    ii=1

filenames = list(filenames)
filenames.sort()
#print(filenames)
#ftp = sc.broadcast(ftp)
rdd = sc.parallelize(enumerate(filenames), 8)

rdd = rdd.map(lambda kv: (readlocal(kv[1])))

from scipy.misc import imsave
from io import BytesIO
from thunder.writers import get_parallel_writer

def transform(kv):
    key, imgaes = kv

    imgaes=color.rgb2gray(imgaes)


    return [(key, imgaes)]

#rdd = rdd.flatMap(transform)
# writer = get_parallel_writer('/home/amit/processeda')('/home/amit/processeda', overwrite=True)
# data.foreach(lambda x: writer.write(tobuffer(x)))
# data = data.map(transform)
#print(data)


# data.topng('/home/amit/processeda',overwrite=True)

def extract_opencv_features(feature_name):
    def extract_opencv_features_nested(imgfile_imgbytes):
        try:

            fname, imgbytes = imgfile_imgbytes
            # img = cv2.GaussianBlur(imgbytes, (5, 5), 0)
            # img = cv2.Canny(img, 100, 200)
            # nparr = np.fromstring(buffer(imgbytes), np.uint8)
            # img = cv2.imdecode(nparr, 0)
            if feature_name in ["surf", "SURF"]:
                extractor = cv2.xfeatures2d.SURF_create()
            elif feature_name in ["sift", "SIFT"]:
                extractor = cv2.xfeatures2d.SIFT_create()
            kp, descriptors = extractor.detectAndCompute(img_as_ubyte(imgbytes), None)
            return [(fname, descriptors)]
        except Exception, e:
            print(e)
            return []

    return extract_opencv_features_nested


rdd = rdd.flatMap(extract_opencv_features("sift"))
#print(features)


def prnt(x):
    print(x.tolist())

# features.foreach(lambda x: prnt )
rdd = rdd.filter(lambda x: x[1] != None)
rdd = rdd.map(lambda x: (Row(fileName=x[0], features=x[1].tolist())))
#featuresSchema = sqlContext.createDataFrame(features)
#featuresSchema.registerTempTable("images")
#featuresSchema.write.parquet(home + FEATURE_PATH)
#featuresSchema.write.saveAsTable('img_feature',mode='overwrite')
#features = sqlContext.read.parquet(home + FEATURE_PATH)
#features = sqlContext.table('img_feature')
# Create same size vectors of the feature descriptors
# flatMap returns every list item as a new row for the RDD
# hence transforming x, 128 to x rows of 1, 128 in the RDD.
# This is needed for KMeans.
features = rdd.flatMap(lambda x: x['features']).cache()


def fit():
    model = KMeans.train(features, 3, maxIterations=10, initializationMode="random")
    return model

mod = fit()
#mod.save(sc, home + MODEL_PATH)
#print("Clusters have been saved as text file to %s" % MODEL_PATH)
#print("Final centers: " + str(mod.clusterCenters))

def assign_pooling(row, clusterCenters, pooling):
    image_name = row['fileName']
    feature_matrix = np.array(row['features'])
    clusterCenters = clusterCenters.value
    model = KMeansModel(clusterCenters)
    bow = np.zeros(len(clusterCenters))

    for x in feature_matrix:
        k = model.predict(x)
        dist = distance.euclidean(clusterCenters[k], x)
        if pooling == "max":
            bow[k] = max(bow[k], dist)
        elif pooling == "sum":
            bow[k] = bow[k] + dist
    clusters = bow.tolist()
    group = clusters.index(min(clusters)) + 1
    #print(image_name + " in group: " + str(group))
    return [(image_name, group)]

clusterCenters = mod.clusterCenters
clusterCenters = sc.broadcast(clusterCenters)

features_bow = rdd.map(functools.partial(assign_pooling, clusterCenters=clusterCenters, pooling='max'))
#print(features_bow.collect())

transport = paramiko.Transport((SERVER, 22))
transport.connect(username = U_NAME, password = PASSWORD)
sftp = paramiko.SFTPClient.from_transport(transport)
#sftp.mkdir("/home/amit/A1/regResult/")
dirs=set()
dirs_list = []
dirs_dict = dict()
clusters = features_bow.collect()
for lstr in clusters:
    group = lstr[0][1]
    img_name = lstr[0][0]
    if (group in dirs):
        dirs_list.append(img_name)
        dirs_dict.update({group: dirs_list})
    else:
        dirs_list = []
        dirs_list.append(img_name)
        dirs.add(group)
        dirs_dict.update({group: dirs_list})

for itms in dirs_dict.items():
    output = io.StringIO()
    for itm in itms[1]:
        output.write(unicode(itm+"\n", "utf-8"))

    f = sftp.open(LOCAL_PATH + str(itms[0])+".txt", 'wb')

    f.write(output.getvalue())
sftp.close()
#print(str(features_bow))
#featuresSchema = sqlContext.createDataFrame(features_bow)
#featuresSchema.registerTempTable("images")
#featuresSchema.write.parquet(RESULT_PATH)
sc.stop()
