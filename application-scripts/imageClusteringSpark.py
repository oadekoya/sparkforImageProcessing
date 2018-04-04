import os
import cv2
import numpy as np
import subprocess
from StringIO import StringIO
from PIL import Image, ImageFile
from time import time
import copy
import glob
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
import skimage.io as io
#from pyspark.ml.clustering import KMeans
from pyspark.mllib.clustering import KMeans
from scipy.spatial import distance
#from pyspark.ml.clustering import KMeansModel
from pyspark.mllib.clustering import KMeansModel
import functools
from skimage import *
from skimage import color
from pyspark.sql import SQLContext, Row
from io import BytesIO
from scipy.misc import imread, imsave


import operator

import pyspark

from PIL import ImageFile


def images_to_bytes(rawdata):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return rawdata[0], np.array(io.imread(StringIO(rawdata[1])))
    except (ValueError, IOError, SyntaxError) as e:
        pass


def imagesToBytes(images):
    try:
        (key, value) = images
        imagbuf = imread(BytesIO(value))
    except IOError as e:
        if e.errno == errno.ENOENT:
            raise FileNotFoundError(e)
        else:
            raise
    return key, imagbuf




def transform(kv):
    key, imgaes = kv

    imgaes=color.rgb2gray(imgaes)


    return [(key, imgaes)]






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



def buildModel():
    model = KMeans.train(features, 3, maxIterations=5, initializationMode="random")
    return model




def assign_pooling(row, clusterCenters, pooling):
    image_name = row['fileName']
    feature_matrix = np.array(row['features'])
    clusterCenters = clusterCenters.value
    #model = KMeansModel(clusterCenters).cache()
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






if __name__ == "__main__":
    # Total time is the time from where the SparkContext is initiated to its shutting down, that is, the build time + processing time + overhead
    total_time_start = time()

    job_name = sys.argv[1]
    sc = SparkContext(appName = job_name)

    # Remove the output directory and all it contents if it already exist
    subprocess.call(["hadoop", "fs", "-rm", "-r", "/tmp/output_image/"])


    # Time to build the data structures and load input data for computation
    processing_start_time = time()

    images_read_rdd = sc.binaryFiles('hdfs://discus-p2irc-master:54310/user/hduser/timelapse_images_15072016')
    #print(images_read_rdd.getNumPartitions())
    images_buf = images_read_rdd.map(images_to_bytes)

    #images_part = images_buf.repartition(3000)



    images_features = images_buf.flatMap(extract_opencv_features("sift"))


    filtered_features = images_features.filter(lambda x: x[1] != None)
    features_with_filenames = filtered_features.map(lambda x: (Row(fileName=x[0], features=x[1].tolist())))

    features = features_with_filenames.flatMap(lambda x: x['features'])


    mod = buildModel()


    clusterCenters = mod.clusterCenters
    clusterCenters = sc.broadcast(clusterCenters)

    features_bow = features_with_filenames.map(functools.partial(assign_pooling, clusterCenters=clusterCenters, pooling='max'))

    features_bow.coalesce(1, shuffle=True).saveAsTextFile("hdfs://discus-p2irc-master:54310/tmp/output_image/")

    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

    sc.stop()

    total_time_end = time() - total_time_start

    print "SUCCESS: Total application time in {} seconds".format(round(total_time_end, 3))
