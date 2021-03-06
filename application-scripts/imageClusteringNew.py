import os
import sys
import cv2
import pyspark
import subprocess
import numpy as np
import skimage.io as io

from skimage import *
from time import time
from PIL import ImageFile
from pyspark import SparkConf
from pyspark import SparkContext
from io import StringIO, BytesIO
from scipy.spatial import distance
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.clustering import KMeansModel


def images_to_descriptors(rawdata):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        fname = rawdata[0]
        #imgbytes = np.array(io.imread(StringIO(rawdata[1])))
        imgbytes = np.array(io.imread(BytesIO(rawdata[1])))
        extractor = cv2.xfeatures2d.SIFT_create()
        kp, descriptors = extractor.detectAndCompute(img_as_ubyte(imgbytes), None)
        return [fname, descriptors]

    except (ValueError, IOError, SyntaxError) as e:
        pass


def assign_pooling(data):

    image_name, feature_matrix = data[0]
    clusterCenters = data[1]

    feature_matrix = np.array(feature_matrix)

    model = KMeansModel(clusterCenters)
    bow = np.zeros(len(clusterCenters))

    for x in feature_matrix:
        k = model.predict(x)
        dist = distance.euclidean(clusterCenters[k], x)
        bow[k] = max(bow[k], dist)

    clusters = bow.tolist()
    group = clusters.index(min(clusters)) + 1
    return [image_name, group]


if __name__ == "__main__":

    application_start_time =  time()

    #input_path = sys.argv[1]
    #output_path = sys.argv[2]
    job_name = sys.argv[1]

    subprocess.call(["hadoop", "fs", "-rm", "-r", "/tmp/output_image/"])

    sc = SparkContext(appName = job_name)

    build_start_time = time()

    images_rdd = sc.binaryFiles('hdfs://discus-p2irc-master:54310/user/hduser/timelapse_images_15072016') \
        .map(images_to_descriptors) \
        .filter(lambda x: x[1].all() != None) \
        .map(lambda x: (x[0], x[1]))

    """
        .zipWithIndex()

    training_set = images_rdd.filter(lambda x: x[1]%2 == 0) \
        .map(lambda x: x[0])

    clustering_set = images_rdd.filter(lambda x: x[1]%2 != 0) \
        .map(lambda x: x[0])

    features = training_set.flatMap(lambda x: x[1])
    """

    features = images_rdd.flatMap(lambda x: x[1])

    model = KMeans.train(features, 3, maxIterations=5, initializationMode="random")

    clusterCenters = model.clusterCenters

    build_end_time = time() - build_start_time

    processing_start_time = time()

    data_to_cluster = images_rdd.map(lambda x: [x, clusterCenters])

    features_bow = data_to_cluster.map(assign_pooling)
    features_bow.coalesce(1, shuffle=True).saveAsTextFile("hdfs://discus-p2irc-master:54310/tmp/output_image/")

    processing_end_time = time() - processing_start_time
    application_end_time = time() - application_start_time

    sc.stop()

    print("---------------------------------------------")
    print("SUCCESS: Model built in {} seconds".format(round(build_end_time, 3)))
    print("SUCCESS: Images processed in {} seconds".format(round(processing_end_time, 3)))
    print("SUCCESS: Total time spent = {} seconds".format(round(application_end_time, 3)))
    print("---------------------------------------------")

