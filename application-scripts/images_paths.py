import os
from os import listdir
from os.path import isfile, join
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





#paths = []
#mypath = "/data/mounted_hdfs_path/user/hduser/plot_images/2016-07-05_1207"
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#for file in onlyfiles:
#    path = os.path.join(mypath, file)
#    paths.append(path)


#print paths
#print onlyfiles


def get_images_paths(images_path):
    imagesPaths = []
    for dir in os.listdir(images_path):
        #new_dir = os.path.join(images_path, dir)
        #if os.path.isdir(dir)
        imagesPaths.append(os.path.join(images_path, dir))
    return imagesPaths


mypath = "/data/mounted_hdfs_path/user/hduser/plot_images/2016-07-05_1207"
#mypath = "/data/mounted_hdfs_path/user/hduser/plot_images/2016-07-05_1207"



def list_files(images_path):
    """
    returns a list of names (with extension, without full path) of all files
    in folder path
    """
    imageFiles = []
    for filename in os.listdir(images_path):
        if filename.endswith(".jpg"):
        #if os.path.isfile(os.path.join(images_path, filename)):
            imageFiles.append(filename)
    return imageFiles

paths = get_images_paths(mypath)
print (paths)
#directory = listdir(mypath)
#print(len(directory))

#images_files = list_files(mypath)
#print(len(images_files))


#mypath = "/data/mounted_hdfs_path/user/hduser/plot_images/2016-07-05_1207"
#onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#print(len(onlyfiles))




#images_files = list_files("/data/mounted_hdfs_path/user/hduser/plot_images/")
#images_paths = get_images_paths("/data/mounted_hdfs_path/user/hduser/plot_images/")

#job_name="flowerCount"

#print("hdfs://discus-p2irc-master:54310/tmp/output/" + job_name)

#print images_paths
