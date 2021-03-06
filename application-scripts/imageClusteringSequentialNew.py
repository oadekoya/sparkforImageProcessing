import re
import sys
import os
import io
#import errno
import cv2
import numpy as np
#import functools
from skimage import *
from skimage import color
from skimage.feature import blob_doh
from io import BytesIO
import base64
from time import time
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise
from scipy.misc import imread, imsave
import copy_reg
import types
import multiprocessing as mp

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)



class ImgCluster:


    def __init__(self, output_dir):
        #if not os.path.isdir(output_dir):
        #    os.makedirs(output_dir)
        self.output_dir = output_dir



    def loadImages(self, img_lists):
        imgs_with_name = []
        for name in img_lists:
            img = imread(BytesIO(open(name, 'rb').read()))
            imgs_with_name.append((name, img))
        return imgs_with_name


    def estimate_feature(self, img):
        extractor = cv2.ORB_create()

        return extractor.detectAndCompute(img_as_ubyte(img), None)


    def convert(self, img):

        imgaes = color.rgb2gray(img)
        return imgaes


    def commonTransform(self, datapack):
        def common_transform(datapack):
            fname, imgaes = datapack
            procsd_obj = ''
            try:
                procsd_obj = self.convert(imgaes)
            except Exception as e:
                pass

            return (fname, imgaes, procsd_obj)

        return common_transform(datapack)


    def commonEstimate(self, datapack):
        def common_estimate(datapack):

            fname, img, procsd_obj = datapack
            try:
                kp, descriptors = self.estimate_feature(procsd_obj)
            except Exception as e:
                descriptors = None
                pass
            return (fname, img, descriptors)

        return common_estimate(datapack)


    def commonModel(self, params):
        features, K, maxIterations = params

        vec = []
        for f_object in features:

            vec.append(f_object[0].tolist())
        #print(len(vec))
        converted_vec = np.array(vec)
        model = KMeans(init='k-means++', n_clusters=K, max_iter = maxIterations).fit(converted_vec)
        #print(model.labels_)
        #model.predict([67,56])
        #center  = model.cluster_centers_
        #model = KMeans.train(features, K, maxIterations, initializationMode="random")
        return model


    def commonAnalysisTransform(self, datapack, params):
        def common_transform(datapack):
            fname, img, procsdentity = datapack
            model,condition = params
            feature_matrix = np.array(procsdentity[0].tolist())
            clusterCenters = model.cluster_centers_

            #model = KMeansModel(clusterCenters)
            measure = np.zeros(len(clusterCenters))
            k = model.predict(feature_matrix)
            #print(k)
            # for x in feature_matrix:
            #     k = model.predict(x)
            #     dist = pairwise.euclidean_distances(clusterCenters[k], x)
            #
            #     #dist = distance.euclidean(clusterCenters[k], x)
            #     if condition == "max":
            #         measure[k] = max(measure[k], dist)
            #     elif condition == "sum":
            #         measure[k] = measure[k] + dist
            # clusters = measure.tolist()
            # group = clusters.index(min(clusters)) + 1
            return (fname, 'none', k[0] + 1)

        return common_transform(datapack)


    def saveClusterResult(self,result):
         dirs = set()
         dirs_list = []
         dirs_dict = dict()
         clusters = result
         for lstr in clusters:

             group = lstr[2]
             img_name = lstr[0]
             if (group in dirs):
                 exists_list = dirs_dict[group]
                 exists_list.append(img_name)
                 dirs_dict.update({group: exists_list})
             else:
                 dirs_list = []
                 dirs_list.append(img_name)
                 dirs.add(group)
                 dirs_dict.update({group: dirs_list})

         for itms in dirs_dict.items():
             output = io.StringIO()
             for itm in itms[1]:
                 #print("img name: {} | group: {}".format(itm, itms [0]))
                 output.write(unicode(itm + "\n", "utf-8"))

             f =open(self.output_dir + str(itms[0]) + ".txt", 'wb')

             f.write(output.getvalue())
             f.close()


    def img_clustering(self, input_imgs_list):
        K= 3
        itrns = 5
        #imgs_list = list_files("/data/home/ooa712/images/15072016")
        imgs_with_name = self.loadImages(input_imgs_list)
        packs = []
        #processing_start_time = time()
        for bundle in imgs_with_name:
            packs.append(self.commonTransform(bundle))
        #print(packs)
        est_packs = []
        ii = 0
        features = []
        for pack in packs:
            #print(ii)
            if(len(pack[2].shape) !=0):
                fname, img, descriptor = self.commonEstimate(pack)
                features.append(descriptor)
                est_packs.append( (fname, img, descriptor))
            ii = ii+1

        model = self.commonModel((features,K,itrns))
        result = []
        for pack in est_packs:
            result.append(self.commonAnalysisTransform(pack, (model, 'sum')))

        return self.saveClusterResult(result)


def get_images_paths(images_path):
    imageFiles = []
    for name in os.listdir(images_path):
        if not name.endswith(".jpg"):
            continue
        tmpFilePath = os.path.join(images_path, name)
        if os.path.isfile(tmpFilePath):
            imageFiles.append(tmpFilePath)
    return imageFiles


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]



def mp_handler(image_filenames, output_dir, n_per_proc):
    imgCluster = ImgCluster(output_dir)
    pool = mp.Pool()
    pool.map(imgCluster.img_clustering, chunks(image_filenames, n_per_proc))
    #print pool.map(output, chunks(image_filenames, n_per_proc))


if(__name__=="__main__"):

    processing_start_time = time()

    input_files = get_images_paths("/data/15072016")

    #print(input_files)

    #print(list(chunks(input_files, 2)))
   
    mp_handler(input_files, "/data/imgclustering_result", 3)

    #pool.map(img_clustering, image_filenames,1)
    processing_end_time = time() - processing_start_time
    print("SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3)))
