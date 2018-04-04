"""
Canola flower detector

This file contains the methods to detect canola flowers
from Brinno camera images.

Author: Javier Garcia
"""
# Image processing / Math
import cv2
import os
import numpy as np
import PIL.Image
import PIL.ImageFile
import glob
import operator
import matplotlib.pyplot as plt
from time import time

import multiprocessing as mp
# K-Means
from sklearn.cluster import KMeans

# Flower counting
from skimage.feature import blob_doh

# Progress bar
from tqdm import tqdm
import sys

# Timestamp
import pytz
import datetime
from plotdetection import userDefinePlot


class FlowerCounter(object):
    """
    Class to detect canola flowers
    """
    def __init__(self, imgspath = None, imgsformat = "jpg"):
        """
        Initializer
        :param imgspath: Path containing the images to analyze (by default: None)
        :param imgsformat: Format of the image (string) (by default: jpg)
        """
        #: Plot mask
        self.plot_mask = []
        #: Plot bounds
        self.plot_bounds = []
        #: Mask of the area where flowers will be counted
        self.flower_area_mask = []
        #: Bounds of the area where flowers will be counted
        self.flower_area_bounds = []
        #: Average histogram of B component
        self.avg_hist_b = []
        #: Histograms of the b (Lab) channel for all images (only for pixels within the plot)
        self.hist_b_all = {}
        #: Histograms of the Grayscale images
        self.hist_G_all = {}
        #: Histogram shift for component B of each image
        self.hist_b_shifts = {}
        #: Dictionary storing the fraction of flower pixels in each image
        self.flower_pix_perc = {}
        #: Estimation of flower pixels percentage for all images
        self.flower_pix_perc_estimate = {}
        #: Count of number of flowers per image
        self.flower_count = {}
        #: Estimate of number of flowers per image
        self.flower_count_estimate = {}
        #: Result from K-means clustering
        self.img_clusters = {}
        #: Image size
        self.imsize = (720, 1280)
        self.region_matrix = [[0,0,0],[0,1,0],[0,0,0]]
        #: Images path
        self.imgspath = None
        #: Images filenames
        self.imfilenames = []
        #: Filenames of the best images
        self.best_imfnames = []
        #: Images timestamps
        self.imgs_timestamps = {}
        #: Not valid images
        self.notValidImages = []

        # User parameters
        #: Upper limit of B channel for segmentation
        self.segm_B_upper = 255
        #: Bandwidth used for segmentation on the B channel
        self.segm_B_lower = 155
        #: Threshold for percentage of flower pixels to consider that there are flowers in the field
        self.flower_pix_perc_thresh = 0.0001

        # Set images path
        if imgspath is not None:
            self.setImagesFilepaths(path=imgspath, imgsformat=imgsformat)

            # Set default plot bounds & mask
            #self.setPlotMask(np.ones(self.imsize),
            #                 [(0, 0), (0, self.imsize[1]), (self.imsize[0], 0), self.imsize])

    #####################################################################
    ##                        SET / RESET FUNCS                        ##
    #####################################################################

    def resetDefaultParams(self):
        """
        Resets all object's variables
        """
        #: Average histogram of B component
        self.avg_hist_b = []
        #: Histograms of the b (Lab) channel for all images
        self.hist_b_all = {}
        #: Histograms of the Grayscale images
        self.hist_G_all = {}
        #: Histogram shift for component B of each image
        self.hist_b_shifts = {}
        #: Dictionary storing the fraction of flower pixels in each image
        self.flower_pix_perc = {}
        #: Estimation of flower pixels percentage for all images
        self.flower_pix_perc_estimate = {}
        #: Count of number of flowers per image
        self.flower_count = {}
        #: Estimate of number of flowers per image
        self.flower_count_estimate = {}
        #: Filenames of the best images
        self.best_imfnames = []
        #: Not valid images
        self.notValidImages = []

    def resetImageClusters(self):
        """
        Sets all images to cluster 0 (i.e. good images)
        """
        # Initial assert
        assert len(self.imfilenames) > 0, "Failed to reset image cluster: No image names loaded"

        # Assign all images to cluster 0
        for imname in self.imfilenames:
            self.img_clusters[imname] = 0

    def setImagesFilepaths(self, imfilenames = None, path = None, imgsformat = "jpg"):
        """
        Set a new list of filenames that will be analyzed by this flower detector. If both parameters are none,
        the function will return without doing anything
        :param imfilenames: List of the image filepaths
        :param path: Path containing the images
        :param imgsformat: Format of the image (string) (by default: jpg)
        """
        if imfilenames is not None:
            assert isinstance(imfilenames, list), "Parameter imfilepaths must be a list of strings"
            assert len(imfilenames) > 0, "Parameter imfilepaths is empty"
            self.imfilenames = imfilenames

        elif path is not None:
            assert os.path.exists(path), "Selected path does not exist"
            assert isinstance(imgsformat, str), "Images format must be a string (e.g. 'jpg', 'tiff', etc)"

            if not path.endswith('/'):
                path += '/'

            self.imgspath = path

            imfilepaths = glob.glob(path + "*." + imgsformat)
            assert len(imfilepaths) > 0, "No JPG images in the selected path"

            # Sort images filepaths
            imfilepaths = [int(k.split("/")[-1].split(".")[0]) for k in imfilepaths]
            imfilepaths.sort()

            # Store images filepaths
            self.imfilenames = [str(f) + "." + imgsformat for f in imfilepaths]

            # Get image size
            PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
            pil_image = PIL.Image.open(path + self.imfilenames[0]).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            self.imsize = open_cv_image.shape[:2]

        else:
            print("At least one parameter must be not None. Leaving without modifying...")
            return

        # Set all images initially as good
        self.resetImageClusters()

    def setImagesTimestamps(self, imgs_timestamps):
        """
        Set images timestamps extracted from database
        :param imgs_timestamps: Dictionary with the image filenames as keys and the timestamps as values
        """
        # Initial assert
        assert isinstance(imgs_timestamps, dict), "imgs_timestamps must be a dictionary"

        # Sort keys of timestamps dictionary
        keys = [int(k.split(".")[0]) for k in imgs_timestamps.keys()]
        keys.sort()
        keys = [str(k) + ".jpg" for k in keys]

        # Reset timestamps
        self.imgs_timestamps = {}

        # Move timestamps one by one instead of all together so we check that filenames match
        for i in range(len(keys)):
            # Get key and value
            k = keys[i]
            v = imgs_timestamps[k]

            # Estimate timestamp if None
            if v is None:
                imgs_timestamps[k] = imgs_timestamps[keys[i-1]] + datetime.timedelta(minutes=1)

            # Assert
            if not self.imfilenames.__contains__(k):
                print("Image " + str(k) + " not included in the object's images filenames")
                continue

            # Add timestamp
            self.imgs_timestamps[k] = imgs_timestamps[k]

    def setUserParameters(self, segm_B_upper = None, segm_B_lower = None, flower_pix_perc_thresh = None):
        """
        Allow to modify some of the user parameters
        :param segm_B_upper: Upper B value used for segmentation
        :param segm_B_lower: Lower B value used for segmentation
        :param flower_pix_perc_thresh: Threshold for percentage of flower pixels to consider that there are flowers in the field
        """
        if segm_B_upper is not None:
            assert isinstance(segm_B_upper, int) and segm_B_upper >=0 and segm_B_upper <= 255, "Segm_B_upper must be an integer between 0 and 255"
            self.segm_B_upper = segm_B_upper
            self.computeAdditionalAvgHistParams()

        if segm_B_lower is not None:
            assert isinstance(segm_B_lower, int) and segm_B_lower >=0 and segm_B_lower <= 255, "Segm_B_lower must be an integer between 0 and 255"
            self.segm_B_lower = segm_B_lower
            self.computeAdditionalAvgHistParams()

        if flower_pix_perc_thresh is not None:
            assert flower_pix_perc_thresh >=0 and flower_pix_perc_thresh <= 1, "flower_pix_perc_thresh must be a real number between 0 and 1"
            self.flower_pix_perc_thresh = flower_pix_perc_thresh

    def setPlotMask(self, bounds, mask = None):
        """
        Set mask of the plot under analysis
        :param mask: Mask of the plot
        :param bounds: Bounds of the plot
        """

        #global plot_mask_accum, plot_bounds_accum
        # global plot_mask, plot_bounds

        #print(type(mask))
        # Initial assert
        if mask is not None:
            assert isinstance(mask, np.ndarray), "Parameter 'corners' must be Numpy array"
            assert mask.shape == self.imsize, "Mask has a different size"
        assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"


        # Store bounds
        self.plot_bounds = bounds
        #plot_bounds_accum.add(bounds)

        # Store mask
        if mask is None:
            _, self.plot_mask = ((userDefinePlot(np.zeros(self.imsize), bounds)))

        else:
            self.plot_mask = mask
            #plot_mask_accum.add(mask)

        #plot_mask_accum = np.array(plot_mask_accum.value)
        # return plot_mask, plot_bounds

    def setFlowerAreaMask(self, region_matrix, mask):
        """
        Set mask of the flower area within the plot
        :param region_matrix = Region matrix representing the flower area
        :param mask: Mask of the flower area
        """

        # global flower_area_mask
        #global flower_area_mask_accum
        # Initial assert
        if mask is not None:
            assert isinstance(mask, np.ndarray), "Parameter 'mask' must be Numpy array"
            assert mask.shape == self.imsize, "Mask has a different size"

        #assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"

        # Store bounds
        self.flower_region_matrix = region_matrix

        # Store mask
        #flower_area_mask_accum.add(mask)
        self.flower_area_mask = mask

        # return flower_region_matrix, flower_area_mask

    #####################################################################
    ##                              TOOLS                              ##
    #####################################################################

    def getFlowerMask(self, img, fname):
        """
        Take the image in LAB colorspace with the histograms shifted, create mask that identifies flowers
        :param img: Image array (BGR)
        :param fname: Filename of the image (needed to extract the B histogram)
        :return: Mask to segment flowers
        """
        # Initial assert
        if not isinstance(img, np.ndarray):
            print("Parameter 'img_array' must be a numpy array returned by applyHistogramShift()")
            return

        # Shift to LAB colorspace
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # Get inRange array for LAB
        img_mask = cv2.inRange(img_lab,
                               (0, 0, int(self.segm_B_lower + self.hist_b_shifts[fname])),
                               (255, 255, int(self.segm_B_upper + self.hist_b_shifts[fname])))

        # TODO: Get rid of the blue&orange posts!

        # Apply dilation
        # img_mask = cv2.dilate(img_mask, np.ones((3,3)))

        # Apply plot mask
        img_mask[self.plot_mask == 0] = 0

        return img_mask

    def getFlowerHighlight(self, img, fname, segm_out_value = 0.99, segm_dist_from_zerocross = 5):
        """
        Take the image given as parameter and highlight flowers applying a logistic function
        on the B channel. The formula applied is f(x) = 1/(1 + exp(K * (x - T))) being K and T constants
        calculated based on the given parameters.
        :param img: Image array
        :param fname: Image filename
        :param segm_out_value: Value of the logistic function output when the input is the lower B segmentation value i.e. f(S), where S = self.segm_B_lower + self.hist_b_shifts[fname]
        :param segm_dist_from_zerocross: Value that, when substracted from the lower B segmentation value, the output is 0.5 i.e. Value P where f(self.segm_B_lower + self.hist_b_shifts[fname] - P) = 0.5
        :return: Grayscale image highlighting flower pixels (pixels values between 0 and 1)
        """
        # Initial assert
        assert isinstance(img, np.ndarray), "img parameter must be a Numpy array"
        assert img.shape[:2] == self.imsize, "img is not of the same size as this object's"
        assert isinstance(fname, str), "fname must be a string"
        assert self.hist_b_shifts.__contains__(fname), "fname does not exist on this object's scope"
        assert isinstance(segm_out_value, (float, int)), "semg_out_value must be a number"
        assert max(0, min(1, segm_out_value)) == segm_out_value, "segm_out_value must be between 0 and 1"
        assert isinstance(segm_dist_from_zerocross, int), "segm_dist_from_zerocross must be an integer"
        assert segm_dist_from_zerocross > 0, "segm_dist_from_zerocross must be positive"

        # Convert to LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # Get the B channel and convert to float
        img_B = np.array(img_lab[:, :, 2], dtype=np.float32)

        # Get the parameter T for the formula
        t_exp = self.segm_B_lower + self.hist_b_shifts[fname] - segm_dist_from_zerocross

        # Get the parameter K for the formula
        k_exp = np.log(1 / segm_out_value - 1) / segm_dist_from_zerocross

        # Apply logistic transformation
        img_B = 1 / (1 + np.exp(k_exp * (img_B - t_exp)))

        return img_B

    def splitImagesInPeriods(self, periodLimits):
        """
        Splits the images on a day in periods.
        :param periodLimits: List including strings representing the times when a period ends and a new one starts
        :return: List containing lists of images divided in periods
        """
        # Initial assert
        assert isinstance(periodLimits, list), "periodLimits must be a list containing strings. Format: HH:MM"
        assert len(periodLimits) > 0, "periodLimits is empty"

        # Initialize periods of the day
        images_periods = []

        # Get timestamps for only existing pictures
        timestamps = {}
        for k, v in self.imgs_timestamps.items():
            if self.imfilenames.__contains__(k):
                timestamps[k] = v

        # Get timestamp of reference
        timestamp_reference = timestamps[self.imfilenames[0]]

        # Append the last second of the day as the last period limit
        periodLimits = periodLimits + ["23:59"]

        # Get images for each period limit
        for p in range(len(periodLimits)):
            timestamp_highlimit = datetime.datetime.combine(timestamp_reference.date(),
                                                            datetime.datetime.strptime(periodLimits[p],
                                                                                       '%H:%M').time())

            if p == 0:
                timestamp_lowlimit = datetime.datetime.combine(timestamp_reference.date(),
                                                               datetime.datetime.strptime("00:00", '%H:%M').time())
            else:
                timestamp_lowlimit = datetime.datetime.combine(timestamp_reference.date(),
                                                               datetime.datetime.strptime(periodLimits[p - 1],
                                                                                          '%H:%M').time())

            # Shift to UTC
            local = pytz.timezone("America/Regina")
            local_dt = local.localize(timestamp_highlimit, is_dst=None)
            timestamp_highlimit = local_dt.astimezone(pytz.utc)
            local_dt = local.localize(timestamp_lowlimit, is_dst=None)
            timestamp_lowlimit = local_dt.astimezone(pytz.utc)

            lower_timestamps = [t for t in range(len(self.imfilenames)) if
                                timestamps[self.imfilenames[t]] is not None and timestamps[self.imfilenames[t]] >= timestamp_lowlimit]
            higher_timestamps = [t for t in range(len(self.imfilenames)) if
                                 timestamps[self.imfilenames[t]] is not None and timestamps[
                                     self.imfilenames[t]] < timestamp_highlimit]

            # Check if previous sets are not empty
            if len(lower_timestamps) == 0 or len(higher_timestamps) == 0:
                images_periods.append([])
                continue

            # Get the filenames
            lower_ID = min(lower_timestamps)
            higher_ID = max(higher_timestamps)

            # Append new period
            images_periods.append(self.imfilenames[lower_ID:higher_ID + 1])

        return images_periods

    def removeNotValidImagesFromAttributes(self):
        """
        Erases from all attributes the invalid images
        """
        for fname_notvalid in self.notValidImages:
            # Erase from images filepaths
            if self.imfilenames.__contains__(fname_notvalid):
                self.imfilenames.remove(fname_notvalid)

            # Erase from timestamps
            _ = self.imgs_timestamps.pop(fname_notvalid, None)

            # Erase from histograms & histograms shift
            _ = self.hist_b_all.pop(fname_notvalid, None)
            _ = self.hist_b_shifts.pop(fname_notvalid, None)

            # Erase from flower pixels percentage (real and estimate)
            _ = self.flower_pix_perc.pop(fname_notvalid, None)
            _ = self.flower_pix_perc_estimate.pop(fname_notvalid, None)

            # Erase from flower count
            _ = self.flower_count.pop(fname_notvalid, None)
            _ = self.flower_count_estimate.pop(fname_notvalid, None)

            # Erase from clustering
            _ = self.img_clusters.pop(fname_notvalid, None)

    def kmeans(self, km_points_orig, nClusters=4):
        """
        Easy implementation of k-means clustering algorithm
        :param km_points_orig: Dictionary with points to be clustered
        :param nClusters: Number of clusters (by default: 3)
        """
        # Initial assert
        assert isinstance(km_points_orig, dict), "Points passed as parameter must be a dictionary with string keys and numeric values"
        assert isinstance(nClusters, int) and nClusters > 1, "Number of clusters must be an integer higher than 1"

        # Create KMeans object
        km = KMeans(n_clusters=nClusters)

        # Get the ordered set of points (i.e. flower pixel percentages of each image)
        km_points = np.array([k[1] for k in sorted(km_points_orig.items(), key=operator.itemgetter(1))]).reshape((-1, 1))

        # Compute KMeans
        km.fit(km_points)

        # Get the centroids ordered
        km_centroids = list(km.cluster_centers_)
        km_centroids.sort()

        # Assign each image to a cluster
        final_img_clusters = {}
        for k, v in km_points_orig.items():
            # Compute distance to each of the centroids
            dist = np.array([abs(v - q) for q in km_centroids])

            # Get the closest centroid
            final_img_clusters[k] = int(dist.argmin())

        return final_img_clusters

    def clusterImages(self):
        """
        Method to flag images into good and bad
        """
        # Eliminate missing pictures
        fw_pix_perc = {}
        for k, v in self.flower_pix_perc.items():
            if not self.notValidImages.__contains__(k):
                fw_pix_perc[k] = v

        # Apply K-means to the percentage of flower pixels in the images
        kmeans_fwperc = self.kmeans(fw_pix_perc, nClusters=3)

        kmeans_copy = {}
        for k, v in kmeans_fwperc.items():
            if v == 0:
                kmeans_copy[k] = fw_pix_perc[k]
            kmeans_fwperc[k] = v+1

        # Second K-means: Divides the cluster 0 into two clusters
        kmeans_fwperc_2 = self.kmeans(kmeans_copy, nClusters=2)

        for k, v in kmeans_fwperc.items():
            if v == 1:
                self.img_clusters[k] = kmeans_fwperc_2[k]
            else:
                self.img_clusters[k] = v

    #####################################################################
    ##                        COMPUTE FUNCTIONS                        ##
    #####################################################################

    def computePlotMask(self):
        """
        Compute plot mask
        """
        # Trace
        print("Computing plot mask...")

        # Read an image
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        pil_image = PIL.Image.open(self.imgspath + self.imfilenames[0]).convert('RGB')
        open_cv_image = np.asarray(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Ask for user interaction to get the plot
        p_bounds, p_mask = userDefinePlot(open_cv_image, None)

        # Store mask and bounds
        self.setPlotMask(p_bounds, p_mask)



    def computeFlowerAreaMask(self, region_matrix):
        """
        Compute the flower area mask based on a matrix that indicates which regions of the plot are part of the
        flower counting.
        :param region_matrix: Mmatrix reflecting which zones are within the flower area mask (e.g. in order to
        sample the center region, the matrix should be [[0,0,0],[0,1,0],[0,0,0]]
        """

        # Trace
        print("Computing flower area mask...")

        #global plot_bounds_accum
        # global plot_bounds

        # Check for plot bounds
        assert len(self.plot_bounds) > 0, "Plot bounds not set. Please set plot bounds before setting flower area mask"

        #region_matrix = region_matrix.value
        # Convert to NumPy array if needed
        if not isinstance(region_matrix, np.ndarray):
            region_matrix = np.array(region_matrix)

        # Assert
        assert region_matrix.ndim == 2, 'region_matrix must be a 2D matrix'

        # Get the number of rows and columns in the region matrix
        rows, cols = region_matrix.shape

        # Get transformation matrix
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]]), np.float32(self.plot_bounds))

        # Initialize flower area mask
        fw_mask = np.zeros(self.imsize)

        # Go over the flower area mask and turn to 1 the marked areas in the region_matrix
        for x in range(cols):
            for y in range(rows):
                # Write a 1 if the correspondant element in the region matrix is 1
                if region_matrix[y,x] == 1:
                    # Get boundaries of this zone as a float32 NumPy array
                    bounds = np.float32([[x,y], [x+1,y], [x,y+1], [x+1,y+1]])
                    bounds = np.array([bounds])

                    # Transform points
                    bounds_T = cv2.perspectiveTransform(bounds, M)[0].astype(np.int)

                    # Get mask for this area
                    _, mask = userDefinePlot(fw_mask, list(bounds_T))

                    # Apply mask
                    fw_mask[mask > 0] = 255

        # Save flower area mask & bounds
        self.setFlowerAreaMask(region_matrix, fw_mask)


    def computeHistograms(self):
        """
        Compute the average A and B histograms over all images
        """
        # Trace
        print("Computing histograms...")

        # Clear average histogram
        self.avg_hist_b = np.zeros(256, np.int32)

        # Clear histograms for b channel
        self.hist_b_all = {}

        # Get the number of images and store it as number of samples
        nSamples = len(self.imfilenames)

        # Preliminary check
        if nSamples == 0:
            print("No images found")
            return

        # Iterate through all images
        for fname in tqdm(self.imfilenames, file=sys.stdout):
            # Read image
            PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
            pil_image = PIL.Image.open(self.imgspath + fname).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            im_bgr = open_cv_image[:, :, ::-1].copy()

            # Shift to grayscale
            im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

            # Shift to LAB
            im_lab_plot = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2Lab)

            # Keep only plot pixels
            im_gray = im_gray[self.plot_mask > 0]
            im_lab_plot = im_lab_plot[self.plot_mask > 0]

            # Get histogram of grayscale image
            hist_G, _ = np.histogram(im_gray, 256, [0, 256])

            # Get histogram of B component
            hist_b, _ = np.histogram(im_lab_plot[:, 2], 256, [0, 256])

            # Save histograms
            self.hist_b_all[fname] = hist_b
            self.hist_G_all[fname] = hist_G

    def computeAverageHistograms(self):
        """
        Compute average B histogram
        """
        print("Computing average histogram..")

        # global hist_b_all, avg_hist_b
        try:

            # Vertically stack all the B histograms
            avg_hist_B = np.vstack(tuple([h for h in self.hist_b_all.values()]))

            # Sum all columns
            avg_hist_B = np.sum(avg_hist_B, axis=0)

            # Divide by the number of images and store
            self.avg_hist_b = np.divide(avg_hist_B, len(self.hist_b_all))

            #print(avg_hist_b)
            #return avg_hist_b

        except (ValueError) as e:
            pass

 
    def computeHistogramShifts(self):
        """
        Compute histogram shifts respect to the average histograms for each image
        """
        # Trace
        print("Computing histogram shifts...")

        # Initialize dictionaries to be saved containing the histograms shifts
        self.hist_b_shifts = {}

        # Iterate through images
        for fname in tqdm(self.imfilenames, file=sys.stdout):
            # Calculate correlation
            correlation_b = np.correlate(self.hist_b_all[fname], self.avg_hist_b, "full")

            # Get the shift on the X axis
            x_shift_b = correlation_b.argmax().astype(np.int8)

            # Append it to the list
            self.hist_b_shifts[fname] = x_shift_b

    def computeFlowerPixelsPercentage(self):
        """
        Compute the percentage of flower pixels in all of the pictures.
        Since we only apply thresholding in the B channel to get the flower pixels,
        we only need the B histogram to get the number of flower pixels in the plot.
        """
        # Trace
        print("Computing flower pixels percentage...")

        # Get number of pixels in the plot
        n_plot_pix = len(self.plot_mask[self.plot_mask > 0])

        # Iterate through all images
        for fname in tqdm(self.imfilenames, file=sys.stdout):
            # Get segmentation parameters
            hist_b = self.hist_b_all[fname]
            hist_b_shift = self.hist_b_shifts[fname]

            # Calculate number of flower pixels
            n_flower_pixels = np.sum(hist_b[self.segm_B_lower + hist_b_shift: self.segm_B_upper + hist_b_shift + 1])

            # Get percentage of flower pixels
            fw_pix_perc = n_flower_pixels / n_plot_pix

            # Store it in dictionary
            self.flower_pix_perc[fname] = fw_pix_perc

        # Copy dictionary
        self.flower_pix_perc_estimate = self.flower_pix_perc

    def computeBestImages(self, perc = 0.10):
        """
        Extracts which are the best evaluated images of the day. This is based on the assumption that
        false positives only increase the number of flower pixels in the images, therefore the best images
        will be those that have less number of flower pixels
        :param perc: Percentage of total number of images (by default: 10%)
        """
        # Initial assert
        assert isinstance(perc, float), "Parameter perc must be a float number"
        assert perc > 0 and perc <= 1, "Parameter perc must be between 0 and 1"

        # Trace
        print("Compute best images...")

        # Get number of images that we will consider as the best ones
        n_best_images = int(len(self.imfilenames) * perc)

        # Get ordered list of images filenames with their flower pixels percentage
        fwpix_list = sorted(self.flower_pix_perc.items(), key=operator.itemgetter(1))

        # Get best images
        candidates_best_imfnames = [k[0] for k in fwpix_list[:n_best_images]]

        # Combine the candidates for best images with the images from cluster 0
        self.best_imfnames = [k for k in candidates_best_imfnames if self.img_clusters[k] == 0]

    def computeFlowerCount(self):
        """
        Get flower count for the best images
        """
        # Trace
        print("Computing number of flowers...")

        # Initialize dictionary to store number of flowers per image
        n_flowers = {}

        # Get first if there are any flowers in the field. If there aren't any, leave before starting calculations
        if not self.getFlowersPresent():
            # Trace
            print("No flowers present")

            # Set all best images with number of flowers equal to zero
            n_flowers = dict([(t, 0) for t in self.best_imfnames])

        else:
            # Get pairs of image filename and flower pixels percentage for this period
            images_best = [(t, self.flower_pix_perc[t]) for t in self.best_imfnames]

            # Iterate through the selected images
            for fname, fw_pix_perc in tqdm(images_best, file=sys.stdout):

                # Get complete image filepath
                img_filename = self.imgspath + fname

                # Get flower mask for this image
                PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
                pil_image = PIL.Image.open(img_filename).convert('RGB')
                open_cv_image = np.array(pil_image)
                img = open_cv_image[:, :, ::-1].copy()

                # Highlight flowers
                img_flowers = self.getFlowerHighlight(img, fname, segm_dist_from_zerocross=8)

                # Apply flower area mask
                img_flowers[self.flower_area_mask == 0] = 0

                # Get number of flowers using blob counter on the B channel
                blobs = blob_doh(img_flowers, max_sigma=5, min_sigma=1)

                # Append result
                n_flowers[fname] = blobs

        # Save to object
        self.flower_count = n_flowers
        self.flower_count_estimate = n_flowers

    def computeNotValidImages(self):
        """
        Determines which images are invalid (i.e. corrupted)
        """
        # Trace
        print("Computing non-valid images...")

        # Initialize list
        self.notValidImages = []

        # Get threshold (i.e. 20% of number of pixels in image)
        n_pixels_thresh = 0.2 * np.multiply(*self.imsize)

        # Iterate through all the image filepaths
        for fname in tqdm(self.imfilenames, file=sys.stdout):
            # Check if maximum value is above 20% of the total number of pixels
            if max(self.hist_G_all[fname]) >= n_pixels_thresh:
                print(fname + " is not valid")
                self.notValidImages.append(fname)

    #####################################################################
    ##                       SAVE / LOAD FUNCTIONS                     ##
    #####################################################################

    def loadTimestamps(self, path = None, timestamps_name = "timestamps.npy"):
        """
        Load timestamps saved as a Numpy file
        :param path: Path to the timestamp file (by default: Same as images' path)
        :param timestamps_name: (Optional) Filename of the timestamps (by default: "timestamps.npy"_
        :return: 1 if load successful. 0 otherwise
        """
        # Path assignment
        if path is None:
            path = self.imgspath

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        # Check if histograms files exist
        if os.path.exists(path + timestamps_name):
            # Load average positions
            d = np.load(path + timestamps_name).item()

            # Set timestamps
            self.setImagesTimestamps(d)
            return 1

        else:
            print("Timestamps file not found. Leaving without loading...")
        return 0

    def saveHistograms(self, path = None, histograms_name = "histograms"):
        """
        Save histograms calculated by computeHistograms as Numpy files
        :param path: Path where the histograms files will be saved (by default: Same as images' path)
        :param histograms_name: (Optional) Filename of the histograms without extension (by default: "histograms")
        """
        # Path assignment
        if path is None:
            path = self.imgspath

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"

        # Check
        if len(self.avg_hist_b) == 0:
            print("Average histogram is empty. Leaving without saving...")
            return

        if len(self.hist_b_all) == 0:
            print("b histograms are empty. Leaving without saving...")
            return

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        # Create dictionary to store
        dict_to_save = {"hist_G_all": self.hist_G_all,
                        "hist_b_all": self.hist_b_all,
                        "avg_hist_b": self.avg_hist_b,
                        "plot_bounds" : self.plot_bounds}

        # Save/Overwrite dictionary
        np.save(path + histograms_name, dict_to_save)

    def loadHistograms(self, path = None, histograms_name = "histograms.npy"):
        """
        Load histograms previously saved as numpy files
        :param path: Path to the histograms files (by default: Same as images' path)
        :param histograms_name: (Optional) Filename of the histograms (by default: "histograms.npy")
        :return: 1 if load successful. 0 otherwise
        """
        # Path assignment
        if path is None:
            path = self.imgspath

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        # Check if histograms files exist
        if os.path.exists(path + histograms_name):
            # Load average positions
            d = np.load(path + histograms_name).item()

            try:
                # Check if segmentation parameters have not changed
                if self.plot_bounds == d["plot_bounds"]:
                    self.hist_G_all = d["hist_G_all"]
                    self.hist_b_all = d["hist_b_all"]
                    self.avg_hist_b = d["avg_hist_b"]
                    return 1
            except KeyError:
                print("One or more variable/s missing from Average histograms data. Leaving without loading...")
                return 0
        else:
            print("Histograms file not found. Leaving without loading...")

        return 0

    def saveFlowerCount(self, path = None, flower_count_name = "flower_count"):
        """
        Save number of flowers in each image calculated by getFlowerCount_bestOfX as Numpy files
        :param path: Path where the flower count files will be saved (by default: Same as images' path)
        :param flower_count_name: (Optional) Filename of the flower count (by default: "flower_count")
        """
        # Path assignment
        if path is None:
            path = self.imgspath

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"

        # Check
        if len(self.flower_count) == 0:
            print("Flower count is empty. Leaving without saving...")
            return

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        # Create dictionary to store
        dict_to_save = {"flower_count": self.flower_count,
                        "segm_B_upper": self.segm_B_upper,
                        "segm_B_lower": self.segm_B_lower,
                        "plot_bounds" : self.plot_bounds}

        # Save/Overwrite dictionary
        np.save(path + flower_count_name, dict_to_save)

    def loadFlowerCount(self, path = None, flower_count_name = "flower_count.npy"):
        """
        Load number of flowers in all images previously saved as numpy files
        :param path: Path to the flower count files (by default: Same as images' path)
        :param flower_pix_perc_name: (Optional) Filename of the flower count (by default: "flower_count.npy")
        :return: 1 if load successful. 0 otherwise
        """
        # Path assignment
        if path is None:
            path = self.imgspath

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        # Check if histograms exist
        if os.path.exists(path + flower_count_name):
            # Load average positions
            d = np.load(path + flower_count_name).item()

            try:
                # Check if segmentation parameters have not changed
                if self.segm_B_upper == d["segm_B_upper"] and self.segm_B_lower == d["segm_B_lower"] and self.plot_bounds == d["plot_bounds"]:
                    self.flower_count = d["flower_count"]
                    self.flower_count_estimate = d["flower_count"]
                    return 1
            except KeyError:
                print("One or more variable/s missing from Flower count data. Leaving without loading...")
                return 0
        else:
            print("Flower count files not found. Leaving without loading...")

        return 0

    def savePlotMask(self, path = None, plot_mask_name = "plot_mask"):
        """
        Save plot and flower area masks calculated by computePlotMask as Numpy files
        :param path: Path where the plot mask file will be saved (by default: Same as images' path)
        :param plot_mask_name: (Optional) Filename of the plot mask (by default: "plot_mask")
        """
        # Path assignment
        if path is None:
            path = self.imgspath

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        # Check if variables to save are not empty
        if len(self.plot_mask) == 0 or len(self.plot_bounds) == 0:
            print("Plot mask is empty. Leaving without saving...")
            return

        if len(self.flower_area_bounds) == 0 or len(self.flower_area_mask) == 0:
            print("Flower area mask is empty. Leaving without saving...")
            return

        # Create dictionary to store
        dict_to_save = {"plot_bounds": self.plot_bounds,
                        "plot_mask": self.plot_mask,
                        "flower_area_bounds" : self.flower_area_bounds,
                        "flower_area_mask" : self.flower_area_mask}

        # Save/Overwrite dictionary
        np.save(path + plot_mask_name, dict_to_save)

    def loadPlotMask(self, path = None, plot_mask_name = "plot_mask.npy"):
        """
        Load plot and flower area mask previously saved as numpy files
        :param path: Path to the plot mask file (by default: Same as images' path)
        :param plot_mask_name: (Optional) Filename of the plot mask (by default: "plot_mask.npy")
        :return: 1 if load successful. 0 otherwise
        """
        # Path assignment
        if path is None:
            path = self.imgspath

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        # Check if histograms exist
        if os.path.exists(path + plot_mask_name):
            # Load average positions
            d = np.load(path + plot_mask_name).item()

            try:
                # Save plot parameters
                plot_mask = d["plot_mask"]
                plot_bounds = d["plot_bounds"]
                flower_area_mask = d["flower_area_mask"]
                flower_area_bounds = d["flower_area_bounds"]

                # Set plot parameters
                self.setPlotMask(plot_mask, plot_bounds)
                self.setFlowerAreaMask(flower_area_mask, flower_area_bounds)
                return 1
            except KeyError:
                print("One or more variable/s missing from Plot mask data. Leaving without loading...")
                return 0
        else:
            print("Plot mask file not found. Leaving without loading...")

        return 0

    def loadFolder(self, plotMaskPath = None, path = None, saveIfNotExist = False):
        """
        Interact with folder to retrieve the data stored in it
        :param plotMaskPath: Path of the plot mask (usually different than the rest of the files)
        :param path: Path to the folder. Contains images and data from the FlowerDetector
        :param saveIfNotExist: Flag to indicate whether to save the data extracted if it was not in the folder (by default: False)
        """
        # Path assignment
        if path is None:
            path = self.imgspath

        if plotMaskPath is None:
            plotMaskPath = self.imgspath

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"
        assert os.path.exists(plotMaskPath), "Selected or default plot mask path does not exist"

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        if not plotMaskPath.endswith('/'):
            plotMaskPath += '/'

        # Reset all variables
        self.resetDefaultParams()
        self.resetImageClusters()

        # Load data stored as numpy arrays in the folder. Force the result to be 0 if previous item was not loaded successfully
        #self.loadTimestamps(path=path)
        loadPlotMask = self.loadPlotMask(path=plotMaskPath)
        loadHist = self.loadHistograms(path=path) * loadPlotMask
        loadFlowerCount = self.loadFlowerCount(path=path) * loadHist

        # Check if plot mask was loaded
        if loadPlotMask == 0:
            # Compute plot mask
            self.computePlotMask()

            # Save data if requested
            if saveIfNotExist:
                self.savePlotMask(path=plotMaskPath)

        # Check if the average histograms were loaded
        if loadHist == 0:
            # Compute histograms
            self.computeHistograms()

            # Save data if requested
            if saveIfNotExist:
                self.saveHistograms(path=path)

        # Compute non-valid images
        self.computeNotValidImages()

        # Compute histograms shifts
        self.computeHistogramShifts()

        # Compute flower pixels percentage
        self.computeFlowerPixelsPercentage()

        # Cluster images
        self.clusterImages()

        # Compute the best images
        self.computeBestImages()

        if loadFlowerCount == 0:
            # Compute flower count
            self.computeFlowerCount()

            # Save data if requested
            if saveIfNotExist:
                self.saveFlowerCount(path=path)

        # Remove corrupted images
        self.removeNotValidImagesFromAttributes()

        print("All objects loaded! Ready to go")

    #####################################################################
    ##                       ESTIMATE FUNCTIONS                        ##
    #####################################################################

    def estimateFlowerPixels(self):
        """
        Calculate the amount of flower pixels for flagged (i.e. bad) images by doing an interpolation
        using the flower pixels percentage of the previous and following images. After that, an averaging
        filter is applied to smoothen the flower pixels percentage
        """
        # Get an ordered list of tuples containing (timestamp, image filename, flower pixels percentage, cluster)
        imgs_list = []
        for fname, timestamp in self.imgs_timestamps.items():
            # Get flower pixels percentage of the image
            fw_pix_perc = self.flower_pix_perc[fname]

            # Get cluster of the image
            cluster = self.img_clusters[fname]

            # Append tuple to the list
            imgs_list.append((timestamp, fname, fw_pix_perc, cluster))

        # Sort the list
        imgs_list.sort()

        # Extract the index of the best images and sort
        imgs_list_best = [(i, imgs_list[i]) for i in range(len(imgs_list)) if self.best_imfnames.__contains__(imgs_list[i][1])]
        imgs_list_best.sort()

        # Initialize estimation of flower pixels percentage with the first image
        flower_pix_perc_est = [(imgs_list[0][0], imgs_list[0][1], imgs_list_best[0][3])]

        # Iterate through all the elements
        for i in range(len(imgs_list)):
            # Get current image
            t = imgs_list[i]

            # If current image is one of the best ones, take its flower pixels percentage
            if self.best_imfnames.__contains__(t[3]):
                flower_pix_perc_est.append((t[0],t[1],t[2]))
                continue

            # Else, perform interpolation between previous image and the next best image.
            # Get the closest image after the current one
            after = [p for p in imgs_list_best if p[0] > i]

            # If there are no best images after the current one, assign the previous flower pixels percentage to this image
            if len(after) == 0:
                flower_pix_perc_est.append((t[0],t[1],flower_pix_perc_est[-1][2]))
                continue

            # Else, perform interpolation. Get index of the next cluster-0 image
            after.sort()
            next_idx_dist = after[0][0] - i

            # Get flower pixels percentages of previous image and next cluster-0 images
            prev_fwpxperc = flower_pix_perc_est[-1][2]
            after_fwpxperc = after[0][3]

            # Perform interpolation
            flower_pix_perc_est.append((t[0],t[1],(prev_fwpxperc * next_idx_dist + after_fwpxperc)/(next_idx_dist + 1)))

        # Sort the results
        flower_pix_perc_est.sort()

        # Initialize final flower pixels percentage estimation
        final_fw_pix_perc_est = {}

        # Filter the estimation: Filter width = 1 hour (60 frames)
        for i in range(len(flower_pix_perc_est)):
            # Get current image's filename
            fname = flower_pix_perc_est[i][1]

            # Get window limits
            w_range = (max(0, i-30), min(len(flower_pix_perc_est), i+30))

            # Get window
            window = flower_pix_perc_est[w_range[0]:w_range[1]]

            # Get average
            average = np.average([w[2] for w in window])

            # Store
            final_fw_pix_perc_est[fname] = average

        # Store as object's attribute
        self.flower_pix_perc_estimate = final_fw_pix_perc_est

    def estimateFlowerCount(self):
        """
        Calculate the number of flowers pixels for flagged (i.e. bad) images by doing an interpolation
        using the flower count of the previous and following images. After that, an averaging
        filter is applied to smoothen the result
        """
        # Get an ordered list of tuples containing (timestamp, image filename, flower count, cluster)
        imgs_list = []
        for fname, timestamp in self.imgs_timestamps.items():
            # Get flower count of the image
            if self.flower_count.__contains__(fname):
                fCount = len(self.flower_count[fname])
            else:
                fCount = None

            # Get cluster of the image
            cluster = self.img_clusters[fname]

            # Append tuple to the list
            imgs_list.append((timestamp, fname, fCount, cluster))

        # Sort the list
        imgs_list.sort()

        # Extract only the images' indexes that have a flower count value
        imgs_list_best = [(i, imgs_list[i]) for i in range(len(imgs_list)) if imgs_list[i][2] is not None]
        imgs_list_best.sort()

        # Initialize estimation of flower count with the first image
        flower_count_est = [(imgs_list[0][0], imgs_list[0][1], imgs_list_best[0][3])]

        # Iterate through all the elements
        for i in range(len(imgs_list)):
            # Get current image
            t = imgs_list[i]

            # If current image has a flower count, take its flower count
            if t[2] is not None:
                flower_count_est.append((t[0], t[1], t[2]))
                continue

            # Else, perform interpolation between previous image and the next cluster-0 image.
            # Get the closest image after the current one
            after = [p for p in imgs_list_best if p[0] > i]

            # If there are no images with flower coutn after the current one, assign the previous flower count to this image
            if len(after) == 0:
                flower_count_est.append((t[0], t[1], flower_count_est[-1][2]))
                continue

            # Else, perform interpolation. Get index of the next cluster-0 image
            after.sort()
            next_idx_dist = after[0][0] - i

            # Get flower count of previous image and next cluster-0 images
            prev_count = flower_count_est[-1][2]
            after_count = after[0][3]

            # Perform interpolation
            flower_count_est.append(
                (t[0], t[1], (prev_count * next_idx_dist + after_count) / (next_idx_dist + 1)))

        # Sort the results
        flower_count_est.sort()

        # Initialize final flower pixels percentage estimation
        final_fw_count_est = {}

        # Filter the estimation: Filter width = 1 hour (60 frames)
        for i in range(len(flower_count_est)):
            # Get current image's filename
            fname = flower_count_est[i][1]

            # Get window limits
            w_range = (max(0, i - 30), min(len(flower_count_est), i + 30))

            # Get window
            window = flower_count_est[w_range[0]:w_range[1]]

            # Get average
            average = np.average([w[2] for w in window])

            # Store
            final_fw_count_est[fname] = average

        # Store as object's attribute
        self.flower_count_estimate = final_fw_count_est

    #####################################################################
    ##                         GETTER FUNCTIONS                        ##
    #####################################################################

    def getFlowersPresent(self):
        """
        Despite the unclear name, get whether there are flowers on the field or not
        :return: True if flowers are present on the field. False otherwise
        """
        # TODO
        return True

    def getPercentageOfFlowerPixels(self, periodLimits = None):
        """
        Gets an approximation of the percentage of flower pixels along the day
        :param periodLimits: List containing the limits of the periods in which the day will be divided (String. Format: HH:MM)
        :return: An approximate percentage of the number of flower pixels in the day
        """
        # Initialize periods of the day
        fwpix_periods = []

        # Divide the images in time periods
        if periodLimits is not None:
            fwpix_periods = self.splitImagesInPeriods(periodLimits)
        else:
            fwpix_periods.append(self.imfilenames)

        # Initialize list with the average flower pixels per period
        fwpix_per_period = []

        # Iterate through the periods
        for k in range(len(fwpix_periods)):
            # Get percentages of images from cluster 0
            valid_percentages = [self.flower_pix_perc_estimate[p] for p in fwpix_periods[k]]

            # Check that valid_percentages is not empty
            if len(valid_percentages) == 0:
                fwpix_per_period.append(-1)

            # Calculate average
            else:
                fwpix_per_period.append(np.average(valid_percentages))

        return fwpix_per_period

    def getNumberOfFlowers(self, periodLimits = None, tolerance = 5):
        """
        Returns an estimation of the number of flowers per each of the periods set by the parameters
        :param periodLimits: List including strings representing the times when a period ends and a new one starts
        :param tolerance: Tolerance on counting (by default: 5)
        :return: An approximate number of flowers in the day
        """
        # Initialize periods of the day
        nFlowers_periods = []

        # Divide the images in time periods
        if periodLimits is not None:
            nFlowers_periods = self.splitImagesInPeriods(periodLimits)
        else:
            nFlowers_periods.append(self.imfilenames)

        # Initialize list with the average flower pixels per period
        nFlowers_per_period = []

        # Iterate through the periods
        for k in range(len(nFlowers_periods)):
            # Get the number of flowers for the images in this period
            fwcount = [len(v) for t, v in self.flower_count.items() if nFlowers_periods[k].__contains__(t)]

            if len(fwcount) == 0:
                nFlowers_per_period.append(-1)
                continue

            # Get histogram of the flower count for the present period
            # Get histogram limits
            fwcount_hist_max = int(np.ceil(np.max(fwcount) / tolerance) * tolerance)
            fwcount_hist_min = int(int(np.min(fwcount) / tolerance) * tolerance)

            # Get number of bins
            fwcount_hist_nbins = max(1, int((fwcount_hist_max - fwcount_hist_min) / tolerance))

            # Get histogram
            fwcount_hist, fwcount_hist_edges = np.histogram(fwcount, fwcount_hist_nbins, [fwcount_hist_min, fwcount_hist_max])

            # Store flower count
            nFlowers_per_period.append(fwcount_hist_edges[fwcount_hist.argmax()] + tolerance/2)

        return nFlowers_per_period

    #####################################################################
    ##                         MAIN FUNCTIONS                          ##
    #####################################################################

    def runAnalysis(self, periodLimits = None, skipEstimateFlowerPerc = False, skipEstimateFlowerCount = False, skipFlowerPercCount = False, skipFlowerCount = False):
        """
        Runs complete analysis
        :param periodLimits: Periods in which the day is divided. Contains the times when a period ends and a new one start
        :param skipEstimateFlowerPerc: Determines if estimation of flowers pixels percentage must be skipped (by default: False)
        :param skipEstimateFlowerCount: Determines if estimation of number of flowers must be skipped (by default: False)
        :param skipFlowerPercCount: Determines if flower pixels percentage counting must be skipped (by default: False)
        :param skipFlowerCount: Determines if flower counting must be skipped (by default: False)
        :param doFlowerCountAll: Determines if flower counting should be done in all images (by default: False)
        :param doFlowerCountBestOf20: Determines if flower counting should be done using the best 20 images. This flag is not used if doFlowerCountAll is True (by default: False)
        :return: Flower pixels percentage and number of flowers for each period
        """
        # Check if there are flowers in the field
        if not self.getFlowersPresent():
            fw_pix_perc = 0
            n_flowers = 0

        else:
            # Estimate flower pixels
            if not skipEstimateFlowerPerc:
                self.estimateFlowerPixels()

            # Estimate flower count
            if not skipEstimateFlowerCount:
                self.estimateFlowerCount()

            # Getting results
            # Get percentage of flower pixels
            if not skipFlowerPercCount:
                fw_pix_perc = self.getPercentageOfFlowerPixels(periodLimits)
            else:
                fw_pix_perc = -1

            # Get number of flowers
            if not skipFlowerCount:
                n_flowers = self.getNumberOfFlowers(periodLimits)
            else:
                n_flowers = -1

        return fw_pix_perc, n_flowers

    #####################################################################
    ##                     SHOW TIME LAPSE FUNCS                       ##
    #####################################################################

    def showFlowersDetected(self, showFlowerHighlight = False, showFlowerAreaBoundaries = True):
        """
        Shows the images where the flowers were counted and their locations
        :param showFlowerHighlight: Additionally show the flower-highlight image where the flowers were counted (by default: False)
        :param showFlowerAreaBoundaries: Show flower area boundaries (by default: True)
        """
        # Iterate through images
        for fname, blobs in self.flower_count.items():
            # Read image
            PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
            pil_image = PIL.Image.open(self.imgspath + fname).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            img = open_cv_image[:, :, ::-1].copy()

            # Show flower mask in RED if requested
            if showFlowerHighlight:
                # Get flower highlight image
                img_fwhigh = self.getFlowerHighlight(img, fname, segm_dist_from_zerocross=10)

                # Show image
                cv2.imshow("Flower highlight", img_fwhigh)

            # Show plot boundaries in PURPLE if requested
            if showFlowerAreaBoundaries:
                cv2.line(img, self.flower_area_bounds[0], self.flower_area_bounds[1], (255, 0, 128), 2)
                cv2.line(img, self.flower_area_bounds[0], self.flower_area_bounds[2], (255, 0, 128), 2)
                cv2.line(img, self.flower_area_bounds[1], self.flower_area_bounds[3], (255, 0, 128), 2)
                cv2.line(img, self.flower_area_bounds[2], self.flower_area_bounds[3], (255, 0, 128), 2)

            # Paint circles surrounding flowers
            for b in blobs:
                # Extract position and radius
                x, y, r = b

                # Draw circle
                cv2.circle(img, center=(int(y), int(x)), radius=int(r), color=(0, 255, 255), thickness=2)

            # Show image
            cv2.imshow("Flowers detected", img)
            cv2.waitKey(0)

        # Destroy all open windows
        cv2.destroyAllWindows()

    def showFlowerCountGraph(self, title, flower_area_meters = None, bin_width = 1):
        """
        Plots the flower count for this flower detector along the day (estimation)
        along with the average.
        :param title: Title of the graph
        :param flower_area_meters: Area of the flower area. Specify this parameter to obtain flower count per square meter (by default: None)
        :param bin_width: Integer. Width of the bins of the histogram to be plotted (by default: 1)
        :return: List containing pairs of timestamp and estimated number of flowers along the day
        """
        # Initial assert
        assert isinstance(bin_width, int) and bin_width > 0, "bin_width must be a positive integer"

        # Check title
        if title is None:
            title = ""
        else:
            title = str(title)

        # Check plot area in meters
        if flower_area_meters is not None:
            assert isinstance(flower_area_meters, (int, float)) and flower_area_meters > 0, "flower_area_meters must be a positive number"

        # Get list of flower count estimations (regardless the time of the day)
        fwcount = [len(v) for v in self.flower_count.values()]

        # Get histogram limits rounded to the next multiple of 5
        fwcount_hist_max = int(np.ceil(np.max(fwcount) / bin_width) * bin_width)
        fwcount_hist_min = int(int(np.min(fwcount) / bin_width) * bin_width)

        # Get number of bins (force to be at least one bin)
        fwcount_hist_nbins = max(1, int((fwcount_hist_max - fwcount_hist_min)/bin_width))

        # Get axis names
        if flower_area_meters is None:
            x_axis_name = "Number of flowers in area"
        else:
            x_axis_name = "Number of flowers per square meter"
        y_axis_name = "Frequency"

        # Create graph
        plt.figure(1)
        plt.title(title + ": Real count")
        plt.hist(fwcount, bins=fwcount_hist_nbins, range=[fwcount_hist_min, fwcount_hist_max])
        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)

        # Get first and last of the images that have flower count
        fwcount_imgs = [(self.imgs_timestamps[k], k) for k in self.flower_count.keys()]
        fwcount_imgs.sort()
        fwcount_img_first = fwcount_imgs[0]
        fwcount_img_last = fwcount_imgs[-1]

        # Get list with the images between the first and the last images with flower count
        fwcount_list = [self.flower_count_estimate[k] for k in self.flower_count_estimate.keys() if
                        self.imgs_timestamps[k] >= fwcount_img_first[0] and self.imgs_timestamps[k] <= fwcount_img_last[
                            0]]

        # Plot estimation
        plt.figure(2)
        plt.title(title + ": Estimate count")
        plt.hist(fwcount_list, bins=fwcount_hist_nbins, range=[fwcount_hist_min, fwcount_hist_max])
        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)

        # Show plots
        plt.show()

    def showFlowerPixelsPercentageGraph(self, title):
        """
        Plots the flower pixels percentage for this flower detector along the day (estimation)
        along with the average.
        :param title: Title of the graph
        :return: List containing pairs of timestamp and estimated flowers pixel percentage along the day
        """
        # Check title
        if title is None:
            title = ""
        else:
            title = str(title)

        # Get list of tuples (timestamp, str(timestamp), flower count)
        timestamp_fwPix = [(v, str(v.time()), self.flower_pix_perc_estimate[k]) for k, v in self.imgs_timestamps.items()]

        # Sort list
        timestamp_fwPix.sort()

        # Get values for Y axis
        y_values = [k[-1] for k in timestamp_fwPix]

        # Get ticks labels for X axis
        x_ticks_labels = [k[0] for k in timestamp_fwPix]

        # Get plot parameters
        max_x = len(x_ticks_labels)
        min_x = 0
        max_y = max(y_values) + 1
        min_y = 0

        # Get axis names
        y_axis_name = "Flowers pixels percentage in plot"
        x_axis_name = "Time"

        # Create graph
        plt.title(title)
        plt.plot(range(len(y_values)), y_values)
        plt.xticks(range(len(y_values)), x_ticks_labels, rotation='vertical')
        plt.show()

    def showTimeLapse(self, playback_ms = 40, showFlowerMask = True, showPlotBoundaries = True, showCluster = (0,1,2,3)):
        """
        Show time lapse video with features extracted
        :param playback_ms: Playback period between frames in milliseconds (by default: 40 ms)
        :param showFlowerMask: Flag to determine whether to show the detected flowers or not (by default: True)
        :param showPlotBoundaries: Flag to determine whether to show the plot boundaries or not (by default: True)
        :param showCluster: Tuple including the clusters to be shown (integer if only one cluster is shown) (by default, all)
        """
        # Initial assert
        assert isinstance(playback_ms, int) and playback_ms > 0, "Playback_ms must be a positive integer"
        assert isinstance(showFlowerMask, bool), "ShowFlowerMask must be either True or False"

        # Adjust showCluster variable
        if isinstance(showCluster, int):
            showCluster = [showCluster]
        else:
            assert isinstance(showCluster, tuple), "ShowCluster must be a tuple or an integer"

        # Iterate through images
        for fname in self.imfilenames:
            # Avoid showing flagged pics if requested:
            if not showCluster.__contains__(self.img_clusters[fname]):
                continue

            # Read image
            PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
            pil_image = PIL.Image.open(self.imgspath + fname).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            img = open_cv_image[:, :, ::-1].copy()

            # Show flower mask in RED if requested
            if showFlowerMask:
                # Get flower mask
                img_mask = self.getFlowerMask(img, fname)

                # Apply mask to original image
                img[img_mask > 0] = (0, 0, 255)

            # Show plot boundaries in PURPLE if requested
            if showPlotBoundaries:
                cv2.line(img, self.plot_bounds[0], self.plot_bounds[1], (255, 0, 128), 2)
                cv2.line(img, self.plot_bounds[0], self.plot_bounds[2], (255, 0, 128), 2)
                cv2.line(img, self.plot_bounds[1], self.plot_bounds[3], (255, 0, 128), 2)
                cv2.line(img, self.plot_bounds[2], self.plot_bounds[3], (255, 0, 128), 2)

            # Show image
            cv2.imshow("Time lapse", img)
            cv2.waitKey(playback_ms)

        # Destroy all open windows
        cv2.destroyAllWindows()

    def showTimeLapse_demo(self, n_images, playback_ms = 40, showFlowerMask = True, showPlotBoundaries = True, showCluster = (0,1,2)):
        """
        Show time lapse for the demo
        :param n_images: Number of images to display
        :param playback_ms: Playback rate in milliseconds (by default: 40)
        :param showFlowerMask: Flag to show a mask with the detected flower pixels (by default: True)
        :param showPlotBoundaries: Flag to determine whether to show the plot boundaries or not (by default: True)
        :param showCluster: Tuple including the clusters to be shown (integer if only one cluster is shown) (by default, all)
        """
        # Initial assert
        assert isinstance(n_images, int) and n_images > 0, "n_images must be a positive integer (seconds)"
        assert isinstance(playback_ms, int) and playback_ms > 0, "playback_ms must be a positive integer (seconds)"
        assert isinstance(showFlowerMask, bool), "ShowFlowerMask must be either True or False"

        # Adjust showCluster variable
        if isinstance(showCluster, int):
            showCluster = [showCluster]
        else:
            assert isinstance(showCluster, tuple), "ShowCluster must be a tuple or an integer"

        # Subsample images
        if n_images >= len(self.imfilenames):
            subsample_images = self.imfilenames
        else:
            sampling_rate = len(self.imfilenames) / n_images
            subsample_images = [self.imfilenames[round(i * sampling_rate)] for i in range(n_images)]

        # Iterate through images
        for fname in subsample_images:
            # Avoid showing flagged pics if requested:
            if not showCluster.__contains__(self.img_clusters[fname]):
                continue

            # Read image
            PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
            pil_image = PIL.Image.open(self.imgspath + fname).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            img = open_cv_image[:, :, ::-1].copy()
            img_copy = np.copy(img)

            # Show flower mask in RED if requested
            if showFlowerMask:
                # Get flower mask
                img_mask = self.getFlowerMask(img, fname)

                # Apply mask to original image
                img[img_mask > 0] = (0, 0, 255)

            # Show plot boundaries in PURPLE if requested
            if showPlotBoundaries:
                cv2.line(img, self.plot_bounds[0], self.plot_bounds[1], (255, 0, 128), 2)
                cv2.line(img, self.plot_bounds[0], self.plot_bounds[2], (255, 0, 128), 2)
                cv2.line(img, self.plot_bounds[1], self.plot_bounds[3], (255, 0, 128), 2)
                cv2.line(img, self.plot_bounds[2], self.plot_bounds[3], (255, 0, 128), 2)

            # Show image
            cv2.imshow("Time lapse", img)
            cv2.imshow("Original image", img_copy)
            cv2.waitKey(playback_ms)



if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        imgspath = sys.argv[1]


    processing_start_time = time()

    flowerCounter = FlowerCounter('/home/hduser/plot_images/2016-07-05_1207')
    flowerCounter.loadFolder()
    
    processing_end_time = time() - processing_start_time
    print("SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3)))
