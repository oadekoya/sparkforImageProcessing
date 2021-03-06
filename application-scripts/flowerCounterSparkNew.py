import os
import cv2
import numpy as np
import subprocess
import pyspark
import sys
import PIL
from time import time
from StringIO import StringIO
from PIL import Image, ImageFile
import copy
import sys
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
import skimage.io as io
from pyspark.sql import SQLContext
from pyspark.sql.types import Row
import subprocess
from operator import add
from sklearn.cluster import KMeans
from skimage.feature import blob_doh
from scipy.misc import imread, imsave

def images_to_bytes(rawdata):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return rawdata[0], np.array(io.imread(StringIO(rawdata[1])))
    except (ValueError, IOError, SyntaxError) as e:
        pass


class FlowerCounter(object):


    def __init__(self, imsize, region_matrix, segm_B_upper, segm_B_lower, flower_pix_perc_thresh):
        self.imsize = imsize
        self.region_matrix=region_matrix
        self.segm_B_upper = segm_B_upper
        self.segm_B_lower = segm_B_lower
        self.flower_pix_perc_thresh = flower_pix_perc_thresh
        # class level varaibles replacing global variables
        self.hist_G_all = {}
        #: Plot mask
        self.plot_mask = []    
        #: Plot bounds
        self.plot_bounds = []
        #: Histograms of the b (Lab) channel for all images (only for pixels within the plot)
        self.hist_b_all = {}
        #: Average histogram of B component
        self.avg_hist_b = np.zeros(256)
        #: Mask of the area where flowers will be counted
        self.flower_area_mask = []
        #: Histogram shift for component B of each image
        self.hist_b_shifts = {}
        #: Dictionary storing the fraction of flower pixels in each image
        self.flower_pix_perc = {}
        #: Estimate of number of flowers per image
        self.flower_count_estimate = {}


    @staticmethod
    def crossProduct(p1, p2, p3):
        """
        Cross product implementation: (P2 - P1) X (P3 - P2)
        :param p1: Point #1
        :param p2: Point #2
        :param p3: Point #3
        :return: Cross product
        """
        v1 = [p2[0] - p1[0], p2[1] - p1[1]]
        v2 = [p3[0] - p2[0], p3[1] - p2[1]]
        return v1[0] * v2[1] - v1[1] * v2[0]

    @staticmethod
    def userDefinePlot(img, bounds = None):
        """

        :param image: The image array that contains the crop
        :param bounds: Optionally user can set up previously the bounds without using GUI
        :return: The four points selected by user and the mask to apply to the image
        """
        # Initial assert
        if not isinstance(img, np.ndarray):
            print("Image is not a numpy array")
            return

        # Get image shape
        shape = img.shape[::-1]

        # Eliminate 3rd dimension if image is colored
        if len(shape) == 3:
            shape = shape[1:]


        # Function definitions
        def getMask(boundM):
            """
            Get mask from bounds
            :return: Mask in a numpy array
            """
            # Initialize mask
            #shapeM = img.shape[1::-1]
            mask = np.zeros(shape[::-1])

            # Get boundaries of the square containing our ROI
            minX = max([min([x[0] for x in boundM]), 0])
            minY = max([min([y[1] for y in boundM]), 0])
            maxX = min(max([x[0] for x in boundM]), shape[0])
            maxY = min(max([y[1] for y in boundM]), shape[1])


            # Reshape bounds
            #boundM = [(minX, minY), (maxX, minY), (minX, maxY), (maxX, maxY)]

            # Iterate through the containing-square and eliminate points
            # that are out of the ROI
            for x in range(minX, maxX):
                for y in range(minY, maxY):
                    h1 = FlowerCounter.crossProduct(boundM[2], boundM[0], (x, y))
                    h2 = FlowerCounter.crossProduct(boundM[3], boundM[1], (x, y))
                    v1 = FlowerCounter.crossProduct(boundM[0], boundM[1], (x, y))
                    v2 = FlowerCounter.crossProduct(boundM[2], boundM[3], (x, y))
                    if h1 > 0 and h2 < 0 and v1 > 0 and v2 < 0:
                        mask[y, x] = 255

            return mask


            # Check if bounds have been provided

        if isinstance(bounds, list):
            if len(bounds) != 4:
                print("Bounds length must be 4. Setting up GUI...")
            else:
                mask = getMask(bounds)
                return bounds, mask

        # Get image shape
        #shape = img.shape[1::-1]

        # Initialize boudaries
        bounds = [(207,156), (1014, 156), (207, 677), (1014, 677)]

        #if plot == False:
        #    #for flower area
        #    bounds = [(308, 247), (923, 247), (308, 612), (923, 612)]


        # Get binary mask for the user-selected ROI
        mask = getMask(bounds)

        return bounds, mask


    def setPlotMask(self, bounds, mask = None):
        """
        Set mask of the plot under analysis
        :param mask: Mask of the plot
        :param bounds: Bounds of the plot
        """

        # global plot_mask, plot_bounds

        # Initial assert
        if mask is not None:
            assert isinstance(mask, np.ndarray), "Parameter 'corners' must be Numpy array"
            assert mask.shape == imsize, "Mask has a different size"
        assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"


        # Store bounds
        self.plot_bounds = bounds

        # Store mask
        if mask is None:
            _, self.plot_mask = ((self.userDefinePlot(np.zeros(imsize), bounds)))

        else:
            self.plot_mask = mask


    
    def setFlowerAreaMask(self, region_matrix, mask):
        """
        Set mask of the flower area within the plot
        :param region_matrix = Region matrix representing the flower area
        :param mask: Mask of the flower area
        """

        # global flower_area_mask
        # Initial assert
        if mask is not None:
            assert isinstance(mask, np.ndarray), "Parameter 'mask' must be Numpy array"
            assert mask.shape == imsize, "Mask has a different size"

        #assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"

        # Store bounds
        self.flower_region_matrix = region_matrix

        # Store mask
        self.flower_area_mask = mask



    def computePlotMask(self, images):
        """
        Compute plot mask
        """
        # Trace
        print("Computing plot mask...")
        (key, value) = images

        # Read an image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        open_cv_image = np.array(value)

        p_bounds, p_mask = self.userDefinePlot(open_cv_image, None)

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
        fw_mask = np.zeros(imsize)

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
                    _, mask = self.userDefinePlot(fw_mask, list(bounds_T))

                    # Apply mask
                    fw_mask[mask > 0] = 255

        # Save flower area mask & bounds
        self.setFlowerAreaMask(region_matrix, fw_mask)


    
    def computeHistograms(self, images_bytes_rdd):
        """
        Compute the average A and B histograms over all images
        """

        # Trace
        print("Computing histograms...")

        histogram_rdd = (images_bytes_rdd.map(self.histogram))

        for hist_b, key, hist_G in histogram_rdd.collect():
            self.hist_b_all[key] = hist_b
            self.hist_G_all[key] = hist_G


        return self.hist_b_all, self.hist_G_all


    
    def histogram(self, histogramRDD):
        (key, value) = histogramRDD

        print "Calculating histogram"

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        array_image = np.asarray(value)
        im_bgr = np.array(array_image)

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

        return hist_b, key, hist_G


    def computeHistogramShifts(self, images_bytes_rdd):
        """
        Compute histogram shifts respect to the average histograms for each image
        """
        # Trace
        print("Computing histogram shifts...")


        hist_shifts_rdd = (images_bytes_rdd.map(self.histShifts))

        for x_shift_b, key in hist_shifts_rdd.collect():
            self.hist_b_shifts[key] = x_shift_b

        correlation_reference = np.correlate(self.hist_b_all.values()[0],np.average(list(self.hist_b_all.values()), axis=0),"full")

        overall_shift_b = correlation_reference.argmax().astype(np.uint8)

        for key, shift in self.hist_b_shifts.items():
            self.hist_b_shifts[key] = shift + overall_shift_b

        return self.hist_b_shifts



    def histShifts(self, images_bytes_rdd):
        (key, value) = images_bytes_rdd

        # Calculate correlation
        #correlation_b = np.correlate(hist_b_all[key], avg_hist_b, "full")
        #if not key == hist_b_all.keys()[0]:
        correlation_b = np.correlate(self.hist_b_all[key], self.hist_b_all.values()[0], "full")
        # Get the shift on the X axis
        x_shift_b = correlation_b.argmax().astype(np.int8)

        return x_shift_b, key



    def computeAverageHistograms(self):
        """
        Compute average B histogram
        """
        print("Computing average histogram..")

        self.avg_hist_b = np.average([np.roll(h, -self.hist_b_shifts[f]) for f, h in self.hist_b_all.items()], axis=0)


    
    def getFlowerHighlight(self, img, fname, segm_dist_from_zerocross = 8):
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
        assert img.shape[:2] == imsize, "img is not of the same size as this object's"
        #assert isinstance(fname, str), "fname must be a string"
        assert self.hist_b_shifts.__contains__(fname), "fname does not exist on this object's scope"
        #assert isinstance(segm_out_value, (float, int)), "semg_out_value must be a number"
        #assert max(0, min(1, segm_out_value)) == segm_out_value, "segm_out_value must be between 0 and 1"
        assert isinstance(segm_dist_from_zerocross, int), "segm_dist_from_zerocross must be an integer"
        assert segm_dist_from_zerocross > 0, "segm_dist_from_zerocross must be positive"

        # Convert to LAB
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        # Get the B channel and convert to float
        img_B = np.array(img_lab[:, :, 2], dtype=np.float32)

        threshold = (segm_B_lower + self.hist_b_shifts[fname]) % 256
        img_B = np.clip(img_B, 0, threshold + 10)
        img_B -= threshold - 8
        img_B /= threshold + 10
        img_B = np.clip(img_B, 0, 1)

        return img_B


    def computeFlowerPixelsPercentage(self, images_bytes_rdd):
        """
        Compute the percentage of flower pixels in all of the pictures.
        Since we only apply thresholding in the B channel to get the flower pixels,
        we only need the B histogram to get the number of flower pixels in the plot.
        """
        # Trace
        print("Computing flower pixels percentage...")

        global flower_pix_perc

        # Iterate through all images
        flower_pix_perc_rdd = (images_bytes_rdd.map(self.flowerPixelsPercentage))

        for key, fw_pix_perc in flower_pix_perc_rdd.collect():
            # Store it in dictionary
            self.flower_pix_perc[key] = fw_pix_perc

        return self.flower_pix_perc



   
    def flowerPixelsPercentage(self, images):
        (key, value) = images

        # Get number of pixels in the plot
        n_plot_pix = len(self.plot_mask[self.plot_mask > 0])

        hist_b = self.hist_b_all[key]
        hist_b_shift = self.hist_b_shifts[key]


        # Calculate number of flower pixels
        n_flower_pixels = np.sum(hist_b[segm_B_lower + hist_b_shift: segm_B_upper + hist_b_shift + 1])

        # Get percentage of flower pixels
        fw_pix_perc = n_flower_pixels / n_plot_pix

        return key, fw_pix_perc



    def computeFlowerCount(self, images):
        """
        Get flower count for the best images
        """
        # Trace
        print("Computing number of flowers...")


        flower_count_RDD = (images.map(self.flowerCount))


        flower_count_RDD.coalesce(1, shuffle=True).saveAsTextFile("hdfs://discus-p2irc-master:54310/tmp/output/")


    
    def flowerCount(self, imageRDD):
        (key, value) = imageRDD
        # Trace
        print "Computing flower count"

        # Get flower mask for this image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        #pil_image = PIL.Image.open(value).convert('RGB')
        open_cv_image = np.array(value)
        img = open_cv_image[:, :, ::-1]

        # Highlight flowers
        img_flowers = self.getFlowerHighlight(img, key, segm_dist_from_zerocross=6)

        # Apply flower area mask
        img_flowers[self.flower_area_mask == 0] = 0

        try:
            # Get number of flowers using blob counter on the B channel
            blobs = blob_doh(img_flowers, max_sigma=5, min_sigma=3)

            n_blobs = len(blobs)

            return key, n_blobs

        except (IndexError) as e:
            pass


    # helper function to create a pipe
    def runner(self, images_bytes) :
        print("Running runner pipeline")

        self.computePlotMask(images_bytes.first())
        self.computeFlowerAreaMask(region_matrix)

        self.computeHistograms(images_bytes)
        self.computeHistogramShifts(images_bytes)
        self.computeAverageHistograms()
        self.computeFlowerPixelsPercentage(images_bytes)

        self.computeFlowerCount(images_bytes)



if __name__ == "__main__":

    # Total time is the time from where the SparkContext is initiated to its shutting down, that is, the build time + processing time + overhead
    total_time_start = time()

    job_name = sys.argv[1]
    #filenames = list_files(sys.argv[2])
    #job_name = sys.argv[3]

    sc = SparkContext(appName = job_name)


    # Remove the output directory and all it contents if it already exist
    subprocess.call(["hadoop", "fs", "-rm", "-r", "/tmp/output/"])

    # Time to build the data structures and load input data for computation
    build_start_time = time()

    #: Image size
    imsize = (720, 1280)

    region_matrix = [[0,0,0],[0,1,0],[0,0,0]]

    #: Upper limit of B channel for segmentation
    segm_B_upper = 255


    #: Bandwidth used for segmentation on the B channel
    segm_B_lower = 155

    #: Threshold for percentage of flower pixels to consider that there are flowers in the field 
    flower_pix_perc_thresh = 0.0001

    images_read_rdd = (sc.binaryFiles('hdfs://discus-p2irc-master:54310/user/hduser/plot_images/2016-07-05_1207'))

    images_bytes = (images_read_rdd.map(images_to_bytes))

    build_end_time = time() - build_start_time

    print "SUCCESS: Images data structures procesed in {} seconds".format(round(build_end_time, 3))

    processing_start_time = time()

    flowerCounter = FlowerCounter(imsize=imsize, region_matrix=region_matrix, segm_B_upper=segm_B_upper, segm_B_lower=segm_B_lower, flower_pix_perc_thresh=flower_pix_perc_thresh)

    flowerCounter.runner(images_bytes)

    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))

    sc.stop()

    total_time_end = time() - total_time_start

    print "SUCCESS: Total application time in {} seconds".format(round(total_time_end, 3))

