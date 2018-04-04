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
#import PIL.Image
#import PIL.ImageFile
import PIL
import glob
import operator
import matplotlib.pyplot as plt
from time import time
from PIL import Image, ImageFile
# K-Means
from sklearn.cluster import KMeans
import multiprocessing
# Flower counting
from skimage.feature import blob_doh

# Progress bar
from tqdm import tqdm
import sys

# Timestamp
#import pytz
import datetime

#os.system("taskset -p 0xff %d" % os.getpid())

def list_files(images_path):
    """
    returns a list of names (with extension, without full path) of all files
    in folder path
    """
    imageFiles = []
    for name in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, name)):
            imageFiles.append(images_path + "/" + name)
    return imageFiles



def get_images_paths(images_path):
    imagesPaths = []
    for dir in os.listdir(images_path):
        new_dir = os.path.join(images_path, dir)
        if os.path.isdir(new_dir):
            imagesPaths.append(new_dir)
    return imagesPaths




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
                h1 = crossProduct(boundM[2], boundM[0], (x, y))
                h2 = crossProduct(boundM[3], boundM[1], (x, y))
                v1 = crossProduct(boundM[0], boundM[1], (x, y))
                v2 = crossProduct(boundM[2], boundM[3], (x, y))
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




def setPlotMask(bounds, mask = None):
    """
    Set mask of the plot under analysis
    :param mask: Mask of the plot
    :param bounds: Bounds of the plot
    """

    #global plot_mask_accum, plot_bounds_accum
    global plot_mask, plot_bounds

    print(type(mask))
    # Initial assert
    if mask is not None:
        assert isinstance(mask, np.ndarray), "Parameter 'corners' must be Numpy array"
        assert mask.shape == imsize, "Mask has a different size"
    assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"


    # Store bounds
    plot_bounds = bounds
    #plot_bounds_accum.add(bounds)

    # Store mask
    if mask is None:
        _, plot_mask = ((userDefinePlot(np.zeros(imsize), bounds)))

    else:
        plot_mask = mask
        #plot_mask_accum.add(mask)

    #plot_mask_accum = np.array(plot_mask_accum.value)
    return plot_mask, plot_bounds




def setFlowerAreaMask(region_matrix, mask):
    """
    Set mask of the flower area within the plot
    :param region_matrix = Region matrix representing the flower area
    :param mask: Mask of the flower area
    """

    global flower_area_mask
    #global flower_area_mask_accum
    # Initial assert
    if mask is not None:
        assert isinstance(mask, np.ndarray), "Parameter 'mask' must be Numpy array"
        assert mask.shape == imsize, "Mask has a different size"

    #assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"

    # Store bounds
    flower_region_matrix = region_matrix

    # Store mask
    #flower_area_mask_accum.add(mask)
    flower_area_mask = mask

    return flower_region_matrix, flower_area_mask




def computePlotMask(images):
    """
    Compute plot mask
    """
    # Trace
    print("Computing plot mask...")

    # Read an image
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
    pil_image = PIL.Image.open(images[0]).convert('RGB')
    open_cv_image = np.asarray(pil_image)
   
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    
    
    p_bounds, p_mask = userDefinePlot(open_cv_image, None)

    # Store mask and bounds
    setPlotMask(p_bounds, p_mask)





def computeFlowerAreaMask(region_matrix):
    """
    Compute the flower area mask based on a matrix that indicates which regions of the plot are part of the
    flower counting.
    :param region_matrix: Mmatrix reflecting which zones are within the flower area mask (e.g. in order to
    sample the center region, the matrix should be [[0,0,0],[0,1,0],[0,0,0]]
    """

    # Trace
    print("Computing flower area mask...")

    #global plot_bounds_accum
    global plot_bounds

    # Check for plot bounds
    assert len(plot_bounds) > 0, "Plot bounds not set. Please set plot bounds before setting flower area mask"

    #region_matrix = region_matrix.value
    # Convert to NumPy array if needed
    if not isinstance(region_matrix, np.ndarray):
        region_matrix = np.array(region_matrix)

    # Assert
    assert region_matrix.ndim == 2, 'region_matrix must be a 2D matrix'

    # Get the number of rows and columns in the region matrix
    rows, cols = region_matrix.shape

    # Get transformation matrix
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]]), np.float32(plot_bounds))

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
                _, mask = userDefinePlot(fw_mask, list(bounds_T))

                # Apply mask
                fw_mask[mask > 0] = 255

    # Save flower area mask & bounds
    setFlowerAreaMask(region_matrix, fw_mask)





def computeHistograms(images):
    """
    Compute the average A and B histograms over all images
    """
    # Trace
    print("Computing histograms...")

        

    global hist_b_all, hist_G_all, plot_mask

    # Get the number of images and store it as number of samples
    nSamples = len(images)

    # Preliminary check
    if nSamples == 0:
        print("No images found")
        return

    # Iterate through all images
    for image in tqdm(images, file=sys.stdout):
        # Read image
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        pil_image = PIL.Image.open(image).convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        im_bgr = open_cv_image[:, :, ::-1].copy()

        # Shift to grayscale
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

        # Shift to LAB
        im_lab_plot = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2Lab)

        # Keep only plot pixels
        im_gray = im_gray[plot_mask > 0]
        im_lab_plot = im_lab_plot[plot_mask > 0]

        # Get histogram of grayscale image
        hist_G, _ = np.histogram(im_gray, 256, [0, 256])

        # Get histogram of B component
        hist_b, _ = np.histogram(im_lab_plot[:, 2], 256, [0, 256])

        # Save histograms
        hist_b_all[image] = hist_b
        hist_G_all[image] = hist_G

    return hist_b_all, hist_G_all
    


def computeAverageHistograms():
    """
    Compute average B histogram
    """
    print("Computing average histogram..")

    global hist_b_all, avg_hist_b
    try:

        # Vertically stack all the B histograms
        avg_hist_B = np.vstack(tuple([h for h in hist_b_all.values()]))

        # Sum all columns
        avg_hist_B = np.sum(avg_hist_B, axis=0)

        # Divide by the number of images and store
        avg_hist_b = np.divide(avg_hist_B, len(hist_b_all))

        print(avg_hist_b)
        return avg_hist_b

    except (ValueError) as e:
        pass


def computeHistogramShifts(images):
    """
    Compute histogram shifts respect to the average histograms for each image
    """
    # Trace
    print("Computing histogram shifts...")

    # Initialize dictionaries to be saved containing the histograms shifts
    global hist_b_shifts, avg_hist_b, hist_b_all

    # Iterate through images
    for image in tqdm(images, file=sys.stdout):
        try:
            # Calculate correlation
            correlation_b = np.correlate(hist_b_all[image], avg_hist_b, "full")

            # Get the shift on the X axis
            x_shift_b = correlation_b.argmax().astype(np.int8)

            # Append it to the list
            hist_b_shifts[image] = x_shift_b
        except (KeyError) as e:
            pass
    print(hist_b_shifts)


def computeFlowerPixelsPercentage(images):
    """
    Compute the percentage of flower pixels in all of the pictures.
    Since we only apply thresholding in the B channel to get the flower pixels,
    we only need the B histogram to get the number of flower pixels in the plot.
    """
    # Trace
    print("Computing flower pixels percentage...")
    global plot_mask, hist_b_all, hist_b_shifts, segm_B_lower, segm_B_upper,flower_pix_perc  

    # Get number of pixels in the plot
    n_plot_pix = len(plot_mask[plot_mask > 0])

    # Iterate through all images
    for image in tqdm(images, file=sys.stdout):
        try:
            # Get segmentation parameters
            hist_b = hist_b_all[image]
            hist_b_shift = hist_b_shifts[image]

            # Calculate number of flower pixels
            n_flower_pixels = np.sum(hist_b[segm_B_lower + hist_b_shift: segm_B_upper + hist_b_shift + 1])

            # Get percentage of flower pixels
            fw_pix_perc = n_flower_pixels / n_plot_pix

            # Store it in dictionary
            flower_pix_perc[image] = fw_pix_perc

        except (KeyError, ValueError) as e:
            pass

     # Copy dictionary
    flower_pix_perc_estimate = flower_pix_perc
    print(flower_pix_perc)



def getFlowerHighlight(img, fname, segm_out_value = 0.99, segm_dist_from_zerocross = 5):
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

    #global hist_b_shifts_accum, segm_B_lower
    global hist_b_shifts, segm_B_lower
    
    # Initial assert
    assert isinstance(img, np.ndarray), "img parameter must be a Numpy array"
    assert img.shape[:2] == imsize, "img is not of the same size as this object's"
    #assert isinstance(fname, str), "fname must be a string"
    assert hist_b_shifts.__contains__(fname), "fname does not exist on this object's scope"
    assert isinstance(segm_out_value, (float, int)), "semg_out_value must be a number"
    assert max(0, min(1, segm_out_value)) == segm_out_value, "segm_out_value must be between 0 and 1"
    assert isinstance(segm_dist_from_zerocross, int), "segm_dist_from_zerocross must be an integer"
    assert segm_dist_from_zerocross > 0, "segm_dist_from_zerocross must be positive"

   

    # Convert to LAB
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Get the B channel and convert to float
    img_B = np.array(img_lab[:, :, 2], dtype=np.float32)

    # Get the parameter T for the formula
    t_exp = segm_B_lower + hist_b_shifts[fname] - segm_dist_from_zerocross

    # Get the parameter K for the formula
    k_exp = np.log(1 / segm_out_value - 1) / segm_dist_from_zerocross

    # Apply logistic transformation
    img_B = 1 / (1 + np.exp(k_exp * (img_B - t_exp)))

    return img_B


def computeFlowerCount(images):
    for image in tqdm(images, file=sys.stdout):

        global flower_area_mask
        # Initialize dictionary to store number of flowers per image
        n_flowers = {}
        # Get flower mask for this image
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
        pil_image = PIL.Image.open(image).convert('RGB')
        open_cv_image = np.array(pil_image)
        img = open_cv_image[:, :, ::-1].copy()

        # Highlight flowers
        img_flowers = getFlowerHighlight(img, image, segm_dist_from_zerocross=8)

        # Apply flower area mask
        img_flowers[flower_area_mask == 0] = 0

        # Get number of flowers using blob counter on the B channel
        blobs = blob_doh(img_flowers, max_sigma=5, min_sigma=1)

        # Append result
        n_flowers[image] = blobs

        # Save to object
    flower_count = n_flowers
    flower_count_estimate = n_flowers



if __name__ == "__main__":

    #job_name = sys.argv[1]
    pool = multiprocessing.Pool(6)
    #: Image size
    imsize = (720, 1280)
   
   
    region_matrix = [[0,0,0],[0,1,0],[0,0,0]]

    #: Histograms of the Grayscale images
    hist_G_all = {}
    

    #: Plot mask
    plot_mask = []
    

    #: Plot bounds
    plot_bounds = []
    

    #: Histograms of the b (Lab) channel for all images (only for pixels within the plot)
    hist_b_all = {}
    

    #: Upper limit of B channel for segmentation
    segm_B_upper = 255

    #: Bandwidth used for segmentation on the B channel
    segm_B_lower = 155
   

    #: Threshold for percentage of flower pixels to consider that there are flowers in the field 
    flower_pix_perc_thresh = 0.0001
    
    #: Average histogram of B component
    avg_hist_b = []
   

    #: Mask of the area where flowers will be counted
    flower_area_mask = []
    

    #: Histogram shift for component B of each image
    hist_b_shifts = {}


    #: Dictionary storing the fraction of flower pixels in each image
    flower_pix_perc = {}


    image_filenames = list_files("/home/hduser/plot_images/2016-07-05_1207")
    #print(image_filenames[0])

    processing_start_time = time()

    computePlotMask(image_filenames)
    computeFlowerAreaMask(region_matrix)
    computeHistograms(image_filenames)
    
    computeAverageHistograms()
    computeHistogramShifts(image_filenames)
    computeFlowerPixelsPercentage(image_filenames)
    computeFlowerCount(image_filenames)

    processing_end_time = time() - processing_start_time
    print "SUCCESS: Images procesed in {} seconds".format(round(processing_end_time, 3))
