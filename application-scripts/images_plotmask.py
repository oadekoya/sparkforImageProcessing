import os
import cv2
import numpy as np
#import PIL.Image
#import PIL.ImageFile
from StringIO import StringIO
from PIL import Image, ImageFile
import copy
import glob
import sys
from pyspark import SparkContext
from pyspark import SparkConf
import skimage.io as io

from tqdm import tqdm

import pyspark

from PIL import ImageFile

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


def userDefinePlot(img, bounds = None, plot = True):
    """
    A square is drawn in the image passed as parameter. The user is
    asked to drag its four corners so that the square completely contains
    the plot under study.

    To reset the original square corner locations: Press R
    To finish: Press Enter

    :param image: The image array that contains the crop
    :param bounds: Optionally user can set up previously the bounds without using GUI
    :return: The four points selected by user and the mask to apply to the image
    """
    # Initial assert
    if not isinstance(img, np.ndarray):
        print("Image is not a numpy array")
        return

    # Function definitions
    def getMask(boundM):
        """
        Get mask from bounds
        :return: Mask in a numpy array
        """
        # Initialize mask
        shapeM = img.shape[1::-1]
        mask = np.zeros(img.shape[:-1])

        # Get boundaries of the square containing our ROI
        minX = max([min([x[0] for x in boundM]), 0])
        minY = max([min([y[1] for y in boundM]), 0])
        maxX = min(max([x[0] for x in boundM]), shapeM[0])
        maxY = min(max([y[1] for y in boundM]), shapeM[1])

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

   # def paintLines(imgP, boundP):
   #     """
   #     Paint lines connected to the points specified in the parameters.
   #     :param imgP: Image where the lines will be painted
   #     :param boundP: Contains the points that connect the four lines
   #     :return: Image with lines painted
   #     """
   #     im = np.copy(imgP)
   #     cv2.line(im, boundP[0], boundP[1], (0, 0, 255), 2)    # Line 1
   #     cv2.line(im, boundP[0], boundP[2], (0, 255, 0), 2)    # Line 2
   #     cv2.line(im, boundP[1], boundP[3], (255, 0, 0), 2)    # Line 3
   #     cv2.line(im, boundP[2], boundP[3], (0, 255, 255), 2)  # Line 4
   #     cv2.imshow(wname, im)

   # def mouseCallback(event, x, y, flags, param):
   #     """
   #     Mouse callback function
   #     :param event: Mouse event
   #     :param x: Mouse X position
   #     :param y: Mouse Y position
   #     :param flags: Mouse flags
   #     :param param: User parameter
   #     """
   #     global pointAttached
   #     #nonlocal pointAttached
   #     # Only execute when left button pressed
   #     if event == cv2.EVENT_LBUTTONDOWN:
   #         if pointAttached == -1:
   #             # Attach closest point to the mouse
   #             maxDistance = 100 ** 2
   #             distances = [(pX - x) ** 2 + (pY - y) ** 2 for pX, pY in bounds]
   #             if min(distances) > maxDistance:
   #                 return
   #             pointAttached = distances.index(min(distances))
   #             # Update bound point
   #             bounds[pointAttached] = (x, y)

   #     elif event == cv2.EVENT_LBUTTONUP:
   #         # Deattach point
   #         pointAttached = -1

   #     if pointAttached >= 0:
   #         bounds[pointAttached] = (x,y)

            # Print lines if point is attached
   #         paintLines(img, bounds)

    # Check if bounds have been provided
   # if isinstance(bounds, list):
   #     if len(bounds) != 4:
   #         print("Bounds length must be 4. Setting up GUI...")
   #     else:
   #         mask = getMask(bounds)
   #         return bounds, mask

    # Get image shape
    shape = img.shape[1::-1]

    # Initialize boudaries
    bounds = [(207,156), (1014, 156), (207, 677), (1014, 677)]

    if plot == False:
        #for flower area
        bounds = [(308, 247), (923, 247), (308, 612), (923, 612)]

    # Window name
    #wname = "Plot identification"

    # Flag to signal if a point is attached to the mouse or not
    #pointAttached = -1

    # Set up window
    #cv2.namedWindow(wname)
    #cv2.setMouseCallback(wname, mouseCallback)
    #paintLines(img, bounds)

    # Start interaction
    #k = 0
    #while (k != 10 and k != 141):  # "Enter" key
        #k = cv2.waitKey(0) & 0xFF

        # R or r = Reset
     #   if k == 114 or k == 82:
    #        bounds = [(10, 10), (shape[0] - 10, 10), (10, shape[1] - 10), (shape[0] - 10, shape[1] - 10)]
    #        paintLines(img, bounds)

    # Destroy window
    #cv2.destroyAllWindows()

    # Get binary mask for the user-selected ROI
    mask = getMask(bounds)

    return bounds, mask





def list_files(images_path):
    """
    returns a list of names (with extension, without full path) of all files
    in folder path
    """
    imageFiles = []
    for name in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, name)):
            imageFiles.append(name)
    return imageFiles


plot_mask_path = "/sparkdata/tmp-dir/"
#plot_mask_path ='/sparkdata/hduser_images/user/hduser/plot_images_processed/'
imfilenames = list_files("/sparkdata/tmp-dir/2016-07-05_1207/")
img_clusters = {}
plot_mask = []
plot_bounds = []
flower_area_mask = []
flower_area_bounds = []
hist_b_all = {}
hist_G_all = {}
avg_hist_b = []
counter = 0
hist_b_shifts = {}
segm_B_upper = 255
segm_B_lower = 155
flower_pix_perc = {}
flower_pix_perc_estimate = {}


def resetImageClusters():
    """
    Sets all images to cluster 0 (i.e. good images)
    """
    # Initial assert
    global imfilenames
    assert len(imfilenames) > 0, "Failed to reset image cluster: No image names loaded"

    # Assign all images to cluster 0
    for imname in imfilenames:
        img_clusters[imname] = 0



def setImagesFilepaths(path, imgsformat="jpg"):
    """
    Set a new list of filenames that will be analyzed by this flower detector. If both parameters are none,
    the function will return without doing anything
    :param imfilenames: List of thimfilenames = []e image filepaths
    :param path: Path containing the images
    :param imgsformat: Format of the image (string) (by default: jpg)
    """

    global imfilenames, imgspath
    if imfilenames is not None:
        assert isinstance(imfilenames, list), "Parameter imfilepaths must be a list of strings"
        assert len(imfilenames) > 0, "Parameter imfilepaths is empty"
        imfilenames = imfilenames

    if path is not None:
        assert os.path.exists(path), "Selected path does not exist"
        assert isinstance(imgsformat, str), "Images format must be a string (e.g. 'jpg', 'tiff', etc)"

        if not path.endswith('/'):
            path += '/'

        imgspath = path

        imfilepaths = glob.glob(path + "*." + imgsformat)
        assert len(imfilepaths) > 0, "No JPG images in the selected path"

        # Sort images filepaths
        imfilepaths = [int(k.split("/")[-1].split(".")[0]) for k in imfilepaths]
        imfilepaths.sort()

        # Store images filepaths
        imfilenames = [str(f) + "." + imgsformat for f in imfilepaths]

        # Get image size
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        pil_image = Image.open(path + imfilenames[0]).convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        imsize = open_cv_image.shape[:2]

    else:
        print("At least one parameter must be not None. Leaving without modifying...")
        return

    # Set all images initially as good
    resetImageClusters()


def setPlotMask(mask, bounds):
    """
    Set mask of the plot under analysis
    :param mask: Mask of the plot
    :param bounds: Bounds of the plot
    """
    global plot_mask, plot_bounds

    # Initial assert
    assert isinstance(mask, np.ndarray), "Parameter 'corners' must be Numpy array"
    #assert mask.shape == imsize, "Mask has a different size"
    assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"

    # Store mask
    plot_mask = mask

    # Store bounds
    plot_bounds = bounds
    


def setFlowerAreaMask(mask, bounds):
    """
    Set mask of the flower area within the plot
    :param mask: Mask of the flower area
    :param bounds: Bounds of the flower area
    """
    global flower_area_mask, flower_area_bounds

    # Initial assert
    assert isinstance(mask, np.ndarray), "Parameter 'corners' must be Numpy array"
    #assert mask.shape == self.imsize, "Mask has a different size"
    assert isinstance(bounds, list) and len(bounds) == 4, "Bounds must be a 4-element list"

    # Store mask
    flower_area_mask = mask

    # Store bounds
    flower_area_bounds = bounds



def setHistogram(hist):
    
    global avg_hist_b

    # Store histogram
    avg_hist_b += hist

    return avg_hist_b




Class ComputeMask:
    def computePlotMask(self, imgspath, path = None, plot_mask_name = "plot_mask"):
        """
        Compute plot mask
        """
        # Trace
        print("Computing plot mask...")
    
        #global plot_mask_path, plot_mask, plot_bounds, flower_area_mask, flower_area_bounds

        # Read an image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        #pil_image = Image.open(imfilenames[0]).convert('RGB')
        #pil_image = Image.open(imgspath + imfilenames[0]).convert('RGB')
        open_cv_image = np.array(imgspath)
        # Convert RGB to BGR
        #print open_cv_image
        #open_cv_image = copy.copy(imgspath[:, :, ::-1])
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        # Ask for user interaction to get the plot
        print("Select plot boundaries. R = Reset boundaries. Enter/Spacebar = finish")
        p_bounds, p_mask = userDefinePlot(open_cv_image, None, True)

        # Ask for user interaction to get the flower area
        print("Select flower area boundaries. R = Reset boundaries. Enter/Spacebar = finish")
        fw_bounds, fw_mask = userDefinePlot(open_cv_image, None,  False)
        # Store mask and bounds

        #global plot_mask_path, plot_mask, plot_bounds, flower_area_mask, flower_area_bounds
        #return p_mask, p_bounds, fw_mask, fw_bounds
        setPlotMask(p_mask, p_bounds)
        setFlowerAreaMask(fw_mask, fw_bounds)

        global plot_mask_path

        # Path assignment
        if path is None:
            path = plot_mask_path

        # Initial assert
        assert os.path.exists(path), "Selected or default path does not exist"

        # Check if path ends with slash
        if not path.endswith('/'):
            path += '/'

        # Check if variables to save are not empty
        if len(plot_mask) == 0 or len(plot_bounds) == 0:
            print("Plot mask is empty. Leaving without saving...")
            return

        if len(flower_area_bounds) == 0 or len(flower_area_mask) == 0:
            print("Flower area mask is empty. Leaving without saving...")
            return

        # Create dictionary to store
        dict_to_save = {"plot_bounds": plot_bounds,
                       "plot_mask": plot_mask,
                       "flower_area_bounds" : flower_area_bounds,
                       "flower_area_mask" : flower_area_mask}
        #return dict_to_save
        #print plot_mask
        np.save(plot_mask_path + plot_mask_name, dict_to_save)
        


def computeHistograms(images):
    """
    Compute the average A and B histograms over all images
    """
    # Trace
    print("Computing histograms...")

    global avg_hist_b, hist_b_all, imfilenames, hist_G_all, plot_mask, counter, hist_b_shifts, segm_B_upper, segm_B_lower, flower_pix_perc, flower_pix_perc_estimate
    # Clear average histogram
    avg_hist_b = np.zeros(256, np.int32)

    # Clear histograms for b channel
    #hist_b_all = {}

    # Get the number of images and store it as number of samples
    nSamples = len(imfilenames)

    # Preliminary check
    if nSamples == 0:
        print("No images found")
        return

   
    # Iterate through all images
    #for image in images:
    #    image_no = 0
    #for fname in tqdm(imfilenames, file=sys.stdout):
        # Read image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
        #pil_image = PIL.Image.open(self.imgspath + fname).convert('RGB')
        #open_cv_image = np.array(pil_image)
    open_cv_image = np.array(images)
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
    hist_b_all[imfilenames[counter]] = hist_b
    hist_G_all[imfilenames[counter]] = hist_G

    return hist_b

        # Accumulate the histograms
    
    #avg_hist_b += hist_b
    
    #avg_hist_b = np.divide(avg_hist_b, nSamples)

    """
    Compute histogram shifts respect to the average histograms for each image
    """

    # Trace
    print("Computing histogram shifts...")


    # Calculate correlation
    correlation_b = np.correlate(hist_b_all[imfilenames[counter]], avg_hist_b, "full")    
    
    # Get the shift on the X axis
    x_shift_b = correlation_b.argmax().astype(np.int8)

    # Append it to the list
    hist_b_shifts[imfilenames[counter]] = x_shift_b


    """
    Compute the percentage of flower pixels in all of the pictures.
    Since we only apply thresholding in the B channel to get the flower pixels,
    we only need the B histogram to get the number of flower pixels in the plot.
    """
    # Trace
    print("Computing flower pixels percentage...")

    # Get number of pixels in the plot
    n_plot_pix = len(plot_mask[plot_mask > 0])

        # Iterate through all images
    #for fname in tqdm(self.imfilenames, file=sys.stdout):
    # Get segmentation parameters
    hist_b = hist_b_all[imfilenames[counter]]
    hist_b_shift = hist_b_shifts[imfilenames[counter]]

    # Calculate number of flower pixels
    n_flower_pixels = np.sum(hist_b[segm_B_lower + hist_b_shift: segm_B_upper + hist_b_shift + 1])

    # Get percentage of flower pixels
    fw_pix_perc = n_flower_pixels / n_plot_pix

    # Store it in dictionary
    flower_pix_perc[imfilenames[counter]] = fw_pix_perc

    # Copy dictionary
    flower_pix_perc_estimate = flower_pix_perc

    counter += 1



def images_to_bytes(rawdata):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    #image_read = rawdata.read()
    #return (rawdata[0], PIL.Image.open(rawdata[1]).convert('RGB') )
    return Image.open(StringIO(rawdata[1])).convert('RGB')



def savePlotMask(path = None, plot_mask_name = "plot_mask"):
    """
    Save plot and flower area masks calculated by computePlotMask as Numpy files
    :param path: Path where the plot mask file will be saved (by default: Same as images' path)
    :param plot_mask_name: (Optional) Filename of the plot mask (by default: "plot_mask")
    """

    global plot_mask_path, plot_mask, plot_bounds, flower_area_mask, flower_area_bounds
    # Path assignment
    if path is None:
        path = plot_mask_path

    # Initial assert
    assert os.path.exists(path), "Selected or default path does not exist"

    # Check if path ends with slash
    if not path.endswith('/'):
        path += '/'

    # Check if variables to save are not empty
    if len(plot_mask) == 0 or len(plot_bounds) == 0:
        print("Plot mask is empty. Leaving without saving...")
        return

    if len(flower_area_bounds) == 0 or len(flower_area_mask) == 0:
        print("Flower area mask is empty. Leaving without saving...")
        return

    # Create dictionary to store
    dict_to_save = {"plot_bounds": plot_bounds,
                    "plot_mask": plot_mask,
                    "flower_area_bounds" : flower_area_bounds,
                    "flower_area_mask" : flower_area_mask}

    # Save/Overwrite dictionary
    #io.imsave(path + plot_mask_name, dict_to_save)
    np.save(path + plot_mask_name, dict_to_save)

#setImagesFilepaths('/sparkdata/tmp-dir/2016-07-05_1207/')

#sc = SparkContext("local[4]", "images_plot_mask")

sc = SparkContext("spark://discus-p2irc-master:7077", "images_plot_mask")

#images_read = (sc.binaryFiles('hdfs://discus-p2irc-master:54310/user/hduser/plot_images/2016-07-05_1207', 12))

images_read = (sc.binaryFiles('hdfs://discus-p2irc-master:54310/user/hduser/plot_images/2016-07-05_1207', 600))

images_bytes = (images_read.map(images_to_bytes))

images_bytes.persist(pyspark.StorageLevel.MEMORY_AND_DISK_SER)

images_mask_computed = images_bytes.foreach(computePlotMask)

#images_histogram_computed = images_bytes.foreach(computeHistograms)

#images_histogram_computed = images_bytes.foreach(computeHistograms)
print plot_mask
print "=========================="
print "images plot mask completed"
print "=========================="
