import numpy as np

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

