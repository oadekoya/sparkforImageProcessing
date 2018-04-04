# March 31, 2017
# Habib Sabiu - Script to copy files recursively from a regular file system
#               into a flat directory

import os
import sys
import argparse

from shutil import copyfile

if __name__ == "__main__":  

    # Create arguments to parse     
    ap = argparse.ArgumentParser(description="Copy images recursively from local directory into a flat directory")
    ap.add_argument("-i", "--inputpath", required=True, help="Path to the root input directory")     
    ap.add_argument("-o", "--outputpath", required=True, help="Path to the desired output directory.")     

    args = vars(ap.parse_args())

    inputDirPath = args["inputpath"]
    outputDirPath = args["outputpath"]

    if not inputDirPath.endswith('/'):
        inputDirPath = inputDirPath + '/'

    if not outputDirPath.endswith('/'):
        outputDirPath = outputDirPath + '/'

    if '..' in inputDirPath:
        print 'Please give compele path for the input directory'
        sys.exit()

    imageCount = 0

    for subdir, dirs, files in os.walk(inputDirPath):
        dirs.sort()
        files.sort()
        for file in files:
            filePath = os.path.join(subdir, file)
            if filePath.endswith(('.jpg', '.tiff', '.tif', '.png', '.JPG', '.TIFF', '.TIF', '.PNG')) and os.path.getsize(filePath) > 0:
                flattenedPath = subdir.replace("/", "_")
                if flattenedPath.startswith('_'):
                    flattenedPath = flattenedPath[1:]
                src = filePath
                localFileName = flattenedPath + file               
                dst = outputDirPath + localFileName
                copyfile(src, dst)
	        #os.remove(src)
	        imageCount += 1
	        print '[' + str(imageCount) + '] file: ' + localFileName + ' ===> ' + dst + '     Size = ' + str(os.path.getsize(src))

    print '========================================='
    print '= SUCCESS: Images successifully copied  ='
    print '========================================='





