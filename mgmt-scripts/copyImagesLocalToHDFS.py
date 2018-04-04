# March 31, 2017
# Habib Sabiu - Script to copy files recursively from a regular file system
#               into the HDFS file system on the local DISCUS cluster


import os
import pydoop.hdfs as hdfs
import argparse
import sys

if __name__ == "__main__":  
   
    # Create arguments to parse     
    ap = argparse.ArgumentParser(description="Copy images recursively from local directory into a flat HDFS directory")
    ap.add_argument("-i", "--inputpath", required=True, help="Path to the root local directory")     
    ap.add_argument("-o", "--outputpath", required=True, help="Path to the desired HDFS directory.")     

    args = vars(ap.parse_args())

    localInputDirPath = args["inputpath"]
    hdfsOutputDirPath = args["outputpath"]

    if not localInputDirPath.endswith('/'):
        localInputDirPath = localInputDirPath + '/'

    if not hdfsOutputDirPath.endswith('/'):
        hdfsOutputDirPath = hdfsOutputDirPath + '/'
    if '..' in localInputDirPath:
        print 'Please give complete path for the input directory'
        sys.exit()

    imageCount = 0

    for subdir, dirs, files in os.walk(localInputDirPath):
        dirs.sort()
        files.sort()
        for file in files:
            filePath = os.path.join(subdir, file)
            if (filePath.endswith(('.jpg', '.tiff', '.tif', '.png', '.JPG', '.TIFF', '.TIF', '.PNG')) and os.path.getsize(filePath) > 0):
                flattenedPath = subdir.replace("/", "_")
                if flattenedPath.startswith('_'):
                    flattenedPath = flattenedPath[1:]
                hdfsFileName = flattenedPath + file               
                hdfs_path = hdfsOutputDirPath + hdfsFileName

                try:
		    hdfs.put(filePath, hdfs_path)
		    imageCount += 1
		    print '[' + str(imageCount) + '] file: ' + hdfsFileName + ' ===> ' + hdfs_path + '   Size = ' + str(os.path.getsize(filePath))
		    #os.remove(filePath)
    	        except IOError:
		    #os.remove(filePath)
		    continue
                
    print '======================================================================='
    print '= SUCCESS: All files successifully moved from local folder to HDFS    ='
    print '======================================================================='

