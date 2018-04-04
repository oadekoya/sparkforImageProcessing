# March 31, 2017
# Habib Sabiu - Script to copy files recursively from a remote server
#               into the HDFS file system on the local DISCUS cluster

import os
import sys
import stat
import shutil
import getpass
import paramiko
import argparse
import posixpath
import pydoop.hdfs as hdfs

server = ''
username = ''
password = ''

remoterootdir = ''
hdfsrootpath = ''
tempdir = 'tmp-dir/'

imageCount = 0

def rmtree(sftp, remotepath, level=0):
    global imageCount
    for f in sftp.listdir_attr(remotepath):
        rpath = posixpath.join(remotepath, f.filename)
        if stat.S_ISDIR(f.st_mode):
            rmtree(sftp, rpath, level=(level + 1))
        else:
            rpath = posixpath.join(remotepath, f.filename)
            if rpath.endswith(('.jpg', '.tiff', '.tif', '.png', '.JPG', '.TIFF', '.TIF', '.PNG')) and os.path.getsize(rpath) > 0:

                flattenedPath = remotepath.replace("/", "_")
                if flattenedPath.startswith('_'):
                    flattenedPath = flattenedPath[1:]

                localFileName = flattenedPath + f.filename
                localpath = tempdir + localFileName

                sftp.get(rpath, localpath)
                imageCount += 1
                print '[' + str(imageCount) + '] LOCAL: ' + rpath + " ===> " + localpath
                hdfspath = hdfsrootpath + localFileName

                try:
                    hdfs.put(localpath, hdfspath)
                    print '       HDFS: ' + localpath + " ===> " + hdfspath
                    os.remove(localpath)
                except IOError:
                    os.remove(localpath)
                    continue

if __name__ == "__main__":

    # Create arguments to parse
    ap = argparse.ArgumentParser(description="Copy images recursively from remote directory into a flat HDFS directory")
    ap.add_argument("-s", "--servername", required=True, help="Remote server name")
    ap.add_argument("-u", "--username", required=True, help="User name")
    ap.add_argument("-i", "--inputpath", required=True, help="Path to the remote root input directory")
    ap.add_argument("-o", "--outputpath", required=True, help="Path to the desired output HDFS directory.")

    args = vars(ap.parse_args())

    server = args["servername"]
    username = args["username"]
    password = getpass.getpass('Password:')
    
    if password == "":
        print 'No password provided'
        sys.exit()

    remoterootdir = args["inputpath"]
    hdfsrootpath = args["outputpath"]

    if '../' in remoterootdir:
        print 'Please give complete path for the remote input directory'
        sys.exit()

    if not remoterootdir.endswith('/'):
        remoterootdir = remoterootdir + '/'

    if not hdfsrootpath.endswith('/'):
        hdfsrootpath = hdfsrootpath + '/'

    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password)
    sftp = ssh.open_sftp()
    rmtree(sftp, remoterootdir)

    sftp.close()
    ssh.close()

    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

    print '==============================================='
    print '= SUCCESS: Images successfully copied to HDFS ='
    print '==============================================='



