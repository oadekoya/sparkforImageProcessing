#collects data from a single revision

# set up folder for storing data
rm -r /cmpt816/data
mkdir /cmpt816/data

# make sure we are in the linux directory
cd /cmpt816/linux

# configure the kernel with default options
make defconfig 

# build the kernel and redirect the output so we can use it
make C=2 > /cmpt816/data/output.txt 2>&1

cd /cmpt816/data

# extract files from output
grep CHECK output.txt > files.txt

# extract warnings, errors, and notes
grep "error:\|warning:\|note:" output.txt > errors.txt

# remove CHECK from files and append full path and place in cscope.files
cat files.txt | awk -F ' ' '{print "/cmpt816/linux/" $2}' > cscope.files

# create cscope database
cscope -b -k -q

# extract all functions
python /cmpt816/scripts/extract_functions.py > functions

# create the dsm files
python /cmpt816/scripts/create_dsm.py

# calculate the change cost for the per file and per folder dsms
python /cmpt816/scripts/compute_change_cost.py dsm.txt > file_cc.txt
python /cmpt816/scripts/compute_change_cost.py group_dsm.txt > group_cc.txt

# create file statistics file
python /cmpt816/scripts/generate_file_statistics.py > file_statistics.txt

# copy the linux files to temporary directory
sh /cmpt816/scripts/copy_linux_files.sh

# copy the gitlog to current directory
cp /cmpt816/linux/gitlog.txt /cmpt816/data/gitlog.txt

# run the python script to calculate statistics for git log
python /cmpt816/scripts/process_git_log.py > /cmpt816/data/contributor.txt

# run iclones on the temp folder
cd /cmpt816/iclones
./iclones  -input /cmpt816/tmp/ -output /cmpt816/data/iclones.txt -language c -outformat ccfl

# generate stats
python /cmpt816/scripts/process_iclones.py > iclones_statistics.txt

# run nicad3
rm -r /cmpt816/tmp_blocks*
cd /cmpt816/data
nicad3 blocks c /cmpt816/tmp

# nicad is stupid.. so fix it
mkdir /cmpt816/data/nicad
mv /cmpt816/tmp_blocks* /cmpt816/data/nicad 


