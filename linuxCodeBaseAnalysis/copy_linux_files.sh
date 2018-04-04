# create a temporary folder all the source files
rm -r /cmpt816/tmp
mkdir /cmpt816/tmp

# copy the folder structure to temporary folder
cd /cmpt816/linux
find . -type d > /cmpt816/scripts/dirs.txt
cd /cmpt816/tmp
xargs mkdir -p < /cmpt816/scripts/dirs.txt
rm /cmpt816/scripts/dirs.txt

# copy the files to temporary folder
cat /cmpt816/data/cscope.files | awk -F '/cmpt816/linux/' '{ print $2}' | xargs -I % cp /cmpt816/linux/% /cmpt816/tmp/%