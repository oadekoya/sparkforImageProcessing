# so. stuff and woot
cd /cmpt816/linux

# get the list of versions in the repository (Except 2.6.11 which is not real)
git tag | grep -v rc | grep -v 2.6.11 > /cmpt816/scripts/versions.txt

LAST_VERSION="";

# loop through the versions in the file
while read VERSION
do
    # remove the contents of the working directory
    rm -r *
    
    # checkout the version
    git checkout $VERSION -f

    # get the git logs and place in the linux directory...
    if [ "$LAST_VERSION" = "" ]
    then
        git log $VERSION > /cmpt816/linux/gitlog.txt
    else
        git log $LAST_VERSION..$VERSION > /cmpt816/linux/gitlog.txt
    fi

    #run the script on it
    sh /cmpt816/scripts/run.sh
    
    # move the output
    mv /cmpt816/data /cmpt816/results/$VERSION
    
    LAST_VERSION="$VERSION"
    
done < /cmpt816/scripts/versions.txt
