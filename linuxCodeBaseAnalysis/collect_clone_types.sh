# get the different clone types for the respective versions


# remove any folder names tmp_blocks recursively
rm -r /cmpt816/tmp_blocks*
# rm -r /cmpt816/data/nicad/clone_types*

# change into the directory that contains nicad
cd /cmpt816/data

# run nicad against the temp directory that contains the particular source filesversion to get respective clone types

	# for type1 exact clones
nicad3 blocks c /cmpt816/tmp type1

	# for type2 blind-renamed clone pairs
nicad3 blocks c /cmpt816/tmp type2

	# for type2 consistently renamed clone pairs
nicad3 blocks c /cmpt816/tmp type2c

	# for type3-1 near-miss exact clone pairs
nicad3 blocks c /cmpt816/tmp type3-1

	# for type3-2 blind-renamed near-miss exact clone pairs
nicad3 blocks c /cmpt816/tmp type3-2

	# for type3-2 consistently renamed near-miss exact clone pairs
nicad3 blocks c /cmpt816/tmp type3-2c 


# make a folder to store the output of the generated clone types
mkdir /cmpt816/data/nicad/clone_types
mkdir /cmpt816/data/nicad/clone_types/tmp_blocks-blind
mkdir /cmpt816/data/nicad/clone_types/tmp_blocks-consistent
mkdir /cmpt816/data/nicad/clone_types/tmp_blocks

mv /cmpt816/tmp_blocks-blind-clones /cmpt816/data/nicad/clone_types/tmp_blocks-blind
mv /cmpt816/tmp_blocks-consistent-clones /cmpt816/data/nicad/clone_types/tmp_blocks-consistent
mv /cmpt816/tmp_blocks-clones /cmpt816/data/nicad/clone_types/tmp_blocks
mv /cmpt816/tmp_blocks* /cmpt816/data/nicad/clone_types


comm -2 -3 <(sort /cmpt816/data/nicad/clone_types/tmp_blocks-blind/tmp_blocks-blind-clones/tmp_blocks-blind-clones-0.00.xml) <(sort /cmpt816/data/nicad/clone_types/tmp_blocks-blind/tmp_blocks-blind-clones/tmp_blocks-blind-clones-0.30.xml) > /cmpt816/data/nicad/clone_types/tmp_blocks-blind/tmp_blocks-blind-clones/blind-clones-diff.txt


comm -2 -3 <(sort /cmpt816/data/nicad/clone_types/tmp_blocks-consistent/tmp_blocks-consistent-clones/tmp_blocks-consistent-clones-0.00.xml) <(sort /cmpt816/data/nicad/clone_types/tmp_blocks-consistent/tmp_blocks-consistent-clones/tmp_blocks-consistent-clones-0.30.xml) > /cmpt816/data/nicad/clone_types/tmp_blocks-consistent/tmp_blocks-consistent-clones/consistent-clones-diff.txt


comm -2 -3 <(sort /cmpt816/data/nicad/clone_types/tmp_blocks/tmp_blocks-clones/tmp_blocks-clones-0.00.xml) <(sort /cmpt816/data/nicad/clone_types/tmp_blocks/tmp_blocks-clones/tmp_blocks-clones-0.30.xml) > /cmpt816/data/nicad/clone_types/tmp_blocks/tmp_blocks-clones/block-clones-diff.txt


