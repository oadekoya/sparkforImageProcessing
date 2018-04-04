import re
import numpy

f = open( "iclones.txt", 'r' )

cloneType = 0
#
cloneTypes = {"1":{ "uniqueFiles": 0, "totalClones": 0, "size": []},"2":{ "uniqueFiles": 0, "totalClones": 0, "size": []},"3":{ "uniqueFiles": 0, "totalClones": 0, "size": []}}
totalStats = { "uniqueFiles": 0, "totalClones": 0, "size": []}
uniqueFiles = {}

for line in f:
	if line.startswith( '@' ):
		cloneType = line[1:].strip()
		totalStats[ 'totalClones' ] += 1
		cloneTypes[ cloneType ][ "totalClones" ] += 1
	elif not line.startswith( 'CCF' ):
		bits = re.split( "\s*", line.strip() )
		fileName = bits[ 0 ]
		startLine = int( bits[ 1 ] )
		endLine = int( bits[ 2 ] )
		
		if not fileName in uniqueFiles:
			uniqueFiles[ fileName ] = 0
			cloneTypes[ cloneType ][ "uniqueFiles" ] += 1
			totalStats[ "uniqueFiles" ] += 1
		
		uniqueFiles[ fileName ] += 1

		cloneTypes[ cloneType ][ "size" ].append( endLine - startLine + 1 )
		totalStats[ "size" ].append( endLine - startLine + 1 )


print "files, clones, avg size, std, type1, avg size, std, type2, avg size, std, type3, avg size, std";
print str( len( uniqueFiles ) ) + ", ",
print str( totalStats[ 'totalClones' ] ) + ", ",
print str( numpy.mean( totalStats[ 'size' ] ) ) + ", ",
print str( numpy.std( totalStats[ 'size' ] ) ) + ", ",

for t in [ '1', '2', '3' ]:
	print str( cloneTypes[ t ][ 'totalClones' ] ) + ", ",
	print str( numpy.mean( cloneTypes[ t ][ 'size' ] ) ) + ", ",
	if t == '3':
		print str( numpy.std( cloneTypes[ t ][ 'size' ] ) )
	else:
		print str( numpy.std( cloneTypes[ t ][ 'size' ] ) ) + ", ",

f.close()
