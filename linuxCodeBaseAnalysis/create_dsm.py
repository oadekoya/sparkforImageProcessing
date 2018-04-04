from subprocess import *   
import numpy
from multiprocessing import Pool


# sorts files with the files above the folders
def fileSort( a, b ):

	a1 = a#[len(folder):]
	b1 = b#[len(folder):]

	b2 = b1.split( '/' );
	a2 = a1.split( '/' );

	length = min( len( a2 ), len( b2 ) )

	for index in range( length - 1 ):
		if b2[ index ] != a2[ index ]:
			return cmp( a2[ index ], b2[ index ] )


	if len( a2 ) == len( b2 ):
		return cmp( a, b )

	if len( a2 ) < len( b2 ):
		return -1
	if len( b2 ) <  len( a2 ):
		return 1

	return cmp( a, b )

# gets the folder a file is in
def getPathFromFile(file):
	bits = file.split( '/' )
	bits = bits[0:-1]
	path = '/'.join( bits ) + '/'
	return path

# calculates the dsm for folders based on the file dsm
def groupDSM( dsm, files, groups ):
	newDSM =  numpy.matrix( numpy.zeros( ( len(groups), len( groups) ) ) ) 
	for index in range(len(files)):
		file = files[ index ]
		path = getPathFromFile( file )
		groupIndex = groups[ path ]
		for index2 in range(len(files)):
			file2 = files[ index2 ]
			path2 = getPathFromFile( file2 )
			groupIndex2 = groups[ path2 ]
			
			newDSM[ groupIndex,  groupIndex2 ] += dsm[ index, index2 ]

	return newDSM

# get the files a function is found in
def getFileForFunction(function):
	function = function.strip()
	cscope = Popen( [ "cscope", "-dL1", function ], stdout=PIPE )
	output = cscope.communicate()[0].strip()
	lines = output.split( '\n' )
	files = map( lambda x: x.split( ' ' )[ 0 ], lines )
	groups = map( getPathFromFile, files )
	return ( function, files )


# add function and files to the functionToFile, fileToFunction, etc dictionaries
def combineFunctionData( function, files, functionToFile, fileToFunction, inFiles, inFunctions ):
	if function:
		if files:
			for f in files:
				if f:
					if not f in fileToFunction:
						fileToFunction[ f ] = []
						inFiles.append( f )
					if not function in functionToFile:
						functionToFile[ function ] = []
						inFunctions.append( function )
					#	else:
					#		if len ( functionToFile[ function ] ) == 1:
					#			print "Duplicate function", function,  functionToFile[ function ][ 0 ]
					#		print "Duplicate function", function, f
					if not function in fileToFunction[ f ]:
						fileToFunction[ f ].append( function )
					if not f in functionToFile[ function ]:
						functionToFile[ function ].append( f )
	return (functionToFile, fileToFunction, inFiles, inFunctions )

# get all the files that another file depends on
def getDependencies( file, fileToFunction, functionToFile, fileToIndex ):
	dependencies = {}
	fromFile = fileToIndex[ file ]
	for function in fileToFunction[ file ]:
		# get all the functions this function calls
		cscope = Popen(["cscope", "-dL2", function], stdout=PIPE)
		output = cscope.communicate()[0]
		lines = output.split( '\n' )
		for line in lines:
			if line.strip():	
				calledFunction = line.split( ' ')[1]
				if calledFunction in functionToFile:
					for f in functionToFile[ calledFunction ]:
						if f in fileToIndex:
							toFile = fileToIndex[ f ]
							if not toFile in dependencies:
								dependencies[ toFile ] = 0
							dependencies[ toFile ] += 1

	return ( fromFile, dependencies )

# converts from a tuple to non tuple
def getDependenciesWithTuple( data ):
	return getDependencies( data[ 0 ], data[ 1 ], data[ 2 ], data[ 3 ] )

# increments value in dsm and returns it
def addResultsToDSM( dsm, fileFrom, fileTo, count ):
	dsm[ fileFrom, fileTo ] += count
	return dsm

# determines the folder each file is in
def group_functions():

	pool = Pool(8)

	print "parse functions"
	
	functions = open( "functions" )
	return pool.map( getFileForFunction, functions )

# gets depedencies 
def get_dependencies(data):
	pool = Pool(8)
	
	print "getting dependencies"
	
	return pool.map ( getDependenciesWithTuple, data )

# group files
def group_files( files ):
	pool = Pool(8)
	print "something"
	return pool.map( getPathFromFile, files )

# write list to a file
def write_list_to_file( elements, filename ):
	f = open( filename, 'w' )
	for line in elements:
		f.write( line + "\n" )
	f.close()

# create csv file for meatrix
def write_matrix_to_file( matrix, filename ):
	f = open( filename, 'w' )
	for row in range( matrix.shape[0] ):
		strs = map( lambda index: str( matrix[row, index] ), range( matrix.shape[0] ) )
		line = ",".join(strs) + "\n"
		#line = reduce ( lambda a, x: x if len( a ) == 0 else a + "," + str( x ), matrix[row,], "" ) + "\n"			
		f.write( line )
	f.close()

#dms = numpy.matrix( numpy.zeros((10,10)) )

#write_matrix_to_file( dms, "tmp.txt" )

#die()

print "Get files for functions"

data = group_functions()

print "combining functions"	

( functionToFile, fileToFunction, files, functions ) = reduce( lambda ( functionToFile, fileToFunction, inFiles, inFunctions), ( function, files ) : combineFunctionData( function, files, functionToFile, fileToFunction, inFiles, inFunctions ), data, ( {},{},[],[] )  )

print "sorting files"

files = sorted( list( set( files ) ), cmp=fileSort )

#files = files[:24]

print "create groups" 

groups = sorted( list( set( group_files(  files ) ) ), cmp=fileSort )	

print "inverting files"

fileToIndex = { files[ k ]: k for k in range( len( files ) ) }

print "inverting groups"

groupToIndex = { groups[ k ]: k for k in range( len( groups ) ) }

print "combining data for dependencies"

data2 = map( lambda file: ( file, fileToFunction, functionToFile, fileToIndex ), files )
	
data = get_dependencies( data2 )	
	
print "construct dsm"

dsm = reduce( lambda dsm, (fileFrom, filesTo) : reduce( lambda d, fileTo: addResultsToDSM( d, fileFrom, fileTo, filesTo[ fileTo ] )  ,filesTo ,dsm ) , data  ,numpy.matrix( numpy.zeros( ( len(files), len(files) ) ) ) )

print "group dsm"

groupedDSM = groupDSM( dsm, files, groupToIndex )

# now we just write the dsm and group dsm file
# as well as the files and groups

write_list_to_file( files, "dsm_files.txt" )
write_list_to_file( groups, "dsm_groups.txt" )

write_matrix_to_file( dsm, "dsm.txt" )
write_matrix_to_file( groupedDSM, "group_dsm.txt" )




