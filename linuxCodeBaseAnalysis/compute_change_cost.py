import numpy
import sys

# computes the propagation / change cost for a dsm
def calculatePropagationCost( dsm ):
	# make sure that the dsm is a numpy matrix
	matrix = numpy.matrix(dsm)
	
	# make all values greater than 1 into a 1
	matrix[ matrix > 1 ] = 1
	
	# get the size of the matrix
	length = matrix.shape[0]
	
	# initialize the total sum  with the identity matrix
	totalSum = numpy.identity( length )
	
	# copy the matrix so we can raise the matrix to successive powers
	product = numpy.copy( matrix )
	
	# add the initial prduct (matrix^1) to the sum
	totalSum += product
	
	# iterate through the max function call chain length
	for index in range(length - 1):
		
		# increase the matrix by one more power
		product = product * matrix
		
		# make all values greater than 1 into 1
		product[ product > 1 ] = 1
		
		# add the current matrix power to the total sum
		totalSum += product
	
	# make sure there is only 1s and 0s in the matrix	
	totalSum[ totalSum > 1 ] = 1
	
	# calculate the cost ( sum of all values in matrix / ( rows * columns ) )
	cost = 0
	for index in range(length):
		cost += numpy.sum( totalSum[index,] ) 
	cost /= ( length * length )
	
	return cost

# loads a csv file in as a matrix
def loadMatrix( filename ):
	f = open( filename, 'r' )
	array = reduce( lambda a, x1: a + x1, map( lambda x4: map( lambda x2: float( x2.strip() ), x4 ), map( lambda x3: x3.split(','), f ) ), [] )
	length = int( numpy.sqrt( len( array ) ) )
	return numpy.reshape( array, (length, length) )

if len( sys.argv ) > 1:
	
	filename = sys.argv[1]
	
	matrix = loadMatrix( filename )
	
	#print matrix
	print calculatePropagationCost( matrix )
	
else:
	print "please provide a file to process"