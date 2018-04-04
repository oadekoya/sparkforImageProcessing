from subprocess import *
from multiprocessing import Pool

# once upon a time there was a strange man named george who wanted
# extract functions from a database in mulit-processed kind of way
# this is his story

def extract_functions( (letters, mode) ):
	cscope = Popen( [ "cscope", "-dL" + mode, letters + ".*" ], stdout=PIPE )
	output = cscope.communicate()[0].strip()
	lines = output.split( '\n' )
	return map( lambda x: x.split( ' ' )[ 1 ] if len( x.strip() ) > 0 else '' , lines )
	
def extract_all_functions():
	
	lower_case_letters = [ chr( ord( 'a' ) + x ) for x in range( 26 ) ]
	upper_case_letters = [ chr( ord( 'A' ) + x ) for x in range( 26 ) ]
	non_letters = [ '[^A-Za-z]' ]

	all_letters = lower_case_letters + upper_case_letters + non_letters
	
	all_pairs = zip( all_letters, [ '2' for x in range( len( all_letters ) ) ] ) 		
	all_pairs += zip( all_letters, [ '3' for x in range( len( all_letters ) ) ] ) 		

	pool = Pool(8)
	
	return pool.map( extract_functions, all_pairs )

functions = reduce( lambda a, x: a.union( set( x ) ), extract_all_functions(), set() )

functions = sorted( list( functions ) )

for function in functions:
		print function
