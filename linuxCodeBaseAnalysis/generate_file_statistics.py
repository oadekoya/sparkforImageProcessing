from subprocess import *   
from multiprocessing import Pool
import os
import sys
import re
from pyparsing import nestedExpr

def getFileForFunction(function):
	function = function.strip()
	cscope = Popen( [ "cscope", "-dL1", function ], stdout=PIPE )
	output = cscope.communicate()[0].strip()
	lines = output.split( '\n' )
	files = map( lambda x: x.split( ' ' )[ 0 ], lines )
	return ( function, files )

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
					if not function in fileToFunction[ f ]:
						fileToFunction[ f ].append( function )
					if not f in functionToFile[ function ]:
						functionToFile[ function ].append( f )
	return (functionToFile, fileToFunction, inFiles, inFunctions )

def processFile( filename ):
	# count lines (whitespace, comments?, code )
	# count functions?
	# lines per function?
	# it would be nice if I could diffentiate functions and external code
	# can we strip out multi line comments?
	
	commentLines = 0
	totalLines = 0
	functionLines = 0
	functionCount = 0
	contents = ""
	whitespaceLines = 0
	includeLines = 0

	try:		
		contents = open( filename, 'r' ).read()		
	except:
		pass
	
	contents = contents.replace( "\t", "    " )
	totalLines = len( contents.split( '\n' ) )
	
	# generate statistics about the comments
	# and strip them out of the file
	comments = re.compile( r"(/\*.*?\*/\s*)$", re.S | re.M )
	m = comments.findall( contents )
	for comment in m:
		if comment + "\n" in contents:
			contents = contents.replace( comment + "\n", "" )
		else:
			contents = contents.replace( comment, "" )
		commentLines += len( comment.strip().split( '\n' ) )
		
	# find count and remove all blank lines	
	whitespace = re.findall( r"^(\s*)$", contents, re.M )
	for line in whitespace:
		whitespaceLines += 1
		if len( line ) > 0:
			if "\n" + line + "\n" in contents:
				contents = contents.replace( "\n" + line + "\n", "\n" )
			#elif "\n" + line in contents:
				#contents = contents.replace( "\n" + line , "" ) 
		else:
			contents = contents.replace( "\n\n", "\n" )
		
	# count the number of functions and the number of lines in a function
	results = nestedExpr( opener="{", closer="}" ).scanString( contents )
	for t, s, e in results:
		s = len( contents[:s].split( '\n' ) )
		e = len( contents[:e].split( '\n' ) )
		functionCount += 1
		functionLines += e - s + 2

	# count includes
	includes = re.findall( r"#include", contents )
	for include in includes:
		includeLines += 1

	# return the values
	return { 'filename': filename, 'commentLines': commentLines, 'totalLines': totalLines, 'functionCount': functionCount, 'functionLines': functionLines, 'whitespaceLines': whitespaceLines, 'includeLines': includeLines }
		
	

def processFileList( filename ):
	pool = Pool(8)
	
	f = open( filename, 'r' )
	results = pool.map( processFile, map( lambda filename: os.path.abspath( filename.strip() ), f ) )

	keys = [ ]

	if len( results ) > 0:
		for key in results[ 0 ]:
			if not key == "filename":
				keys.append( key )

	keys = [ 'filename' ] + sorted( keys )

	print ','.join( keys )

	for result in results:
		print ','.join( map( lambda x: str( result[ x ] ), keys ) )
		
	



if len( sys.argv ) > 1:
	
	filename = sys.argv[1]

	processFileList( filename )
	
else:

	processFileList( 'cscope.files' )
