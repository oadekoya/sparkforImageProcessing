import re
import operator

commitCount = 0
uniqueEmail = {}
uniqueNames = {}
emailToName = {}
domains = {}

f = open( "gitlog.txt", 'r' )

for line in f:
	if re.match( "^Author: ", line ):
		commitCount += 1
		name, address = re.findall( "^Author: (.*?) <([^>]*)>", line )[0];
		domain = address.split('@')
		if not name in uniqueNames:
			uniqueNames[ name ] = 0
		if not address in uniqueEmail:
			uniqueEmail[ address ] = 0
			emailToName[ address ] = name
		if not domain[1] in domains:
			domains[ domain[1] ] = 0
		uniqueNames[ name ] += 1
		uniqueEmail[ address ] += 1
		domains[ domain[1] ] += 1

f.close()

print "commits,contributors,corporations"		
print str(commitCount) +',' + str(len(uniqueNames)) + ',' + str( len(domains))


def writeFile( filename, data ):
	f = open(filename, 'w' )
	f.write( "contributor,count\n" )
	for key, value in data:
		f.write( key + ", " + str(value) + "\n" )
	f.close()

email = sorted( uniqueEmail.items(), key=operator.itemgetter(1) )
names = sorted( uniqueNames.items(), key=operator.itemgetter(1) )
domains =  sorted( domains.items(), key=operator.itemgetter(1) )

writeFile( "contributor_email.txt", email )
writeFile( "contributor_names.txt", names )
writeFile( "contributor_domains.txt", domains )