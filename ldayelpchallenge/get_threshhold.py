"""
Holds the function that calculates the cutoff for close distances.
"""
def fivePercent(distances, percentage = 0.05):
	"""
	Calculates the cutoff by which a distance is considered close.

	Arguments:
	distances -- A dictionary with the distances as keys and their frequency as values

	Keyword arguments:
	percentage -- Float where the cutoff should be (i.e. 0.05 for the closest 5 percent of all distances) (default 0.05)

	Returns:
	Float that markes the cutoff distance.
	"""
	total = 0
	for value in distances.itervalues():
		total += value
	five = total * percentage
	tmp = 0
	fiver = 1
	dist = sorted([float(key) for key in distances.iterkeys()])
	for dst in dist:
		tmp += distances[dst]
		if(tmp >= five):
			fiver = dst
			break
	return fiver