"""
Holds all needed mathematical functions.
"""
import numpy as np

def mean(posteriors):
	"""
	Calculates the mean distribution for the given distributions.

	Arguments:
	posteriors -- List of lists, where each list represents a probability distribution over topics.

	Returns:
	List that represents the mean distribution.
	"""
	container = [[0]*100]*len(posteriors)
	for index, posterior in enumerate(posteriors):
		for probability in posterior:
			topic = probability[0]
			prob = probability[1]
			container[index][topic] = prob
	a = np.array(container)
	return a.mean(axis=0)

def euclidean(x,y): 
	"""
	Calculate the euclidean distance between two vectors.

	Arguments:
	x -- First vector.
	y -- Second vector.

	Returns:
	Float that represents the euclidean distance.
	"""  
	return np.sqrt(np.sum((x-y)**2))