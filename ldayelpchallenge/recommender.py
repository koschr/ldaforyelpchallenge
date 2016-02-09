"""
Holds the recommender class which is used for different calculations that are needed to produce recommendations in the end.
"""
import json
import numpy as np

class Recommender():
	"""
	Recommender class. Can find the closest neighbor for a given user as well as calculate all neighbors and all distances between all users.
	"""
	def __init__(self, means):
		"""
		Initializes a new Recommender.

		Arguments:
		means -- A dictionary with the user ids as keys and their mean distribution vectors as values
		"""
		self.means = means

	def findClosest(self, userID, distanceMeasure):	
		"""
		Find the neighbor with the shortest distance to the given user.

		Arguments:
		userID -- The user (string) whose neighbor we want to find.
		distanceMeasure -- A python function that takes two numpy arrays and returns a float, the distance between the two distributions given by the arrays.

		Returns:
		List where first entry represents the distance and the second entry the id of the closest neighbor for the given user. 
		"""	
		toCheck = self.means[userID]
		closest = [float('inf'), None]
		for key, value in self.means.iteritems():
			if key != userID:
				x = distanceMeasure(np.array(toCheck), np.array(value))				
				if  x < closest[0]:
					closest[0] = x
					closest[1] = key
		return closest

	def calc_neighbors(self, userID, distanceMeasure, threshhold = None):
		"""
		Calculates the distances to all other users for a given user by default, or only those who are closer than threshhold if given.

		Arguments:
		userID -- The user id as string for which we want all distances.
		distanceMeasure -- A python function that takes two numpy arrays and returns a float, the distance between the two distributions given by the arrays.
		threshhold -- Float that represents the cutoff for being a close neighbor (default None)

		Returns:
		List of lists, where each list is of the form [distance, id of neighbor].
		"""
		toCheck = self.means[userID]
		neighbors = []
		for key, value in self.means.iteritems():
			if key != userID:
				x = distanceMeasure(np.array(toCheck), np.array(value))
				if(threshhold):
					if(round(x,2) <= threshhold):
						neighbors.append([x, key])
				else:
					neighbors.append([x, key])
		return neighbors

	def calc_distances(self, distanceMeasure):
		"""
		Calculates all distances between all users. Distance values are used as bins and their frequencies are then counted.

		Arguments:
		distanceMeasure -- A python function that takes two numpy arrays and returns a float, the distance between the two distributions given by the arrays.

		Returns:
		A dictionary where the keys are the rounded distances and the values are their frequencies.
		"""
		distances = {}
		for user in self.means.iterkeys():
			for key, value in self.means.iteritems():
				if key != user:
					x = round(distanceMeasure(np.array(self.means[user]), np.array(value)), 2)
					if x in distances:
						distances[x] += 1
					else:
						distances[x] = 1
		return distances