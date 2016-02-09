"""
This package was developed in order to run the Laten Dirichlet Allocation (LDA) on a set of reviews
provided by Yelp for the Yelp data challenge.

Reviews must be contained in a single file, with each review being a json object, one review per line:
{"votes": {"funny": 0, "useful": 0, "cool": 0}, "user_id": "---_j-GW5aCBtf62ihHwCw", "review_id": "tbF1KI-PGpXnE34hfIvPgQ", "text": "My son, one of his friends and I went to L8 Night Bingo tonite...", "business_id": "v76uEBa0jkRl8AH28piX4w", "stars": 1, "date": "2013-12-21", "type": "review"}

Reviews are first interpreted as json and filtered by language.
Reviews are then tokenized individually to create a list of unique tokens for each review (reviews are seen as a bag of words by LDA)
Next, a dictionary is build that assigns each unique token in the corpus a uniqure id for easier processing.
With that, a document-term-matrix is build that counts for each review the number of occurrences of each unique token.
Finally, the LDA algorith is run, returning a fitted LDA model that is saved in the LDAModelWrapper.
At the same time, the tokenized reviews are sorted and grouped by users.
Next, the LDAModelWrapper is used to return the topic representation of each review.
With this, we can build a mean vector for each user from all the reviews that user authored.
Now that the users are represented as vectors, we calculate all distances between all users.
Next, percentage (default 0.05) is taken as the cutoff for being a close distance between two users.
Finally, for each user we calculate his neighbors whose distance to the user is smaller than or equal to the cutoff.
These are then saved to the provided target directory.
"""
from LDAsetup import *
import json
from filter_lang import filter_by_language
import sys
from math_helper import *
from ldamodelwrapper import LDAModelWrapper
from sortbyusers import sortByUsers
from get_threshhold import fivePercent
from recommender import Recommender

def run(source, target, num_topics = 100, passes = 20, lang = 'en', distance_measure = euclidean, percentage = 0.05):
	"""
	Main entry point for this package. Contains and executes the whole data pipeline. 

	Arguments:
	source -- The path string to the source file containing all reviews
	target -- The path string to the target directory where the neighbors for all users will be saved

	Keyword arguments:
	num_topics -- The number of topics LDA is supposed to discover (default 100)
	passes -- The number of iterations for the statistical inference algorithm (default 20)
	lang -- The language the reviews shall be sorted by (default 'en')
	distance_measure -- A python function that measures the distance between two vectors in a num_topics-dimensional vector space. 
				Must take two numpy arrays and return a float. (default euclidean)
	percentage -- The cutoff for being a close neighbor, i.e. two users are close if their distance is 
			within the closest percentage percent of all distances (default 0.05) 
	"""
	with open(source) as f:
		all_reviews = []
		for line in f:
			all_reviews.append(json.loads(line))

	reviews = filter_by_language(all_reviews, lang)

	rt = ReviewTokenizer(reviews)
	rt.tokenize()

	db = DictionaryBuilder(rt.tokenized_docs)
	db.build()

	dtmb = DTMBuilder(db.dictionary, db.srcTexts)
	dtmb.build()

	ldaw = LDAWrapper(dtmb.dtm, db.dictionary)
	ldaw.run(num_topics = num_topics, passes = passes)

	modelwrapper = LDAModelWrapper(ldaw.ldamodel, db.dictionary, sortByUsers(rt.tokenized_docs))
	posteriors = modelwrapper.get_all_posteriors()

	means = {}
	for key, value in posteriors.iteritems():
		means[key] = mean(value).tolist()

	x = Recommender(means)
	y = x.calc_distances(distance_measure)

	threshhold = fivePercent(y, percentage)

	for user in means.iterkeys():
		z = x.calc_neighbors(user, distance_measure, threshhold = threshhold)
		if len(target) > 0:
			fileName = target + '/' + user + '.json'
		else:
			fileName = user + '.json'
		with open(fileName, 'w') as g:
			json.dump(z, g) 