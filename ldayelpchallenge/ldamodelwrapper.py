"""
Holds the LDAModelWrapper, which wraps a precalculated LDAModel and provides methods for retrieving probability distributions.
"""
from gensim import corpora, models
import os
import json

class LDAModelWrapper:
	"""Wrapper for loading and interacting with a precalculated LDAModel"""

	def __init__(self, LdaModel, dictionary, userTokens):
		"""
		Initializes a new LDAModelWrapper.

		Arguments:
		LdaModel -- Either a file path string or an ldamodel returned by LDA.
		dictionary -- A dictionary that was used to calculate the ldamodel. Assigns each unique token a unique id
		userTokens -- Dictionary with the user id as key and lists of lists of tokens from all reviews authored by that user as values
		"""
		if type(LdaModel) == str:
			self.ldamodel = models.LdaModel.load(LdaModel)	
		else:
			self.ldamodel = LdaModel
		if type(dictionary) == str:
			with open(dictionary) as f:
				self.dictionary = json.loads(f.read())
		else:
			self.dictionary = dictionary
		self.userTokens = userTokens

	def get_user_posteriors(self, userTokens):
		"""
		Retrieves the topic distributions for all reviews authored by a user.

		Arguments:
		userTokens -- A list of lists of all unique tokens that occur in all reviews authored by a specific user.

		Returns:
		List of lists which each list representing the topic distribution of a review authored by the given user.
		"""
		posteriors = []
		for tokens in userTokens:
			bow = self.dictionary.doc2bow(tokens)
			posteriors.append(self.ldamodel.get_document_topics(bow))
		return posteriors

	def get_all_posteriors(self):
		"""
		Retrieves the topic distributions for all users.

		Returns:
		Dictionary with user ids as keys and the lists returned by self.get_user_posteriors as values
		"""
		posteriors = {}
		for user, tokens in self.userTokens.iteritems():						
			posteriors[user] = self.get_user_posteriors(tokens)		
		return posteriors