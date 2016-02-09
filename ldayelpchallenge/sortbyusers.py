"""
Holds the function to sort tokenized reviews by users.
"""
def sortByUsers(tokDocs):
	"""
	Sorts the given tokenized reviews by users.

	Arguments:
	tokDocs -- Dictionary where the review ids are keys and the values are lists representing the tokens in the review referenced by the key.

	Returns:
	Dictionary with user ids as keys and lists of lists as values, where each list represents the tokens of a review authored by that user.
	"""
	userDocs = {}
	for value in tokDocs.itervalues():
		user = value['user']
		tokens = value['tokens']
		try:
			userDocs[user].append(tokens)
		except KeyError:
			userDocs[user] = []
			userDocs[user].append(tokens)
	return userDocs