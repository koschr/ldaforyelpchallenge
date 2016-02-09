"""
Filter reviews by language using the langdetect module.
"""
import json
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def filter_by_language(reviews, lang = 'en'):
    """
    Filter all reviews by language.

    Arguments:
    reviews -- List containing all reviews as strings

    Keyword arguments:
    lang -- The language we want the reviews to be filtered by (default 'en')

    Returns:
    List of filtered reviews containing only reviews written in 'lang' language
    """
    relevant_reviews = []
    for review in reviews:
        review_text = review['text']		
        try:
            language = detect(review_text)
            if language == lang:
                relevant_reviews.append(review)
        except LangDetectException:
            pass
    return relevant_reviews