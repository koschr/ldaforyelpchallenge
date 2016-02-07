from setuptools import setup

setup(name='ldaforyelpchallenge',
      version='1.0',
      description='Run Latent Dirichlet Allocation on the reviews provided by Yelp for the Yelp data challenge',
      url='https://github.com/koschr/ldaforyelpchallenge',
      download_url='https://github.com/koschr/ldaforyelpchallenge/tarball/1.0',
      author='CK',
      author_email='test@example.com',
      license='MIT',
      packages=['ldaforyelpchallenge'],
      install_requires=['gensim', 'numpy', 'nltk', 'stop_words', 'langdetect'],
      zip_safe=False)