# Word Analogy and Word Debiasing Using Word Embeddings

## Objective
The purpose of this project is to create a word analogy with the help of pre-trained word embeddings. By the end of this project, we can create an algorithm such that if we type "large" -> "larger", then the computer hopefully can perceive "small" -> "smaller".

In addition to that, word debiasing algorithm will also be implemented because some word vectors should always be in a neutral space in a finite word-embedding dimensions, i.e should not be biased towards positive or negative section in word-embedding dimensions.

To build word analogy algorithm, pre-trained word embedding is used. The pre-trained model of word embeddings used in this project can be seen at https://nlp.stanford.edu/projects/glove/. This global vectors of words represents 100-dimension of embedding space.

Below is the example of word analogy produced in this project:

<p align="center">
  <img width="500" height="200" src="https://github.com/marcellusruben/Data_Science_Personal_Project/blob/master/Word_Analogy_and_Word_Debiasing/word_analogy.png">
</p>

## Files
There are two files in this project, which are:
- Word analogy and Word Debiasing.ipynb - The Jupyter Notebook file of this project, which contains step-by-step approach used in this project.
- function.py - Collection of Python function applied in this project.

Since the pre-trained vector embeddings (GloVe) file is too large, then you can see the pre-trained model of word embeddings used in this project at https://nlp.stanford.edu/projects/glove/.
