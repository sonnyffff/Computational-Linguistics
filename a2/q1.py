#!/usr/bin/env python3
# Student Name: Zijia(Sonny) Chen
# Student Number: 1005983349
# UTORid: chenz347
import math
from collections import Counter
from typing import *

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np
from numpy.linalg import norm

from q0 import stop_tokenize
from wsd import evaluate, load_eval, load_word2vec, WSDToken


def mfs(sent: Sequence[WSDToken], word_index: int) -> Synset:
    """Most frequent sense of a word.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. See the WSDToken class in wsd.py
    for the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The most frequent sense for the given word.
    """
    ### START CODE HERE
    word = sent[word_index].lemma
    synsets = wn.synsets(word)

    return synsets[0]
    # raise NotImplementedError


def lesk(sent: Sequence[WSDToken], word_index: int) -> Synset:
    """Simplified Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    ### START CODE HERE
    best_sense = mfs(sent, word_index)
    best_score = 0
    word = sent[word_index].lemma
    context = sent
    for sense in wn.synsets(word):
        signature = stop_tokenize(sense.definition())
        for e in sense.examples():
            signature.extend(stop_tokenize(e))
        score = len(set(signature) & set(context))

        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense
    # raise NotImplementedError


def lesk_ext(sent: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    ### START CODE HERE
    best_sense = mfs(sent, word_index)
    best_score = 0
    word = sent[word_index].lemma
    context = sent
    for sense in wn.synsets(word):
        signature = stop_tokenize(sense.definition())
        for e in sense.examples():
            signature.extend(stop_tokenize(e))
        # Iterate through sense's hyponyms, holonyms, and meronyms
        new_bag = sense.hyponyms() + sense.member_holonyms() + sense.part_holonyms() + sense.substance_holonyms() \
                  + sense.member_meronyms() + sense.part_meronyms() + sense.substance_meronyms()
        for sense2 in new_bag:
            signature.extend(stop_tokenize(sense2.definition()))
            for e2 in sense2.examples():
                signature.extend(stop_tokenize(e2))

        score = len(set(signature) & set(context))

        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense
    # raise NotImplementedError


def lesk_cos(sent: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    ### START CODE HERE
    best_sense = mfs(sent, word_index)
    best_score = 0
    word = sent[word_index].lemma
    context = sent
    # create a counter to count occurrence of the words
    signature_vec = Counter()
    context_vec = Counter()
    for sense in wn.synsets(word):
        signature = stop_tokenize(sense.definition())
        for e in sense.examples():
            signature.extend(stop_tokenize(e))
        # Iterate through sense's hyponyms, holonyms, and meronyms
        new_bag = sense.hyponyms() + sense.member_holonyms() + sense.part_holonyms() + sense.substance_holonyms() \
                  + sense.member_meronyms() + sense.part_meronyms() + sense.substance_meronyms()
        for sense2 in new_bag:
            signature.extend(stop_tokenize(sense2.definition()))
            for e2 in sense2.examples():
                signature.extend(stop_tokenize(e2))
        # count
        signature_vec.update(signature)
        context_vec.update(context)
        # cosine similarity
        dot = sum(signature_vec[w] * context_vec[w] for w in signature_vec if w in context_vec)
        mag1 = math.sqrt(sum(i ** 2 for i in signature_vec.values()))
        mag2 = math.sqrt(sum(i ** 2 for i in context_vec.values()))
        score = dot / (mag1 * mag2)
        # score = len(set(signature) & set(context))

        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense
    # raise NotImplementedError


def lesk_cos_onesided(sent: Sequence[WSDToken], word_index: int) -> Synset:
    """Extended Lesk algorithm using one-sided cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    ### START CODE HERE
    best_sense = mfs(sent, word_index)
    best_score = 0
    word = sent[word_index].lemma
    context = sent
    # create a counter to count occurrence of the words
    signature_vec = Counter()
    context_vec = Counter()
    for sense in wn.synsets(word):
        signature = stop_tokenize(sense.definition())
        for e in sense.examples():
            signature.extend(stop_tokenize(e))
        # Iterate through sense's hyponyms, holonyms, and meronyms
        new_bag = sense.hyponyms() + sense.member_holonyms() + sense.part_holonyms() + sense.substance_holonyms() \
                  + sense.member_meronyms() + sense.part_meronyms() + sense.substance_meronyms()
        for sense2 in new_bag:
            signature.extend(stop_tokenize(sense2.definition()))
            for e2 in sense2.examples():
                signature.extend(stop_tokenize(e2))
        # does not include words that occur only in the signature
        for w in signature:
            if w not in context:
                signature.remove(w)
        add_set = set()
        for w in context:
            if w not in signature:
                add_set.add(w)
        # count
        signature_vec.update(signature)
        context_vec.update(context)
        for w in add_set:
            signature_vec[w] = 0
        # cosine similarity
        dot = sum(signature_vec[w] * context_vec[w] for w in signature_vec if w in context_vec)
        mag1 = math.sqrt(sum(i ** 2 for i in signature_vec.values()))
        mag2 = math.sqrt(sum(i ** 2 for i in context_vec.values()))
        score = dot / (mag1 * mag2)
        # score = len(set(signature) & set(context))

        if score > best_score:
            best_sense = sense
            best_score = score
    return best_sense
    # raise NotImplementedError



def lesk_w2v(sent: Sequence[WSDToken], word_index: int,
             vocab: Mapping[str, int], word2vec: np.ndarray) -> Synset:
    """Extended Lesk algorithm using word2vec-based cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    To look up the vector for a word, first you need to look up the word's
    index in the word2vec matrix, which you can then use to get the specific
    vector. More directly, you can look up a string s using word2vec[vocab[s]].

    To look up the vector for a *single word*, use the following rules:
    * If the word exists in the vocabulary, then return the corresponding
      vector.
    * Otherwise, if the lower-cased version of the word exists in the
      vocabulary, return the corresponding vector for the lower-cased version.
    * Otherwise, return a vector of all zeros. You'll need to ensure that
      this vector has the same dimensions as the word2vec vectors.

    But some wordforms are actually multi-word expressions and contain spaces.
    word2vec can handle multi-word expressions, but uses the underscore
    character to separate words rather than spaces. So, to look up a string
    that has a space in it, use the following rules:
    * If the string has a space in it, replace the space characters with
      underscore characters and then follow the above steps on the new string
      (i.e., try the string as-is, then the lower-cased version if that
      fails), but do not return the zero vector if the lookup fails.
    * If the version with underscores doesn't yield anything, split the
      string into multiple words according to the spaces and look each word
      up individually according to the rules in the above paragraph (i.e.,
      as-is, lower-cased, then zero). Take the mean of the vectors for each
      word and return that.
    Recursion will make for more compact code for these.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        word_index (int): The index of the target word in the sentence.
        vocab (dictionary mapping str to int): The word2vec vocabulary,
            mapping strings to their respective indices in the word2vec array.
        word2vec (np.ndarray): The word2vec word vectors, as a VxD matrix,
            where V is the vocabulary and D is the dimensionality of the word
            vectors.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    ### START CODE HERE
    raise NotImplementedError


if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()
    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos, lesk_cos_onesided]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
