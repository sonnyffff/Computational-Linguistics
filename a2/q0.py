#!/usr/bin/env python3
# Student Name: Zijia(Sonny) Chen
# Student Number: 1005983349
# UTORid: chenz347

import typing as T
from string import punctuation

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize


def deepest():
    """Find and print the synset with the largest maximum depth along with its
    depth on each of its hyperonym paths.

    Returns:
        None
    """
    ### START CODE HERE
    longest = None
    len_path = -1
    for syn in wn.all_synsets():
        for hp in syn.hypernym_paths():
            if len(hp) > len_path:
                longest = syn
                len_path = len(hp)
    print(longest)
    for hp in longest.hypernym_paths():
        print(len(hp))
    # raise NotImplementedError


def superdefn(s: str) -> T.List[str]:
    """Get the "superdefinition" of a synset. (Yes, superdefinition is a
    made-up word. All words are made up...)

    We define the superdefinition of a synset to be the list of word tokens,
    here as produced by word_tokenize, in the definitions of the synset, its
    hyperonyms, and its hyponyms.

    Args:
        s (str): The name of the synset to look up

    Returns:
        list of str: The list of word tokens in the superdefinition of s

    Examples:
        >>> superdefn('toughen.v.01')
        ['make', 'tough', 'or', 'tougher', 'gain', 'strength', 'make', 'fit']
    """
    ### START CODE HERE
    superdefinition = []
    synset = wn.synset(s)
    temp = [synset] + synset.hypernyms() + synset.hyponyms()

    for synset in temp:
        definitions = synset.definition()
        tokens = word_tokenize(definitions)
        superdefinition.extend(tokens)

    return superdefinition
    # raise NotImplementedError


def stop_tokenize(s: str) -> T.List[str]:
    """Word-tokenize and remove stop words and punctuation-only tokens.

    Args:
        s (str): String to tokenize

    Returns:
        list[str]: The non-stopword, non-punctuation tokens in s

    Examples:
        >>> stop_tokenize('The Dance of Eternity, sir!')
        ['Dance', 'Eternity', 'sir']
    """
    ### START CODE HERE
    tokens = word_tokenize(s)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha()]
    ret = [t for t in tokens if t.lower() not in stop_words]
    return ret
    # raise NotImplementedError


if __name__ == '__main__':
    import doctest
    doctest.testmod()