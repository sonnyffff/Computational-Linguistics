#!/usr/bin/env python3
# Student Name: NAME
# Student Number: NUMBER
# UTORid: ID

import typing as T
from collections import defaultdict

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import torch
from torch import Tensor
from torch.linalg import norm

from tqdm import tqdm, trange

from q1 import mfs
from wsd import (batch_evaluate, load_bert, run_bert, load_eval, load_train,
                 WSDToken)


def gather_sense_vectors(corpus: T.List[T.List[WSDToken]],
                         batch_size: int = 32) -> T.Dict[str, Tensor]:
    """Gather sense vectors using BERT run over a corpus.

    As with A1, it is much more efficient to batch the sentences up than it is
    to do one sentence at a time, and you can further improve (~twice as fast)
    if you sort the corpus by sentence length first. We've therefore started
    this function out that way for you, but you may implement the code in this
    function however you like.

    The procedure for this function is as follows:
    * Use run_bert to run BERT on each batch
    * Go through all of the WSDTokens in the input batch. For each one, if the
      token has any synsets assigned to it (check WSDToken.synsets), then add
      the BERT output vector to a list of vectors for that sense (**not** for
      the token!).
    * Once this is done for all batches, then for each synset that was seen
      in the corpus, compute the mean of all vectors stored in its list.
    * That yields a single vector associated to each synset; return this as
      a dictionary.

    The run_bert function will handle tokenizing the batch for BERT, including
    padding the tokenized sentences so that each one has the same length, as
    well as converting it to a PyTorch tensor that lives on the GPU. It then
    runs BERT on it and returns the output vectors from the top layer.

    An important point: the tokenizer will produce more tokens than in the
    original input, because sometimes it will split one word into multiple
    pieces. BERT will then produce one vector per token. In order to
    produce a single vector for each *original* word token, so that you can
    then use that vector for its various synsets, you will need to align the
    output tokens back to the originals. You will then sometimes have multiple
    vectors for a single token in the input data; take the mean of these to
    yield a single vector per token. This vector can then be used like any
    other in the procedure described above.

    To provide the needed information to compute the token-word alignments,
    run_bert returns an offset mapping. For each token, the offset mapping
    provides substring indices, indicating the position of the token in the
    original word (or [0, 0] if the token doesn't correspond to any word in the
    original input, such as the [CLS], [SEP], and [PAD] tokens). You can
    inspect the returned values from run-bert in a debugger and/or try running
    the tokenizer on your own test inputs. Below are a couple examples, but
    keep in mind that these are provided purely for illustrative purposes
    and your actual code isn't to call the tokenizer directly itself.
        >>> from wsd import load_bert
        >>> load_bert()
        >>> from wsd import TOKENIZER as tknz
        >>> tknz('This is definitely a sentence.')
        {'input_ids': [101, 1188, 1110, 5397, 170, 5650, 119, 102],
         'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0],
         'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
        >>> out = tknz([['Multiple', ',', 'pre-tokenized', 'sentences', '!'], \
                        ['Much', 'wow', '!']], is_split_into_words=True, \
                        padding=True, return_offsets_mapping=True)
        >>> out.tokens(0)
        ['[CLS]', 'Multiple', ',', 'pre', '-', 'token', '##ized', 'sentences',
         '!', '[SEP]']
        >>> out.tokens(1)
        ['[CLS]', 'Much', 'w', '##ow', '!', '[SEP]', '[PAD]', '[PAD]', '[PAD]',
        '[PAD]']
        >>> out['offset_mapping']
        [[[0, 0], [0, 8], [0, 1], [0, 3], [3, 4], [4, 9], [9, 13], [0, 9],
         [0, 1], [0, 0]],
         [[0, 0], [0, 4], [0, 1], [1, 3], [0, 1], [0, 0], [0, 0], [0, 0],
         [0, 0], [0, 0]]]

    Args:
        corpus (list of list of WSDToken): The corpus to use.
        batch_size (int): The batch size to use.

    Returns:
        dictionary mapping synsets IDs to Tensor: A dictionary that can be used to
        retrieve the (PyTorch) vector for a given sense.
    """
    corpus = sorted(corpus, key=len)
    synset_vectors = {}
    for batch_n in trange(0, len(corpus), batch_size, desc='gathering',
                          leave=False):
        batch = corpus[batch_n:batch_n + batch_size]
        ### START CODE HERE
        batch_lemma = []
        for s in range(len(batch)):
            lemmas = []
            for t in range(len(batch[s])):
                lemmas.append(batch[s][t].lemma)
            batch_lemma.append(lemmas)

        bert_outputs, offset_mapping = run_bert(batch_lemma)
        # original word index
        for s, sentence in enumerate(batch):
            token_vec = torch.zeros(len(bert_outputs[0][0]))
            num_vec = 0
            token_index = 0
            for v in range(len(bert_outputs[0])):
                # [0, 0]

                if offset_mapping[s][v][1] == 0:
                    pass
                # *[0, 1], [1, 3]
                elif offset_mapping[s][v][1] != len(sentence[token_index].lemma):
                    token_vec += bert_outputs[s][v]
                    num_vec += 1
                elif offset_mapping[s][v][1] == len(sentence[token_index].lemma):
                    token_synsets = sentence[token_index].synsets
                    token_vec += bert_outputs[s][v]
                    num_vec += 1
                    # multiple vectors for a single token in the input data
                    token_vec /= num_vec
                    if token_synsets:
                        for synset in token_synsets:
                            if synset not in synset_vectors:
                                synset_vectors[synset] = [token_vec]
                            else:
                                synset_vectors[synset].append(token_vec)
                    token_vec = torch.zeros(len(bert_outputs[0][0]))
                    num_vec = 0
                    # original word index
                    token_index += 1

    for key in synset_vectors:
        mean = torch.mean(torch.stack(synset_vectors[key]), dim=0)
        synset_vectors[key] = mean

    return synset_vectors
        # raise NotImplementedError


def bert_1nn(batch: T.List[T.List[WSDToken]],
             indices: T.Iterable[T.Iterable[int]],
             sense_vectors: T.Mapping[str, Tensor]) -> T.List[T.List[Synset]]:
    """Find the best sense for specified words in a batch of sentences using
    the most cosine-similar sense vector.

    See the docstring for gather_sense_vectors above for examples of how to use
    BERT. You will need to run BERT on the input batch and associate a single
    vector for each input token in the same way. Once you've done this, you can
    compare the vector for the target word with the sense vectors for its
    possible senses, and then return the sense with the highest cosine
    similarity.

    In case none of the senses have vectors, return the most frequent sense
    (e.g., by just calling mfs(), which has been imported from q1 for you).

    **IMPORTANT**: When computing the cosine similarities and finding the sense
    vector with the highest one for a given target word, do not use any loops.
    Implement this aspect via matrix-vector multiplication and other PyTorch
    ops.

    Args:
        batch (list of list of WSDToken): The batch of sentences containing
            words to be disambiguated.
        indices (list of list of int): The indices of the target words in the
            batch sentences.
        sense_vectors: A dictionary mapping synset IDs to PyTorch vectors, as
            generated by gather_sense_vectors(...).

    Returns:
        pred: The predictions of the correct sense for the given words.
    """
    ### START CODE HERE
    batch_lemma = []
    token_vecs = {}
    ret = []
    for i in range(len(batch)):
        temp = []
        for j in range(len(batch[i])):
            temp.append(batch[i][j].lemma)
        batch_lemma.append(temp)
    bert_outputs, offset_mapping = run_bert(batch_lemma)

    indices = [list(i) for i in indices]

    for s, sentence in enumerate(batch):
        token_vec = torch.zeros(len(bert_outputs[0][0]))
        num_vec = 0
        token_index = 0
        temp = []
        for v in range(len(bert_outputs[0])):
            # [0, 0]
            if offset_mapping[s][v][1] == 0:
                pass

            # *[0, 1], [1, 3]
            elif offset_mapping[s][v][1] != len(sentence[token_index].lemma):
                token_vec += bert_outputs[s][v]
                num_vec += 1
            elif offset_mapping[s][v][1] == len(sentence[token_index].lemma):
                token_synsets = sentence[token_index].synsets
                token_vec += bert_outputs[s][v]
                num_vec += 1
                # multiple vectors for a single token in the input data
                token_vec /= num_vec
                token_vecs[token_index] = token_vec
                for i in indices[s]:
                    if i == token_index:
                        senses_vec = []
                        senses = []
                        for sense in wn.synsets(sentence[token_index].lemma): # CHECK THIS
                            if sense.name() in sense_vectors:
                                senses_vec.append(sense_vectors[sense.name()])
                                senses.append(sense)
                        if senses_vec:
                            sense_vectors_matrix = torch.stack(senses_vec)
                            similarity_scores = torch.nn.functional.cosine_similarity(sense_vectors_matrix,
                                                                                      token_vec)
                            best_sense_index = torch.argmax(similarity_scores)
                            best_sense = senses[best_sense_index]
                        else:
                            best_sense = mfs(sentence, i)
                        temp.append(best_sense)
                token_vec = torch.zeros(len(bert_outputs[0][0]))
                num_vec = 0
                # original word index
                token_index += 1
        if len(temp) != 0:
            ret.append(temp)
    return ret




    #
    # # original word index
    # for i, sentence in enumerate(batch):
    #     ass_vec = torch.zeros(len(bert_outputs[0][0]))
    #     num_vec = 0
    #     token_index = 0
    #     for j in range(len(bert_outputs[0])):
    #         if token_index > len(sentence) - 1:
    #             break
    #         if offset_mapping[i][j][1] != 0 and offset_mapping[i][j][1] != len(sentence[token_index].lemma):
    #             ass_vec += bert_outputs[i][j]
    #             num_vec += 1
    #         elif offset_mapping[i][j][1] != 0 and offset_mapping[i][j][1] == len(sentence[token_index].lemma):
    #             token_synsets = sentence[token_index].synsets
    #             ass_vec += bert_outputs[i][j]
    #             num_vec += 1
    #             # multiple vectors for a single token in the input data
    #             ass_vec /= num_vec
    #             # record vec for the token in the batch
    #             token_vec[sentence[token_index].lemma] = ass_vec
    #
    #             ass_vec = torch.zeros(len(bert_outputs[0][0]))
    #             num_vec = 0
    #             # original word index
    #             token_index += 1
    # indices = [list(i) for i in indices]
    # for i, sentence in enumerate(batch):
    #     ret.append([])
    #     for t, token in enumerate(sentence):
    #         if t in indices[i]:
    #             if len(sense_vectors) != 0 and any(sense.name() in sense_vectors for sense in wn.synsets(token.lemma)):
    #                 senses_vec = []
    #                 senses = []
    #                 for sense in wn.synsets(token.lemma):
    #                     if sense.name() in sense_vectors:
    #                         senses_vec.append(sense_vectors[sense.name()])
    #                         senses.append(sense)
    #                 sense_vectors_matrix = torch.stack(senses_vec)
    #                 # print(token_vec.keys(), token)
    #                 # print(sentence)
    #                 similarity_scores = torch.nn.functional.cosine_similarity(sense_vectors_matrix, token_vec[token.lemma])
    #                 best_sense_index = torch.argmax(similarity_scores)
    #                 best_sense = senses[best_sense_index]
    #                 ret[i].append(best_sense)
    #
    #             else:
    #                 # none of the senses have vectors
    #                 ret[i].append(mfs(batch[i], t))
    # return ret
    # raise NotImplementedError


if __name__ == '__main__':
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        tqdm.write(f'Running on GPU: {torch.cuda.get_device_name()}.')
    else:
        tqdm.write('Running on CPU.')

    with torch.no_grad():
        load_bert()
        train_data = load_train()
        eval_data = load_eval()

        sense_vecs = gather_sense_vectors(train_data)
        batch_evaluate(eval_data, bert_1nn, sense_vecs)
