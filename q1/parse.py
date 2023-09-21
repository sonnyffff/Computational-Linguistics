#!/usr/bin/env python3
# Student name: Zijia(Sonny) Chen
# Student number: 1005983349
# UTORid: chenz347
"""Functions and classes that handle parsing"""

from itertools import chain

from nltk.parse import DependencyGraph


class PartialParse(object):
    """A PartialParse is a snapshot of an arc-standard dependency parse

    It is fully defined by a quadruple (sentence, stack, next, arcs).

    sentence is a tuple of ordered pairs of (word, tag), where word
    is a a word string and tag is its part-of-speech tag.

    Index 0 of sentence refers to the special "root" node
    (None, self.root_tag). Index 1 of sentence refers to the sentence's
    first word, index 2 to the second, etc.

    stack is a list of indices referring to elements of
    sentence. The 0-th index of stack should be the bottom of the stack,
    the (-1)-th index is the top of the stack (the side to pop from).

    next is the next index that can be shifted from the buffer to the
    stack. When next == len(sentence), the buffer is empty.

    arcs is a list of triples (idx_head, idx_dep, deprel) signifying the
    dependency relation `idx_head ->_deprel idx_dep`, where idx_head is
    the index of the head word, idx_dep is the index of the dependant,
    and deprel is a string representing the dependency relation label.
    """

    left_arc_id = 0
    """An identifier signifying a left arc transition"""

    right_arc_id = 1
    """An identifier signifying a right arc transition"""

    shift_id = 2
    """An identifier signifying a shift transition"""

    root_tag = "TOP"
    """A POS-tag given exclusively to the root"""

    def __init__(self, sentence):
        # the initial PartialParse of the arc-standard parse
        # **DO NOT ADD ANY MORE ATTRIBUTES TO THIS OBJECT**
        self.sentence = ((None, self.root_tag),) + tuple(sentence)
        self.stack = [0]
        self.next = 1
        self.arcs = []

    @property
    def complete(self):
        """bool: return true iff the PartialParse is complete

        Assume that the PartialParse is valid
        """
        # *** BEGIN YOUR CODE *** #
        # the buffer is empty
        if self.next == len(self.sentence):
            # stack is contains only ROOT
            if len(self.stack) == 1:
                return True
        return False
        # *** END YOUR CODE *** #

    def parse_step(self, transition_id, deprel=None):
        """Update the PartialParse with a transition

        Args:
            transition_id : int
                One of left_arc_id, right_arc_id, or shift_id. You
                should check against `self.left_arc_id`,
                `self.right_arc_id`, and `self.shift_id` rather than
                against the values 0, 1, and 2 directly.
            deprel : str or None
                The dependency label to assign to an arc transition
                (either a left-arc or right-arc). Ignored if
                transition_id == shift_id

        Raises:
            ValueError if transition_id is an invalid id or is illegal
                given the current state
        """
        # *** BEGIN YOUR CODE *** #
        # (idx_head, idx_dep, deprel)
        if transition_id == self.right_arc_id:
            if len(self.stack) < 2:
                raise ValueError("Illegal")
            h_index = self.stack[-2]
            d_index = self.stack.pop(-1)
            self.arcs.append((h_index, d_index, deprel))
        elif transition_id == self.left_arc_id:
            # Root must be the head
            if self.next == len(self.sentence) and len(self.stack) == 2:
                raise ValueError("Illegal")
            d_index = self.stack.pop(-2)
            h_index = self.stack[-1]
            self.arcs.append((h_index, d_index, deprel))
        elif transition_id == self.shift_id:
            # Can't shift from empty buffer
            if self.next == len(self.sentence):
                raise ValueError("Illegal")
            self.stack.append(self.next)
            # reduce buffer
            self.next += 1
        else:
            # should not reach here
            raise ValueError("Invalid id")
        # *** END YOUR CODE *** #

    def get_n_leftmost_deps(self, sentence_idx, n=None):
        """Returns a list of n leftmost dependants of word

        Leftmost means closest to the beginning of the sentence.

        Note that only the direct dependants of the word on the stack
        are returned (i.e. no dependants of dependants).

        Args:
            sentence_idx : refers to word at self.sentence[sentence_idx]
            n : the number of dependants to return. "None" refers to all
                dependants

        Returns:
            deps : The n leftmost dependants as sentence indices.
                If fewer than n, return all dependants. Return in order
                with the leftmost @ 0, immediately right of leftmost @
                1, etc.
        """
        # *** BEGIN YOUR CODE *** #
        deps = []
        if n is None:
            for index in range(0, len(self.sentence)):
                for arc in self.arcs:
                    if arc[0] == sentence_idx and arc[1] == index:
                        deps.append(index)
        else:
            for index in range(0, len(self.sentence)):
                if n != 0:
                    for arc in self.arcs:
                        if arc[0] == sentence_idx and arc[1] == index:
                            deps.append(index)
                            n -= 1
                else:
                    break
        # *** END YOUR CODE *** #
        return deps

    def get_n_rightmost_deps(self, sentence_idx, n=None):
        """Returns a list of n rightmost dependants of word on the stack @ idx

        Rightmost means closest to the end of the sentence.

        Note that only the direct dependants of the word on the stack
        are returned (i.e. no dependants of dependants).

        Args:
            sentence_idx : refers to word at self.sentence[sentence_idx]
            n : the number of dependants to return. "None" refers to all
                dependants

        Returns:
            deps : The n rightmost dependants as sentence indices. If
                fewer than n, return all dependants. Return in order
                with the rightmost @ 0, immediately left of rightmost @
                1, etc.
        """
        # *** BEGIN YOUR CODE *** #
        deps = []
        if n is None:
            for index in range(len(self.sentence), 0, -1):
                for arc in self.arcs:
                    if arc[0] == sentence_idx and arc[1] == index:
                        deps.append(index)
        else:
            for index in range(len(self.sentence), 0, -1):
                if n != 0:
                    for arc in self.arcs:
                        if arc[0] == sentence_idx and arc[1] == index:
                            deps.append(index)
                            n -= 1
                else:
                    break
        # *** END YOUR CODE *** #
        return deps

    def get_oracle(self, graph: DependencyGraph):
        """Given a projective dependency graph, determine an appropriate
        transition

        This method chooses either a left-arc, right-arc, or shift so
        that, after repeated calls to pp.parse_step(*pp.get_oracle(graph)),
        the arc-transitions this object models matches the
        DependencyGraph "graph". For arcs, it also has to pick out the
        correct dependency relationship.
        graph is projective: informally, this means no crossed lines in the
        dependency graph. More formally, if i -> j and j -> k, then:
             if i > j (left-arc), i > k
             if i < j (right-arc), i < k

        You don't need to worry about API specifics about graph; just call the
        relevant helper functions from the HELPER FUNCTIONS section below. In
        particular, you will (probably) need:
         - get_deprel(i, graph), which will return the dependency relation
           label for the word at index i
         - get_head(i, graph), which will return the index of the head word for
           the word at index i
         - get_deps(i, graph), which will return the indices of the dependants
           of the word at index i

        Hint: take a look at get_left_deps and get_right_deps below; their
        implementations may help or give you ideas even if you don't need to
        call the functions themselves.

        *IMPORTANT* if left-arc and shift operations are both valid and
        can lead to the same graph, always choose the left-arc
        operation.

        *ALSO IMPORTANT* make sure to use the values `self.left_arc_id`,
        `self.right_arc_id`, `self.shift_id` for the transition rather than
        0, 1, and 2 directly

        Args:
            graph : nltk.parse.dependencygraph.DependencyGraph
                A projective dependency graph to head towards

        Returns:
            transition, deprel_label : the next transition to take, along
                with the correct dependency relation label; if transition
                indicates shift, deprel_label should be None

        Raises:
            ValueError if already completed. Otherwise you can always
            assume that a valid move exists that heads towards the
            target graph
        """
        if self.complete:
            raise ValueError('PartialParse already completed')
        transition, deprel_label = -1, None
        # *** BEGIN YOUR CODE *** #
        # Left arc
        top_stack = self.stack[-1]
        if len(self.stack) > 1 and get_head(self.stack[-2], graph) == top_stack and \
                all(dep < self.next for dep in get_deps(self.stack[-2], graph)):
            transition = self.left_arc_id
            deprel_label = get_deprel(self.stack[-2], graph)
        # Right arc
        elif len(self.stack) > 1 and get_head(top_stack, graph) == self.stack[-2]and \
                all(dep < self.next for dep in get_deps(top_stack, graph)):
            transition = self.right_arc_id
            deprel_label = get_deprel(top_stack, graph)
        else:
            transition = self.shift_id

        # *** END YOUR CODE *** #
        return transition, deprel_label

    def parse(self, td_pairs):
        """Applies the provided transitions/deprels to this PartialParse

        Simply reapplies parse_step for every element in td_pairs

        Args:
            td_pairs:
                The list of (transition_id, deprel) pairs in the order
                they should be applied
        Returns:
            The list of arcs produced when parsing the sentence.
            Represented as a list of tuples where each tuple is of
            the form (head, dependent)
        """
        for transition_id, deprel in td_pairs:
            self.parse_step(transition_id, deprel)
        return self.arcs


def minibatch_parse(sentences, model, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Note that parse_step may raise a ValueError if your model predicts an
    illegal (transition, label) pair. Remove any such "stuck" partial-parses
    from the list unfinished_parses.

    Args:
        sentences:
            A list of "sentences", where each element is itself a list
            of pairs of (word, pos)
        model:
            The model that makes parsing decisions. It is assumed to
            have a function model.predict(partial_parses) that takes in
            a list of PartialParse as input and returns a list of
            pairs of (transition_id, deprel) predicted for each parse.
            That is, after calling
                td_pairs = model.predict(partial_parses)
            td_pairs[i] will be the next transition/deprel pair to apply
            to partial_parses[i].
        batch_size:
            The number of PartialParse to include in each minibatch
    Returns:
        arcs:
            A list where each element is the arcs list for a parsed
            sentence. Ordering should be the same as in sentences (i.e.,
            arcs[i] should contain the arcs for sentences[i]).
    """
    # *** BEGIN YOUR CODE *** #
    # Initialize a list of partial_parses, one for each sentence in sentences
    partial_parses = [PartialParse(sentence) for sentence in sentences]
    # Initialize a shallow copy of partial_parses called unfinished_parses
    unfinished_parses = partial_parses.copy()
    # While unfinished_parses is not empty do
    while len(unfinished_parses) != 0:
        # Use the first batch_size parses in unfinished_parses as a minibatch
        minibatch = unfinished_parses[0:batch_size]
        steps = model.predict(minibatch)
        # Perform a parse step on each partial parse in the minibatch with its predicted transition
        for partial_parse, step in zip(minibatch, steps):
            try:
                partial_parse.parse_step(step[0], step[1])
            # handle ValueError
            except ValueError:
                unfinished_parses.remove(partial_parse)
            # Remove those parses that are completed from unfinished_parses
            else:
                if partial_parse.complete:
                    unfinished_parses.remove(partial_parse)

    arcs = [parse.arcs for parse in partial_parses]
    # *** END YOUR CODE *** #
    return arcs


# *** HELPER FUNCTIONS (look here!) *** #


def get_deprel(sentence_idx: int, graph: DependencyGraph):
    """Get the dependency relation label for the word at index sentence_idx
    from the provided DependencyGraph"""
    return graph.nodes[sentence_idx]['rel']


def get_head(sentence_idx: int, graph: DependencyGraph):
    """Get the index of the head of the word at index sentence_idx from the
    provided DependencyGraph"""
    return graph.nodes[sentence_idx]['head']


def get_deps(sentence_idx: int, graph: DependencyGraph):
    """Get the indices of the dependants of the word at index sentence_idx
    from the provided DependencyGraph"""
    return list(chain(*graph.nodes[sentence_idx]['deps'].values()))


def get_left_deps(sentence_idx: int, graph: DependencyGraph):
    """Get the arc-left dependants of the word at index sentence_idx from
    the provided DependencyGraph"""
    return (dep for dep in get_deps(sentence_idx, graph)
            if dep < graph.nodes[sentence_idx]['address'])


def get_right_deps(sentence_idx: int, graph: DependencyGraph):
    """Get the arc-right dependants of the word at index sentence_idx from
    the provided DependencyGraph"""
    return (dep for dep in get_deps(sentence_idx, graph)
            if dep > graph.nodes[sentence_idx]['address'])


def get_sentence_from_graph(graph, include_root=False):
    """Get the associated sentence from a DependencyGraph"""
    sentence_w_addresses = [(node['address'], node['word'], node['ctag'])
                            for node in graph.nodes.values()
                            if include_root or node['word'] is not None]
    sentence_w_addresses.sort()
    return tuple(t[1:] for t in sentence_w_addresses)
