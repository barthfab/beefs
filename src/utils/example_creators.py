import logging
import itertools
from operator import itemgetter
import os
import random
from pathlib import Path
import numpy as np
from typing import Tuple, List, Dict
from src.utils.example_classes import Event, Argument, Entity, Relation, Example


PATH = Path(__file__).parent.parent.parent.resolve()


def create_input_example(example: Example, nld_args, blocked_entities: str = '', entity_type: bool = False,
                         task: str = 'ee'):
    """
    Get input in augmented natural language, for example:

    Similarly, the induction of [KILLER]/[DR5] mRNA by the cytokine [TNF-alpha]
    was also delayed in cell lines with mutated [STAT1].

    augmentations = [([(type,), (tail.text,role), (...) ], #, #), (...)]
    """
    augmentations = []
    blocked_entities = blocked_entities.split(',')
    if task == 'ee' or task == 're':
        for entity in example.entities:
            if entity.type in blocked_entities:
                continue
            if entity_type:
                augmentations.append(([(entity.type,)], entity.start, entity.end))
            else:
                augmentations.append(([], entity.start, entity.end))

        # insert entity marker in sentence
        example.input_tokens = augment_sentence(list(example.tokens),
                                                augmentations,
                                                nld_args.begin_entity_token,
                                                nld_args.separator_token,
                                                nld_args.relation_separator_token,
                                                nld_args.end_entity_token,
                                                )
    else:
        example.input_tokens = ''.join(example.tokens)


def create_output_example(example: Example, nld_args, blocked_entities: str = '', task: str = 'ee'):
    """
    Get output in augmented natural language, for example:

    Similarly, the [induction|Positive_regulation|KILLER=Theme|TNF-alpha=Cause] of KILLER/DR5 mRNA
    by the cytokine TNF-alpha was also [delayed|Negative_regulation|induction=Theme] in cell lines
    with mutated STAT1.

    augmentations = [([(type,), (tail.text,role), (...) ], #, #), (...)]
    """
    augmentations = []

    # add event annotation for event extraction
    if task == 'ee':
        for event in example.events:
            arguments = [(''.join(event.type),)]
            for argument in event.arguments:
                entity_arg = next((e for e in example.entities if e.id == argument.ref_id), None)
                if not entity_arg:
                    entity_arg = next((e for e in example.events if e.id == argument.ref_id), None)
                if entity_arg:
                    arguments.append((''.join(example.tokens[entity_arg.start:entity_arg.end]), argument.role))
                else:
                    arguments = []
                    continue
            if arguments:
                augmentations.append((arguments, event.start, event.end))

    # add entity annotation for entity recognition
    elif task == 'er':
        blocked_entities = blocked_entities.split(',')
        for entity in example.entities:
            if entity.type in blocked_entities:
                continue
            augmentations.append(([(entity.type,)], entity.start, entity.end))

    # append output sentence of example and insert annotation
    example.output_tokens = augment_sentence(list(example.tokens),
                                             augmentations,
                                             nld_args.begin_entity_token,
                                             nld_args.separator_token,
                                             nld_args.relation_separator_token,
                                             nld_args.end_entity_token, )


def single_prompt_parser(examples, nld_args):
    """
    builds input prompt for model.
    text never ends in a trailing space, which causes worse performance due to how the
    API splits text into tokens.
    :param examples: List[Example]: the last example is the training sentence
    :param nld_args: Configuration composed by Hydra.
    :return: str: the input prompt for the model

    default example follows

    Convert this text:

    Example: Similarly, the induction of [KILLER]/[DR5] mRNA by the cytokine [TNF-alpha] was also
    delayed in cell lines with mutated [STAT1].
    Output: Similarly, the [induction|Positive_regulation|KILLER=Theme|TNF-alpha=Cause] of
    KILLER/DR5 mRNA by the cytokine TNF-alpha was also [delayed|Negative_regulation|induction=Theme]
    in cell lines with mutated STAT1.

    ###

    Example: This [nim1]-dependent inhibition of the [wee1] protein kinase can be reversed readily
    in vitro by treatment with a protein phosphatase.
    Output:
    """

    # prompt header
    if nld_args.head:
        prompt = f'{nld_args.head}\n\n'
    else:
        prompt = ''

    if len(examples) > 1:
        for example in examples[:-1]:

            # add examples
            prompt = prompt + nld_args.example_head + example.input_tokens + '\n'
            prompt = prompt + nld_args.output_head + example.output_tokens + '\n'

            # add separator
            if nld_args.example_separator:
                prompt = prompt + '\n' + nld_args.example_separator + '\n\n'
            else:
                prompt = prompt + '\n'

    # add training sentence
    if nld_args.output_head.endswith(' '):
        prompt = prompt + nld_args.example_head + examples[-1].input_tokens + '\n' + nld_args.output_head[:-1]
    else:
        prompt = prompt + nld_args.example_head + examples[-1].input_tokens + '\n' + nld_args.output_head

    return prompt


def single_example_parser(prompt, example_labels, nld_args, task):
    examples = []

    example_types = []
    for example in example_labels:
        if task == 'ee':
            example_types.extend([e.type for e in example.events])
        elif task == 'er':
            example_types.extend([e.type for e in example.entities])
        elif task == 're':
            example_types.extend([e.type for e in example.relations])
    example_types = list(set(example_types))

    predicted_entities, wrong_reconstruction, reconstructed_sentence = \
        parse_output_sentence_char(example_tokens=example_labels[-1].tokens,
                                   output_sentence=prompt,
                                   nld_args=nld_args)

    for guid, predicted_entity in enumerate(predicted_entities):
        event_name, tags, start, end = predicted_entity
        if len(tags) == 0 or len(tags[0]) > 1:
            # we do not have a tag for the entity type
            format_error = True
            continue
        else:
            entity_type = tags[0][0].strip()
            arguments = tags[1:]
        if entity_type in example_types:
            event_arguments = []
            for argument in arguments:
                if len(argument) == 2:
                    arg_name, arg_type = argument
                    event_arguments.append(Argument(role=arg_type,
                                                    ref_id=arg_name
                                                    ))
                else:
                    argument_error = True
                    continue
            examples.append(Event(
                id=f'E{guid}',
                type=entity_type,
                text=event_name,
                start=start,
                end=end,
                arguments=event_arguments,
                trigger_id=f'T{guid}'
            ))
        elif entity_type in example_types:
            examples.append(Entity(
                start=start,
                end=end,
                type=entity_type,
                id=f'T{guid}',
            ))
        elif entity_type in example_types:
            examples.append(Relation(
                type=entity_type,
                head=None,
                tail=None,
            ))
        else:
            type_error = True

    if task == 'ee':
        for guid, event in enumerate(examples):
            for argument in event.arguments:
                arg_ref = [e for e in examples if e.text.strip() == argument.ref_id.strip()
                           and e.id != event.id]
                if not arg_ref:
                    arg_ref = [e for e in examples if
                               example_labels[-1].tokens[e.start:e.end].strip() == argument.ref_id.strip()
                               and e.id != event.id]
                if not arg_ref:
                    arg_ref = [e for e in example_labels[-1].entities if
                               example_labels[-1].tokens[e.start:e.end].strip() == argument.ref_id.strip()]
                if arg_ref:
                    min_event = min(arg_ref, key=lambda x: min(
                        filter(lambda i: i > 0, [int(x.start) - event.end, event.start - int(x.end)]),
                        default=float("inf")))

                    if min_event.id.startswith('E'):
                        argument.ref_id = min_event.id
                    else:
                        argument.ref_id = min_event.id.split('_')[-1]

                else:
                    argument_error = True
    return examples


# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
def parse_output_sentence_char(example_tokens: str, output_sentence: str, nld_args) -> Tuple[list, bool, str]:
    """
        Parse an output sentence in augmented language and extract inferred entities, events, relations and tags.
        Return a pair (predicted_entities, wrong_reconstruction), where:
        - each element of predicted_entities is a tuple (entity_name, tags, start, end)
            - entity_name (str) is the name as extracted from the output sentence
            - tags is a list of tuples, obtained by |-splitting the part of the entity after the entity name
            - this entity corresponds to the tokens example.tokens[start:end]
            - note that the entity_name could differ from ' '.join(example.tokens[start:end]), if the model was not
              able to exactly reproduce the entity name, or if alignment failed
        - wrong_reconstruction (bool) says whether the output_sentence does not match example.tokens exactly

        An example follows.

        example.tokens:
        'Tolkien wrote The Lord of the Rings'

        output_sentence:
        [ Tolkien | person ] wrote [ The Lord of the Rings | book | author = Tolkien ]

        output predicted entities:
        [
            ('Tolkien', [('person',)], 0, 1),
            ('The Lord of the Rings', [('book',), ('author', 'Tolkien')], 2, 7)
        ]
        """
    output_tokens = []
    unmatched_predicted_entities = []
    entity_stack = []  # stack of the entities we are extracting from the output sentence
    # this is a list of lists [start, state, entity_name_tokens, entity_other_tokens]
    # where state is "name" (before the first | separator) or "other" (after the first | separator)
    tokens = list(output_sentence)

    for token in tokens:
        if len(token) == 0:
            continue

        elif nld_args['begin_entity_token'] in token:
            # begin entity
            start = len(output_tokens)
            entity_stack.append([start, "name", [], []])

        elif nld_args['end_entity_token'] in token and len(entity_stack) > 0:
            # end entity
            start, state, entity_name_tokens, entity_other_tokens = entity_stack.pop()
            entity_name = ''.join(entity_name_tokens).strip()
            end = len(output_tokens)

            tags = []

            # split entity_other_tokens by |
            splits = [
                list(y) for x, y in itertools.groupby(entity_other_tokens, lambda z: z == nld_args['separator_token'])
                if not x
            ]

            if state == "other" and len(splits) > 0:
                for x in splits:
                    tags.append(tuple(''.join(x).split(nld_args['relation_separator_token'])))

            unmatched_predicted_entities.append((entity_name, tags, start, end))

        else:
            # a normal token
            if len(entity_stack) > 0:
                # inside some entities
                if nld_args['separator_token'] in token:
                    x = entity_stack[-1]

                    if x[1] == "name":
                        # this token marks the end of name tokens for the current entity
                        x[1] = "other"
                    else:
                        # simply add this token to entity_other_tokens
                        x[3].append(token)

                else:
                    is_name_token = True

                    for x in reversed(entity_stack):
                        # check state
                        if x[1] == "name":
                            # add this token to entity_name_tokens
                            x[2].append(token)

                        else:
                            # add this token to entity_other tokens and then stop going up in the tree
                            x[3].append(token)
                            is_name_token = False
                            break

                    if is_name_token:
                        output_tokens.append(token)

            else:
                # outside
                output_tokens.append(token)

    # check if we reconstructed the original sentence correctly, after removing all spaces
    wrong_reconstruction = (''.join(output_tokens) != ''.join(example_tokens))
    reconstructed_sentence = ''.join(output_tokens)
    # now we align self.tokens with output_tokens (with dynamic programming)
    cost = np.zeros((len(example_tokens) + 1, len(output_tokens) + 1))  # cost of alignment between tokens[:i]
    # and output_tokens[:j]
    best = np.zeros_like(cost, dtype=int)  # best choice when aligning tokens[:i] and output_tokens[:j]

    for i in range(len(example_tokens) + 1):
        for j in range(len(output_tokens) + 1):
            if i == 0 and j == 0:
                continue

            candidates = []

            # match
            if i > 0 and j > 0:
                candidates.append(
                    ((0 if example_tokens[i - 1] == output_tokens[j - 1] else 1) + cost[i - 1, j - 1], 1))

            # skip in the first sequence
            if i > 0:
                candidates.append((1 + cost[i - 1, j], 2))

            # skip in the second sequence
            if j > 0:
                candidates.append((1 + cost[i, j - 1], 3))

            chosen_cost, chosen_option = min(candidates)
            cost[i, j] = chosen_cost
            best[i, j] = chosen_option

    # reconstruct best alignment
    matching = {}

    i = len(example_tokens) - 1
    j = len(output_tokens) - 1

    while i >= 0 and j >= 0:
        chosen_option = best[i + 1, j + 1]

        if chosen_option == 1:
            # match
            matching[j] = i
            i, j = i - 1, j - 1

        elif chosen_option == 2:
            # skip in the first sequence
            i -= 1

        else:
            # skip in the second sequence
            j -= 1

    # update predicted entities with the positions in the original sentence
    predicted_entities = []

    for entity_name, entity_tags, start, end in unmatched_predicted_entities:
        new_start = None  # start in the original sequence
        new_end = None  # end in the original sequence

        for j in range(start, end):
            if j in matching:
                if new_start is None:
                    new_start = matching[j]

                new_end = matching[j]

        if new_start is not None:
            # predict entity
            entity_tuple = (entity_name, tuple(entity_tags), new_start, new_end + 1)
            predicted_entities.append(entity_tuple)

    return predicted_entities, wrong_reconstruction, reconstructed_sentence


# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
def augment_sentence(tokens: List[str], augmentations: List[Tuple[List[tuple], int, int]], begin_entity_token: str,
                     sep_token: str, relation_sep_token: str, end_entity_token: str) -> str:
    """
    Augment a sentence by adding tags in the specified positions.

    Args:
        tokens: Tokens of the sentence to augment.
        augmentations: List of tuples (tags, start, end).
        begin_entity_token: Beginning token for an entity, e.g. '['
        sep_token: Separator token, e.g. '|'
        relation_sep_token: Separator token for relations, e.g. '='
        end_entity_token: End token for an entity e.g. ']'

    An example follows.

    tokens:
    ['Tolkien', 'was', 'born', 'here']

    augmentations:
    [
        ([('person',), ('born in', 'here')], 0, 1),
        ([('location',)], 3, 4),
    ]

    output augmented sentence:
    [ Tolkien | person | born in = here ] was born [ here | location ]
    """
    # sort entities by start position, longer entities first
    augmentations = list(sorted(augmentations, key=lambda z: (z[1], -z[2])))

    # check that the entities have a tree structure (if two entities overlap, then one is contained in
    # the other), and build the entity tree
    root = -1  # each node is represented by its position in the list of augmentations, except that the root is -1
    entity_tree = {root: []}  # list of children of each node
    current_stack = [root]  # where we are in the tree

    for j, x in enumerate(augmentations):
        tags, start, end = x
        if any(augmentations[k][1] < start < augmentations[k][2] < end for k in current_stack):
            # tree structure is not satisfied!
            logging.warning(f'Tree structure is not satisfied! Dropping annotation {x}')
            continue

        while current_stack[-1] >= 0 and \
                not (augmentations[current_stack[-1]][1] <= start <= end <= augmentations[current_stack[-1]][2]):
            current_stack.pop()

        # add as a child of its father
        entity_tree[current_stack[-1]].append(j)

        # update stack
        current_stack.append(j)

        # create empty list of children for this new node
        entity_tree[j] = []

    return ''.join(expand_tokens(
        tokens, augmentations, entity_tree, root, begin_entity_token, sep_token, relation_sep_token, end_entity_token
    ))


# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
def expand_tokens(tokens: List[str], augmentations: List[Tuple[List[tuple], int, int]],
                  entity_tree: Dict[int, List[int]], root: int,
                  begin_entity_token: str, sep_token: str, relation_sep_token: str, end_entity_token: str) \
        -> List[str]:
    """
    Recursively expand the tokens to obtain a sentence in augmented natural language.

    Used in the augment_sentence function below (see the documentation there).
    """
    new_tokens = []
    root_start, root_end = augmentations[root][1:] if root >= 0 else (0, len(tokens))
    i = root_start  # current index

    for entity_index in entity_tree[root]:
        tags, start, end = augmentations[entity_index]

        # add tokens before this entity
        new_tokens += tokens[i:start]

        # expand this entity
        new_tokens.append(begin_entity_token)
        new_tokens += expand_tokens(tokens, augmentations, entity_tree, entity_index,
                                    begin_entity_token, sep_token, relation_sep_token, end_entity_token)

        for tag in tags:
            if tag[0]:
                # only append tag[0] if it is a type, otherwise skip the type
                new_tokens.append(sep_token)
                new_tokens.append(tag[0])

            for x in tag[1:]:
                new_tokens.append(relation_sep_token)
                new_tokens.append(x)

        new_tokens.append(end_entity_token)
        i = end

    # add tokens after all entities
    new_tokens += tokens[i:root_end]

    return new_tokens


def cal_offset(offset, offset_sentence, sentence):
    s_t = 0
    if offset_sentence.startswith(' '):
        s_t += 1
    if offset_sentence.startswith(' \n'):
        s_t += 1
    if offset_sentence.startswith('\n'):
        s_t += 1
    offset += len(sentence) + s_t
    return offset


def parse_example(sentence, example_id, events, entities, relations, offset: int = 0) -> Example:
    example_entities = []
    example_events = []
    example_relations = []

    # find all entities, relations and events that belong to the example sentence
    if entities:
        entities = [ta for ta in entities if ta['offsets'][0][0] >= offset
                    and ta['offsets'][0][1] <= offset + len(sentence)]
    if events:
        events = [ta for ta in events if ta['trigger']['offsets'][0][0] >= offset
                  and ta['trigger']['offsets'][0][1] <= offset + len(sentence)]
    if relations:
        events = [ta for ta in events if ta['trigger']['offsets'][0][0] >= offset
                  and ta['trigger']['offsets'][0][1] <= offset + len(sentence)]

    # add all entities in the given sentence
    for entity in entities:
        example_entities.append(Entity(start=entity['offsets'][0][0] - offset,
                                       end=entity['offsets'][0][1] - offset,
                                       type=entity['type'],
                                       id=entity['id']))

    # add all events in the given sentence
    for event in events:
        example_arguments = []
        for argument in event['arguments']:
            example_arguments.append(Argument(
                role=argument['role'],
                ref_id=argument['ref_id']
            ))
        example_events.append(Event(
            id=event['id'],
            type=event['type'],
            text=event['trigger']['text'],
            start=event['trigger']['offsets'][0][0] - offset,
            end=event['trigger']['offsets'][0][1] - offset,
            arguments=example_arguments,
        ))

    # add all relations in the given sentence
    for relation in relations:
        example_relations.append(Relation(
            type=relation['type'],
            head=relation['arg1_id'],
            tail=relation['arg2_id'],
        ))

    return Example(
        id=example_id,
        tokens=sentence,
        entities=example_entities,
        events=example_events,
        relations=example_relations,
        offset=offset,
    )


def sort_examples_by_event_type(examples, event_types, event_type_addition) -> Dict[str, List]:
    """
    sorts the examples by the given event types.
    :param examples: input examples for the prompt
    :param event_types: the event type to be sorted to
    :param event_type_addition: a list of event types that every example have to contain
    :return: Dict[List]: the key is the name of the event and the value is a list of examples
    """
    sorted_examples = {}
    # crawl through all examples
    for example in examples:
        example_event_types = set([event.type for event in example.events])
        if not example_event_types:
            # add example
            try:
                sorted_examples['None'].append(example)
            except:
                sorted_examples['None'] = [example]
            continue

        # sort by given event type
        for event_type in event_types:
            # determine if example contains an event with the given event_type
            if event_type in example_event_types:
                # if all the example just has to contain a single event type
                if event_type_addition == 'all':
                    # add example
                    try:
                        sorted_examples[event_type].append(example)
                    except:
                        sorted_examples[event_type] = [example]
                else:
                    # get all event types the sentence has to contain to be used
                    event_type_additions = set(event_type_addition.split(','))
                    if example_event_types >= event_type_additions:
                        # add example
                        try:
                            sorted_examples[event_type].append(example)
                        except:
                            sorted_examples[event_type] = [example]

    return sorted_examples


def built_eval_doc_sorted(results: Tuple, event_types: List, arg_finder: int = 0, nested_offset: int = 20, example_offset: int = 0):
    entity_offset = 1000
    output_lines = []

    # logging stats
    format_error = 0
    type_error = 0
    arg_not_found_error = 0
    tag_len_error = 0

    sorted_events, example = results
    for guid, events in enumerate(sorted_events):
        # create event trigger
        trigger = create_event_trigger(event=events[0],
                                       example=example,
                                       example_offset=example_offset,
                                       idx=guid,
                                       event_types=event_types,
                                       entity_offset=entity_offset)
        if trigger == -1:
            format_error += 1
            continue
        elif trigger == -2:
            type_error += 1
            continue
        else:
            output_lines.append(trigger)

        banned_entities = []
        banned_events = {}

        for ev_idx, event in enumerate(events):
            event_name, tags, rel_start, rel_end = event
            string_args = ''
            tag_types = {}

            # create an event line for every found event in output string
            for tag in tags[1:]:
                if len(tag) == 2:
                    tag_name, tag_type = tag
                    if tag_type in tag_types.keys():
                        tag_types[tag_type] += 1
                    else:
                        tag_types[tag_type] = 1
                    # check if argument is an entity
                    entity_argument = [e for e in example.entities if
                                       "".join(example.tokens[e.start:e.end]).strip() == tag_name.strip()
                                       and e not in banned_entities]

                    if entity_argument:
                        arg = argument_finder(event=False,
                                              arg_finder=arg_finder,
                                              arguments=entity_argument,
                                              rel_start=rel_start,
                                              rel_end=rel_end,
                                              example=example,
                                              tags=tags,
                                              event_name=event_name,
                                              tag_name=tag_name,
                                              tag_type=tag_type)

                        # append argument
                        banned_entities.append(arg)
                        string_args = create_argument(event=False,
                                                      arg=arg,
                                                      string_args=string_args,
                                                      tag_types=tag_types,
                                                      tag_type=tag_type)
                        continue

                    unfiltered_event_argument = [e for e in sorted_events if
                                                 "".join(example.tokens[e[0][2]:e[0][3]]).strip() == tag_name.strip()]
                    event_argument = [e for e in unfiltered_event_argument
                                      if e not in [e for e in banned_events.keys()
                                                   if banned_events[e] >= 0]]

                    if event_argument:
                        arg = argument_finder(event=True,
                                              arg_finder=arg_finder,
                                              arguments=event_argument,
                                              rel_start=rel_start,
                                              rel_end=rel_end,)

                        #check for banned event
                        try:
                            banned_events[arg[0]] -= 1
                        except KeyError:
                            banned_events[arg[0]] = len(arg) - 1

                        string_args = create_argument(event=True,
                                                      arg=arg,
                                                      string_args=string_args,
                                                      tag_types=tag_types,
                                                      tag_type=tag_type,
                                                      entity_offset=entity_offset,
                                                      banned_events=banned_events,
                                                      example_offset=example_offset,
                                                      nested_offset=nested_offset,
                                                      sorted_events=sorted_events)
                    else:
                        arg_not_found_error += 1
                        continue
                else:
                    tag_len_error += 1
                    continue
            # create event line with the corresponding argument
            output_lines.append(
                f'E{entity_offset + ev_idx + nested_offset * guid + example_offset}\t{tags[0][0]}:T{entity_offset + guid + example_offset}{string_args}\n')
    return output_lines, example_offset


def built_eval_doc(results: List, event_types: List, arg_finder: int = 0):
    entity_offset = 400
    output_lines = []

    # logging stats
    format_error = 0
    type_error = 0
    arg_not_found_error = 0
    tag_len_error = 0

    example_offset = 0

    # all predictions from all sentences
    for pred_tuple in results:
        pred_events, example = pred_tuple
        # parse all events only once for the output file
        for event in pred_events:
            event_name, tags, rel_start, rel_end = event

            # check for format error
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the entity type
                format_error += 1
                continue

            # check for type error
            if tags[0][0].strip() in event_types or not event_types:

                # add offset to start and end value
                start = rel_start + example.offset
                end = rel_end + example.offset

                # create an event trigger line for every event found
                output_lines.append(
                    f'T{entity_offset + pred_events.index(event) + example_offset}\t{tags[0][0]} {start} {end}\t{event_name}\n')
            else:
                type_error += 1
                continue

            string_args = ''
            tag_types = {}

            # create an event line for every found event in output string
            for tag in tags[1:]:
                if len(tag) == 2:
                    tag_name, tag_type = tag
                    if tag_type in tag_types.keys():
                        tag_types[tag_type] += 1
                    else:
                        tag_types[tag_type] = 1
                    # check if argument is an entity
                    entity_argument = [e for e in example.entities if
                                       "".join(example.tokens[e.start:e.end]).strip() == tag_name.strip()]
                    event_argument = [e for e in pred_events if
                                      "".join(example.tokens[e[2]:e[3]]).strip() == tag_name.strip()]
                    if entity_argument:
                        if arg_finder == 0:
                            arg = closest_arg_finder(entity_argument, rel_start, rel_end)
                        elif arg_finder == 1:
                            arg = furthest_arg_finder(entity_argument, rel_start, rel_end)
                        elif arg_finder == 2:
                            new_arg_finder = [e for e in example.events if
                                              tags[0][0].strip() == e.type and
                                              rel_start == e.start and
                                              rel_end == e.end and
                                              event_name == "".join(e.text)]
                            arg = perfect_arg_finder(new_arg_finder, tag_name, tag_type, example.entities, example.tokens)
                        else:
                            arg = random_arg_finder(entity_argument)

                        # append argument
                        if tag_types[tag_type] < 2:
                            string_args += " " + tag_type + ':' + arg.id.split('_')[-1]
                        else:
                            string_args += " " + tag_type + str(tag_types[tag_type]) + ':' + arg.id.split('_')[-1]
                    elif event_argument:
                        string_args = create_event_argument(string_args=string_args,
                                                            event_argument=event_argument,
                                                            rel_start=rel_start,
                                                            rel_end=rel_end,
                                                            tag_types=tag_types,
                                                            tag_type=tag_type,
                                                            arg_finder=arg_finder,
                                                            entity_offset=entity_offset,
                                                            pred_events=pred_events,
                                                            example_offset=example_offset)
                    else:
                        arg_not_found_error += 1
                        continue
                else:
                    tag_len_error += 1
                    continue
            # create event line with the corresponding argument
            output_lines.append(
                f'E{entity_offset + pred_events.index(event) + example_offset}\t{tags[0][0]}:T{entity_offset + pred_events.index(event)+ example_offset}{string_args}\n')
        example_offset += len(example.events)


def del_duplicates(output_lines):
    for line in output_lines:
        if line.startswith('T'):
            z = line.split(f'\t')
            # delete all trigger lines that are duplicates
            if len([k for k in output_lines if k.startswith('T') and z[1] in k]) >= 2:
                for en in [k for k in output_lines if k.startswith('T') and z[1] in k]:
                    if en == line:
                        continue
                    for guid, event_lines in enumerate(output_lines):
                        if event_lines.startswith('E') and en.split(f'\t')[0] in event_lines:
                            output_lines[guid] = event_lines.replace(en.split(f'\t')[0], z[0])
                    output_lines.remove(en)
            # remove ghost trigger (trigger with no event)
            if not [k for k in output_lines if z[0] in k and k.startswith('E')]:
                output_lines.remove(line)
                continue
        # delete all events that have a ghost event as argument (argument id with no existing event)
        elif line.startswith('E'):
            themes = line.split('Theme')
            cause = line.split('Cause:')
            # check for max. two themes
            if len(themes) > 2:
                if themes[1].startswith(':E'):
                    if not [k for k in output_lines if k.startswith(themes[1][1:5])]:
                        output_lines.remove(line)
                        continue
                if themes[2].startswith(':E'):
                    if not [k for k in output_lines if k.startswith(themes[2][1:5])]:
                        output_lines.remove(line)
                        continue
            # check for one theme
            elif len(themes) > 1 and themes[1].startswith(':E'):
                if not [k for k in output_lines if k.startswith(themes[1][1:5])]:
                    output_lines.remove(line)
                    continue
            # check for cause
            if len(cause) > 1 and cause[1].startswith('E'):
                if not [k for k in output_lines if k.startswith(cause[1][0:4])]:
                    output_lines.remove(line)
                    continue
    return output_lines


def event_arg_finder(arguments: List, event: Tuple, events_lines: List):
    print(-1)


def closest_arg_finder(arguments, rel_start: int, rel_end: int):
    if type(arguments[0]) is not Entity:
        min_arg = min(arguments,
                      key=lambda x:
                      min(filter(lambda i: i > 0,
                                 [int(x[0][2]) - rel_end,
                                  rel_start - int(x[0][3])]),
                          default=float("inf")))
    else:
        min_arg = min(arguments,
                      key=lambda x:
                      min(filter(lambda i: i > 0,
                                 [int(x.start) - rel_end,
                                  rel_start - int(x.end)]),
                          default=float("inf")))
    return min_arg


def furthest_arg_finder(arguments, rel_start: int, rel_end: int):
    if type(arguments[0]) is not Entity:
        min_arg = min(arguments,
                      key=lambda x:
                      min(filter(lambda i: i > 0,
                                 [rel_end - int(x[0][2]), int(x[0][3]) - rel_start]),
                          default=float("inf")))
    else:
        min_arg = min(arguments,
                      key=lambda x:
                      min(filter(lambda i: i > 0,
                                 [rel_end - int(x.start), int(x.end) - rel_start]),
                          default=float("inf")))
    return min_arg


def perfect_arg_finder(argument_events, tag_name: str, tag_type: str, example_entities: list = None, sentence: str = ""):
    for argument_event in argument_events:
        for arg in argument_event.arguments:
            if arg.role == tag_type:
                new_arg = [a for a in example_entities if arg.ref_id == a.id]
                if new_arg:
                    if sentence[new_arg[0].start:new_arg[0].end] == tag_name:
                        return new_arg[0]


def random_arg_finder(arguments):
    if type(arguments[0]) is Entity:
        x = random.choice(arguments)
        return x
    else:
        return random.choice(arguments)


def sort_nested_events(pred_events):
    sorted_events = {}
    for guid, event in enumerate(pred_events):
        try:
            sorted_events[event].append(event)
        except KeyError:
            sorted_events[event] = [event]
    x = list(sorted_events.values())
    x.sort(key=len)
    return x


def write_eval_file(output_lines, log_path, doc_id):
    output_lines = del_duplicates(output_lines)
    output_lines.sort()
    try:
        os.mkdir(os.path.join(PATH, log_path))
    except FileExistsError:
        pass
    try:
        os.mkdir(Path.joinpath(PATH, log_path, 'evaluated'))
    except FileExistsError:
        pass
    try:
        os.mkdir(Path.joinpath(PATH, log_path, 'orig'))
    except FileExistsError:
        pass
    with open(f'{Path.joinpath(PATH, log_path, "orig")}/{doc_id}.a2', 'a') as f:
        f.writelines(output_lines)


def argument_finder(event: bool, arg_finder, arguments, rel_start, rel_end, example: Example = None,
                    tags: list = None, event_name: str = "", tag_name: str = "", tag_type: str = ""):
    if event:
        if arg_finder == 0:
            arg = closest_arg_finder(arguments, rel_start, rel_end)
        elif arg_finder == 1:
            arg = furthest_arg_finder(arguments, rel_start, rel_end)
        elif arg_finder == 2:
            arg = closest_arg_finder(arguments, rel_start, rel_end)
        else:
            arg = random_arg_finder(arguments)
    else:
        if arg_finder == 0:
            arg = closest_arg_finder(arguments, rel_start, rel_end)
        elif arg_finder == 1:
            arg = furthest_arg_finder(arguments, rel_start, rel_end)
        elif arg_finder == 2:
            new_arg_finder = [e for e in example.events if
                              tags[0][0].strip() == e.type and
                              rel_start == e.start and
                              rel_end == e.end and
                              event_name == "".join(e.text)]
            arg = perfect_arg_finder(new_arg_finder, tag_name, tag_type, example.entities, example.tokens)
        else:
            arg = random_arg_finder(arguments)
    return arg


def create_argument(event: bool, arg, string_args: str, tag_types: dict, tag_type: str,
                    entity_offset: int = 400, banned_events: dict = None, example_offset: int = 0, nested_offset: int = 20 , sorted_events: list = None):
    if event:
        # create event argument string
        if tag_types[tag_type] < 2:
            string_args += " " + tag_type + ':' + f'E{entity_offset + banned_events[arg[0]] + nested_offset * sorted_events.index(arg) + example_offset}'
        else:
            string_args += " " + tag_type + str(tag_types[
                                                    tag_type]) + ':' + f'E{entity_offset + banned_events[arg[0]] + nested_offset * sorted_events.index(arg) + example_offset}'
    else:
        # append entity argument
        if tag_types[tag_type] < 2:
            string_args += " " + tag_type + ':' + arg.id.split('_')[-1]
        else:
            string_args += " " + tag_type + str(tag_types[tag_type]) + ':' + arg.id.split('_')[-1]
    return string_args


def create_event_trigger(event, example, example_offset: int, idx: int, event_types: list = None,
                         entity_offset: int = 1000):
    event_name, tags, rel_start, rel_end = event

    # check for format error
    if len(tags) == 0 or len(tags[0]) > 1:
        # we do not have a tag for the entity type
        return -1

    # check for type error
    if tags[0][0].strip() in event_types or not event_types:

        # add offset to start and end value
        start = rel_start + example.offset
        end = rel_end + example.offset

        # create an event trigger line for every event found
        trigger = f'T{entity_offset + idx + example_offset}\t{tags[0][0]} {start} {end}\t{event_name}\n'
    else:
        return -2

    return trigger


def collate_fn(data):
    example, nld = data[0]
    return example['prompt'], example['examples'], nld
