from dataclasses import dataclass
from typing import List, Optional, Dict
from torch.utils.data.dataset import Dataset


@dataclass
class Entity:
    """
    An entity in a training/test example.
    """
    start: int  # start index in the sentence
    end: int  # end index in the sentence
    type: Optional[str] = None  # entity type
    id: int = None  # id in the current training/test example

    def to_tuple(self):
        return self.type, self.start, self.end

    def __hash__(self):
        return hash((self.id, self.start, self.end))


@dataclass
class Argument:
    """
    An argument for an event
    """
    role: str  # role of the argument
    ref_id: str = None  # id in the current training/test example


@dataclass
class Relation:
    """
    An (asymmetric) relation in a training/test example.
    """
    type: str  # relation type
    head: Argument  # head of the relation
    tail: Argument  # tail of the relation

    def to_tuple(self):
        return self.type, self.head.to_tuple(), self.tail.to_tuple()


@dataclass
class Event:
    """
    An Event for bigbio schema.
    {'id': 'PMID-10026184_E1',
    'type': 'BindingToProtein',
    'trigger': {'text': ['Interaction'], 'offsets': [[41, 52]]},
    'arguments': [{'role': 'hasPatient2', 'ref_id': 'PMID-10026184_T4'}
    """
    id: str
    type: str  # event type
    text: str  # trigger in example
    start: int  # start index in the sentence
    end: int  # end index in the sentence
    arguments: List[Argument] = None
    trigger_id: str = None


@dataclass
class Example:
    id: str
    tokens: str  # list of tokens (words)
    input_tokens: str = None
    output_tokens: str = None
    dataset: Optional[Dataset] = None  # dataset this example belongs to
    offset: int = 0  # sentence offset (important for BioNLP data set)
    builder_name: str = ""  # dataset name

    # entity-relation extraction
    entities: List[Entity] = None  # list of entities
    relations: List[Relation] = None  # list of relations
    events: List[Event] = None  # list of events
    nld: Optional[str] = None  # the given natural language description
