#import random
import datasets

from typing import Any, Dict, Optional, Tuple, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from segtok import segmenter

from src.utils.example_classes import Example
from src.utils.example_creators import create_output_example, create_input_example, parse_example, cal_offset, sort_examples_by_event_type
from src.utils.biomed_dataset import SingleBioEventDataset
from pathlib import Path

PATH = Path(__file__).parents[3].resolve()


class SingleDataset(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            data_set,
            seed,
            nld: Dict = None,
            entity_type: bool = False,
            blocked_entities: str = '',
            split: str = 'train',
            learning_method: str = None,
            no_event_threshold: float = 0.2,
            example_size: int = 0,
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = False,
            events_only: int = 0,
            event_types: str = 'all',
            skip_oos_examples: bool = False,

    ):
        super().__init__()
        self.seed = seed
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.bigbio_path = Path.joinpath(PATH, 'biomedical', 'bigbio', 'biodatasets', self.hparams.data_set)
        self.event_types = event_types
        if nld:
            self.nld = nld['nld']
        else:
            self.nld = {'begin_entity_token': '[',
                        'separator_token': '|',
                        'relation_separator_token': '=',
                        'end_entity_token': ']',
                        'query_separator_token': ':',
                        'output_head': 'Output: ',
                        'example_head': 'Example: ',
                        'head': 'Convert this text:',
                        'example_separator': '###',
                        'replace': False,
                        'skip_example_with_special_token': False}



    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        datasets.load_dataset(str(self.bigbio_path), name=f"{self.hparams.data_set}_bigbio_kb", split=self.hparams.split)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # get examples for all datasets separately

            complete_dataset = datasets.load_dataset(str(self.bigbio_path),
                                                     name=f"{self.hparams.data_set}_bigbio_kb",
                                                     split=self.hparams.split)

            complete_train_dataset = datasets.load_dataset(str(self.bigbio_path),
                                                           name=f"{self.hparams.data_set}_bigbio_kb",
                                                           split='train')

            example, entity_type, event_type, relation_type = self.get_all_examples(complete_dataset)

            train_example, _, train_event_type, _ = self.get_all_examples(complete_train_dataset)

            sorted_train_examples = sort_examples_by_event_type(train_example, train_event_type, 'all')

            self.data_train = SingleBioEventDataset(data=example,
                                                    train_data=sorted_train_examples,
                                                    nld=self.nld,
                                                    example_size=self.hparams.example_size,
                                                    learning_method=self.hparams.learning_method,
                                                    no_event_threshold=self.hparams.no_event_threshold,
                                                    seed=self.seed)
            self.data_test = SingleBioEventDataset(data=example,
                                                   train_data=sorted_train_examples,
                                                   nld=self.nld,
                                                   example_size=self.hparams.example_size,
                                                   learning_method=self.hparams.learning_method,
                                                   no_event_threshold=self.hparams.no_event_threshold,
                                                   seed=self.seed)
            self.data_val = SingleBioEventDataset(data=example,
                                                  train_data=sorted_train_examples,
                                                  nld=self.nld,
                                                  example_size=self.hparams.example_size,
                                                  learning_method=self.hparams.learning_method,
                                                  no_event_threshold=self.hparams.no_event_threshold,
                                                  seed=self.seed)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    @staticmethod
    def collate_fn(data):
        prompt, example = data[0]
        return prompt, example

    def get_all_examples(self, complete_dataset) -> Tuple[List[Example], List[str], List[str], List[str]]:
        """
        downloads all given datasets from huggingface. Creates an example class for
        every sentence. An example class contains an input example sentence with
        given annotations and an output example sentence depending on the task to
        perform.
        :param complete_dataset: all biomedical datasets
        :return:
            List[Example]: All example sentences from given data sets
            List[str]: All given entity types in the dataset
            List[str]: All given event types in the dataset
            List[str]: All given relation types in the dataset
        """
        examples = []
        event_types = []
        relation_types = []
        entity_types = []

        # collect all examples
        for dataset in tqdm(complete_dataset, desc=f'load dataset'):

            # get all event types
            for event in dataset['events']:
                if event['type'] not in event_types:
                    event_types.append(event['type'])
            # get all entity types
            for entity in dataset['entities']:
                if entity['type'] not in entity_types:
                    entity_types.append(entity['type'])
            # get all relation types
            for relation in dataset['relations']:
                if relation['type'] not in relation_types:
                    relation_types.append(relation['type'])

            if self.event_types == all:
                valid_event_types = event_types
            else:
                valid_event_types = self.event_types.split(',')

            for passage in dataset['passages']:
                # set passage offset
                s_t = 0
                passage_text = passage['text'][0]

                sentences = segmenter.split_single(passage_text)

                # make example out of all given sentences
                for guid, sentence in enumerate(sentences):
                    # skip empty sentence
                    if len(sentence) == 0:
                        continue
                    if self.nld['skip_example_with_special_token']:
                        if self.nld['begin_entity_token'] in sentence or self.nld['end_entity_token'] in sentence:
                            s_t = cal_offset(s_t, passage_text.split(sentence)[-1], sentence)
                            continue
                    if self.nld['replace']:
                        if self.nld['begin_entity_token'] in sentence or self.nld['end_entity_token'] in sentence:
                            sentence = sentence.replace(self.nld['begin_entity_token'],
                                                        self.nld['reformat_begin_entity_token'])
                            sentence = sentence.replace(self.nld['end_entity_token'],
                                                        self.nld['reformat_end_entity_token'])
                    # create example from given sentence
                    example = parse_example(sentence=sentence,
                                            example_id=dataset['document_id'] + f"_{guid}",
                                            events=dataset['events'],
                                            entities=dataset['entities'],
                                            relations=dataset['relations'],
                                            offset=s_t,
                                            skip_oos_examples=self.hparams.skip_oos_examples,
                                            valid_event_types=valid_event_types
                                            )
                    if example:
                        # add natural language description and dataset name
                        example.nld = self.nld
                        example.builder_name = complete_dataset.builder_name

                        # check if example is usable for experiment according to prompt restrictions
                        example_event_types = []

                        # get number of event types
                        for event in example.events:
                            if event.type not in example_event_types:
                                example_event_types.append(event.type)

                        # create input and output sentences for prompt
                        create_input_example(example,
                                             self.nld,
                                             entity_type=self.hparams.entity_type,
                                             blocked_entities=self.hparams.blocked_entities,
                                             task='ee')
                        create_output_example(example,
                                              self.nld,
                                              blocked_entities=self.hparams.blocked_entities,
                                              task='ee',
                                              events_only=self.hparams.events_only)
                        examples.append(example)

                    # update passage offset
                    s_t = cal_offset(s_t, passage_text.split(sentence)[-1], sentence)

        return examples, entity_types, event_types, relation_types


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "biomedical.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg.defaults)
