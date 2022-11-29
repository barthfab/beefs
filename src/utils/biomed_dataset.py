import random
from typing import List
from torch.utils.data.dataset import Dataset
from src.utils.example_creators import single_prompt_parser


class SingleBioEventDataset(Dataset):
    def __init__(self, data, train_data, nld, example_size: int = 1, learning_method: str = None, no_event_threshold: float = 0.2,
                 ):
        super().__init__()
        #todo use train data to fill bucket and dev data for prediction
        self.data = data
        self.train_data = train_data
        self.filler = [e for event_type in self.train_data.values() for e in event_type]
        self.nld = nld

        self.example_size = example_size
        # if random or not
        self.learning_method = learning_method
        self.no_event_threshold = no_event_threshold

    def __getitem__(self, item):
        example = self.data[item]
        if self.learning_method == "random":
            bucket = random.sample(self.filler, k=self.example_size)
        else:
            event_types = set([e.type for e in example.events])
            if event_types:
                bucket = self.event_specific(self.train_data, event_types)
            else:
                bucket = random.sample(self.filler, k=self.example_size)
        prompt = single_prompt_parser(bucket + [example], self.nld)
        return prompt, example

    def __len__(self):
        return len(self.data)

    def event_specific(self, example_list, event_types):
        bucket = []
        filler = []
        for event_type in event_types:
            if event_type in example_list.keys():
                filler.extend(example_list[event_type])
                bucket.append(random.choice([e for e in example_list[event_type] if e not in bucket]))
            else:
                print(f"couldnt find an example for this event type: {event_type}")
                bucket.append(random.choice(example_list['None']))

        # fill the rest of the bucket with random examples
        if self.example_size - len(bucket) > 0:
            bucket.extend(random.sample([e for e in filler + example_list['None'] if e not in bucket], k=self.example_size - len(bucket)))

        random.shuffle(bucket)
        return bucket


class BioEventDataset(Dataset):
    def __init__(self, prompts, nld):
        super().__init__()
        # get train data
        self.data = prompts
        self.nld = {'begin_entity_token': nld.begin_entity_token,
                    'separator_token': nld.separator_token,
                    'relation_separator_token': nld.relation_separator_token,
                    'end_entity_token': nld.end_entity_token,
                    'query_separator_token': nld.query_separator_token,
                    'output_head': nld.output_head,
                    'example_head': nld.example_head,
                    'head': nld.head,
                    'example_separator': nld.example_separator}

    def __getitem__(self, item):
        return self.data[item], self.nld

    def __len__(self):
        return len(self.data)
