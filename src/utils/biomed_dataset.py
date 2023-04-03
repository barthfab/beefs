import random
from typing import List
import math
from torch.utils.data.dataset import Dataset
from src.utils.example_creators import single_prompt_parser


class SingleBioEventDataset(Dataset):
    def __init__(self, data, train_data, nld, seed, example_size: int = 1,
                 learning_method: str = None, no_event_threshold: float = 0.2):
        super().__init__()
        self.data = data
        self.train_data = train_data
        self.event_types = [e for e in train_data.keys() if e != "None"]
        self.filler = [e for event_type in self.train_data.values() for e in event_type]
        self.nld = nld
        self.seed = seed
        random.seed(self.seed)

        self.example_size = example_size
        # if random or not
        self.learning_method = learning_method
        self.no_event_threshold = no_event_threshold

    def __getitem__(self, item):
        example = self.data[item]
        if self.learning_method == "random":
            bucket = random.sample(self.filler, k=self.example_size,)
        else:
            event_types = set([e.type for e in example.events])
            if event_types:
                bucket = self.event_specific(self.train_data, event_types)
            else:
                bucket = self.event_specific(self.train_data, self.event_types)
                #bucket = random.sample(self.filler, k=self.example_size)
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
                try:
                    bucket.append(random.choice([e for e in example_list[event_type] if e not in bucket]))
                except:
                    bucket.append(random.choice([e for e in example_list[event_type] if e not in bucket]))
            else:
                print(f"couldnt find an example for this event type: {event_type}")
                bucket.append(random.choice(example_list['None']))

        # fill the rest of the bucket with random examples
        if self.example_size - len(bucket) > 0:
            try:
                not_nones = random.sample([e for e in filler if e not in bucket],
                                  k=math.ceil((self.example_size - len(bucket)) * (1 - self.no_event_threshold)))
            except:
                not_nones = random.sample([e for e in example_list['None'] if e not in bucket],
                                          k=math.ceil((self.example_size - len(bucket)) * (1 - self.no_event_threshold)))
            nones = random.sample([e for e in example_list['None'] if e not in bucket],
                              k=math.floor((self.example_size - len(bucket)) * self.no_event_threshold))
            bucket.extend(not_nones)
            bucket.extend(nones)

        random.shuffle(bucket)
        return bucket
