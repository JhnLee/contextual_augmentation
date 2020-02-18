from transformers import BertTokenizer, BertConfig
import logging
import os
import csv
import pickle

import torch
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path):
        assert os.path.isfile(file_path)
        model_type = 'test'
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_" + filename
        )

        if os.path.exists(cached_features_file):
            # and not args.overwrite_cache
            logger.info("Loading features from cached file %s",
                        cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.sentence, self.label = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            data = []
            with open(file_path, encoding="utf-8") as tsvreader:
                for line in csv.reader(tsvreader, delimiter='\t'):
                    data.append(line)
            text, label = list(zip(*data[1:]))
            tokenized_text = [tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(t)) for t in text]

            tokenized_text = [tokenizer.build_inputs_with_special_tokens(
                t) for t in tokenized_text]

            self.sentence = tokenized_text
            self.label = [float(l) for l in label]

            logger.info("Saving features into cached file %s",
                        cached_features_file)
            
            with open(cached_features_file, "wb") as handle:
                pickle.dump((self.sentence, self.label), handle,
                            protocol = pickle.HIGHEST_PROTOCOL)
            
    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        sentence = torch.tensor(self.sentence[item], dtype=torch.long)
        label = torch.tensor(self.label[item], dtype=torch.long)
        return sentence, label
    

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=512):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if len(line) > 0]

        self.examples = tokenizer.batch_encode_plus(lines, max_length=block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)