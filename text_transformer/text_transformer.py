import numpy as np

import torch
from transformers import BertTokenizer, BertModel

import regex as re
import string


# 'justin871030/bert-base-uncased-goemotions-original-finetuned'
def clean_text(text):
    """Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers."""
    text = str(text).lower()  # Converting all text to lowercase
    text = re.sub("@[A-Za-z0-9_]+", "", text)  # Removing Mentions (Eg: @tserre)
    text = re.sub('\[.*?\]', '', text)  # Removing Text in brackets
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Removing URLs
    text = re.sub('<.*?>+', ' ', text)  # Removing Text within '< >'
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Removing Punctuations
    text = re.sub('\n', '', text)  # Removing new-line characters
    text = re.sub('\r', '', text)  # Removing Carriage Return characters
    text = re.sub(' +', ' ', text)  # Removing more than 1 space
    text = re.sub('\w*\d\w*', '', text)  # Removing text with digits
    text = text.strip()  # Removing any remaining redundant spaces
    return text


class TextTransformer:
    def __init__(self, version):
        self.model = BertModel.from_pretrained(version)
        print("Loaded model")
        # if torch.cuda.is_available():
        #     self.device = 'cuda'
        # elif torch.backends.mps.is_available():
        #     self.device = 'mps'
        # else:
        #     self.device = 'cpu'
        self.device = 'cpu'
        print("Using device:", self.device)
        self.model.to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(version, do_lower_case=True)
        self.embeddings = torch.Tensor([])
        self.queries = []
        self.query_thresholds = []

    def embed_text(self, comments):
        results = []

        print("Starting embedding...")
        for text in comments:
            input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)
            if self.device != 'cpu':
                input_ids = input_ids.to(self.device)

            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, :512]

            outputs = self.model(input_ids)[0].mean(1).detach()
            if self.device != 'cpu':
                outputs = outputs.cpu()
            results.append(outputs.squeeze().numpy())

        print("Done embedding")
        return torch.Tensor(np.array(results))

    def compare_similarity(self, input_embedding):
        matches = []
        scores = []
        # enumerate over all embeddings
        print("Starting similarity comparison...")
        embeddings_with_input = torch.cat((self.embeddings, input_embedding.unsqueeze(0)), dim=0)
        # TODO add gpu support
        for i, query_embedding in enumerate(embeddings_with_input):
            if i != len(self.embeddings):
                print("Comparing to query", i + 1, "of", len(self.embeddings))
            similarity = torch.nn.functional.cosine_similarity(
                input_embedding, query_embedding, dim=0
            ).item()
            # similarity of the input to itself will be 1, make sure it is included in the matches
            if i == len(self.embeddings):
                assert round(similarity, 5) == 1
                continue
            elif similarity >= self.query_thresholds[i]:
                matches.append(self.queries[i])
                scores.append(similarity)

        return matches, scores

    def load_queries(self, queries, thresholds):
        # TODO add gpu support
        self.queries = queries
        self.query_thresholds = thresholds
        self.embeddings = self.embed_text([clean_text(query) for query in queries])
        print(self.embeddings.shape)
        print("Done loading queries")
