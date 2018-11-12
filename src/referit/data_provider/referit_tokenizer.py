from nltk.tokenize import TweetTokenizer
import json


class ReferitTokenizer:
    """ """
    def __init__(self, dictionary_file, dataset):

        self.tokenizer = TweetTokenizer(preserve_case=False)
        with open(dictionary_file, 'r') as f:
            data = json.load(f)
        self.word2i = data[dataset]['word2i']

        self.dictionary_file = dictionary_file

        self.i2word = {}
        for (k, v) in self.word2i.items():
            self.i2word[v] = k

        # Retrieve key values
        self.no_words = len(self.word2i)

        self.start_token = self.word2i["<start>"]
        self.stop_token = self.word2i["<stop>"]
        self.unknown_question_token = self.word2i["<unk>"]
        self.padding_token = self.word2i["<padding>"]

        assert self.padding_token == 0

    """
    Input: String
    Output: List of tokens
    """
    def encode_question(self, question):
        tokens = [self.start_token]
        for token in self.tokenizer.tokenize(question):
            if token not in self.word2i:
                token = '<unk>'
            tokens.append(self.word2i[token])

        return tokens

    def decode_question(self, tokens):
        return ' '.join([self.i2word[tok] for tok in tokens])

    def tokenize_question(self, question):
        return self.tokenizer.tokenize(question)

