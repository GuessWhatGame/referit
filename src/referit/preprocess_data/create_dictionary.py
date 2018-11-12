from nltk.tokenize import TweetTokenizer
import io
import json
import collections
from referit.data_provider.referit_dataset import ReferitDataset
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Creating dictionary..')

    parser.add_argument("-data_dir", type=str, help="Path to Referit dataset")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Name of the dictionary file")
    parser.add_argument("-min_occ", type=int, default=3, help='Minimum number of occurences to add word to dictionary')

    args = parser.parse_args()

    dataset_name = ["refclef", "refcoco", "refcoco+", "refcocog"]
    split_by = ["unc", "unc", "unc", "google"]

    dico = dict()

    for name, split_by in zip(dataset_name, split_by):

        print("Load Dataset: ", name)

        # Note that we include the valid/test set in our voc,
        # In theory, this is bad practice,
        # In practice, it does not change the final score (as voc overlap) and ease a lot the implementation complexity as there are several test/val split
        dataset = ReferitDataset(args.data_dir, which_set=[], dataset=name, split_by=split_by)

        games = dataset.games

        word2i = {'<padding>': 0,
                  '<start>': 1,
                  '<stop>': 2,
                  '<unk>': 3
                  }

        word2occ = collections.defaultdict(int)

        # Input words
        tknzr = TweetTokenizer(preserve_case=False)

        for game in games:
            input_tokens = tknzr.tokenize(game.sentence)
            for tok in input_tokens:
                word2occ[tok] += 1

        # parse the questions
        for word, occ in word2occ.items():
            if occ >= args.min_occ:
                word2i[word] = len(word2i)

        print("Number of words: {}".format(len(word2i)))

        dico[name] = {"word2i": word2i}

    with io.open(args.dict_file, 'w', encoding='utf8') as f_out:
       data = json.dumps(dico)
       f_out.write(data)
