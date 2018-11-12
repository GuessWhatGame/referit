import argparse
from nltk.tokenize import TweetTokenizer
import io
from generic.utils.file_handlers import pickle_dump

from referit.data_provider.referit_dataset import ReferitDataset


# wget http://nlp.stanford.edu/data/glove.42B.300d.zip

if __name__ == '__main__':


    parser = argparse.ArgumentParser('Creating GLOVE dictionary.. Please first download http://nlp.stanford.edu/data/glove.42B.300d.zip')

    parser.add_argument("-data_dir", type=str, default=".", help="Path to VQA dataset")
    parser.add_argument("-glove_in", type=str, default="glove.42B.300d.zip", help="Name of the stanford glove file")
    parser.add_argument("-glove_out", type=str, default="glove_dict.pkl", help="Name of the output glove file")

    args = parser.parse_args()

    dataset_name = ["refclef", "refcoco", "refcoco+", "refcocog"]
    split_by = ["unc", "unc", "unc", "google"]

    glove_dict = {}
    not_in_dict = {}

    print("Loading glove...")
    with io.open(args.glove_in, 'r', encoding="utf-8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    for name, split_by in zip(dataset_name, split_by):

        print("Load Dataset: ", name)

        # Note that we include the valid/test set in our voc,
        # In theory, this is bad practice,
        # In practice, it does not change the final score (as voc overlap) and ease a lot the implementation complexity as there are several test/val split
        dataset = ReferitDataset(args.data_dir, which_set=[], dataset=name, split_by=split_by)
        tokenizer = TweetTokenizer(preserve_case=False)

        print("Mapping glove...")
        for g in dataset.games:
            words = tokenizer.tokenize(g.sentence)
            for w in words:
                w = w.lower()
                w = w.replace("'s", "")
                if w in vectors:
                    glove_dict[w] = vectors[w]
                else:
                    not_in_dict[w] = 1

    print("Number of glove: {}".format(len(glove_dict)))
    print("Number of words with no glove: {}".format(len(not_in_dict)))

    for k in not_in_dict.keys():
        print(k)

    print("Dumping file...")
    pickle_dump(glove_dict, args.glove_out)

    print("Done!")



