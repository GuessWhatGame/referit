import collections
import numpy as np
import copy
from PIL import Image as PImage

from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image, scaled_crop_and_pad
from generic.data_provider.nlp_utils import padder, padder_3d
import time


class ReferitBatchifier(object):

    def __init__(self, tokenizer, sources, glove=None, split_type=None):
        self.tokenizer = tokenizer
        self.sources = sources
        self.glove = glove

        self.split_type = split_type

    def filter(self, games):

        # TODO : Move into dataset
        if self.split_type is not None:
            if self.split_type == "testA":
                games = [g for g in games if g.object.category == "person"]

            elif self.split_type == "testB":
                games = [g for g in games if g.object.category != "person"]

        return games

    def split(self, games):

        # TODO add option to not split dataset (require for image feature extraction)

        if self.split_type == "no_split":
            return games

        new_games = []

        for game in games:
            for i, obj in enumerate(game.objects):
                new_game = copy.copy(game)
                new_game.object = obj
                new_game.object_id = obj.id
                new_game.correct_object = (obj.id == game.object.id)

                new_games.append(new_game)

        return new_games

    #TODO create a source list
    def apply(self, games):

        batch = collections.defaultdict(list)
        batch_size = len(games)

        for i, game in enumerate(games):

            batch["raw"].append(game)

            # Get referit sentence
            sentence = self.tokenizer.encode_question(game.sentence)
            batch['question'].append(sentence)

            # Get gloves
            if self.glove is not None:
                words = self.tokenizer.tokenize_question(game.sentence)
                glove_vectors = self.glove.get_embeddings(words)
                batch['glove'].append(glove_vectors)

            if 'answer' in self.sources:
                answer = [0, 0]
                answer[int(game.correct_object)] = 1
                batch['answer'].append(answer)

            if "image" in self.sources:
                img = game.image.get_image()
                if "image" not in batch:  # initialize an empty array for better memory consumption
                    batch["image"] = np.zeros((batch_size,) + img.shape)
                batch["image"][i] = img

            if "crop" in self.sources:
                crop = game.object.get_crop()
                if "crop" not in batch:  # initialize an empty array for better memory consumption
                    batch["crop"] = np.zeros((batch_size,) + crop.shape)
                batch["crop"][i] = crop

            if 'img_mask' in self.sources:
                assert "image" in batch, "mask input require the image source"
                mask = game.object.get_mask()

                ft_width, ft_height = batch['image'][-1].shape[1], \
                                      batch['image'][-1].shape[0]  # Use the image feature size (not the original img size)

                mask = resize_image(PImage.fromarray(mask), height=ft_height, width=ft_width)
                batch['img_mask'].append(np.array(mask))

            if 'crop_mask' in self.sources:
                assert "crop" in batch, "mask input require the crop source"
                cmask = game.object.get_mask()

                ft_width, ft_height = batch['crop'][-1].shape[1], \
                                      batch['crop'][-1].shape[0]  # Use the crop feature size (not the original img size)

                cmask = scaled_crop_and_pad(raw_img=PImage.fromarray(cmask), bbox=game.object.bbox, scale=game.object.crop_scale)
                cmask = resize_image(cmask, height=ft_height, width=ft_width)
                batch['crop_mask'].append(np.array(cmask))

            if 'category' in self.sources:
                batch['category'].append(game.object.category_id)

            if 'spatial' in self.sources:
                spat_feat = get_spatial_feat(game.object.bbox, game.image.width, game.image.height)
                batch['spatial'].append(spat_feat)

        # Pad referit sentence
        batch['question'], batch['seq_length'] = padder(batch['question'],
                                                        padding_symbol=self.tokenizer.padding_token)

        if self.glove is not None:
            batch['glove'], _ = padder_3d(batch['glove'])

        return batch
