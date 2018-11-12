from generic.data_provider.dataset import AbstractDataset
import numpy as np
import copy
import os
import re

from generic.data_provider.dataset import AbstractDataset
from referit.data_provider.refer import REFER

import cocoapi.PythonAPI.pycocotools.mask as cocoapi


class Game:

    def __init__(self, id, image, target_object, objects, sentence):
        self.dialogue_id = id

        self.image = image

        self.objects = objects
        self.object = target_object
        self.object_id = target_object.id

        assert self.object_id == self.object.id

        self.correct_object = True  # TODO clean

        self.sentence = sentence

    def __str__(self):
        return "#{id} \t image_id: {im_id} \t object_id: {obj_id} \t {sentence}".format(
            id=self.dialogue_id,
            im_id=self.image.id,
            obj_id=self.object_id,
            sentence=self.sentence)

class Image:
    def __init__(self, image_id, width, height, url, filename, image_builder=None):
        self.id = image_id
        self.width = width
        self.height = height
        self.url = url
        self.filename = filename

        self.image_loader = None
        if image_builder is not None:
            self.image_loader = image_builder.build(image_id, filename=self.filename, optional=False)

    def get_image(self, **kwargs):
        if self.image_loader is not None:
            return self.image_loader.get_image(**kwargs)
        else:
            return None


class Bbox:
    def __init__(self, bbox, im_width, im_height):
        # Retrieve features (COCO format)
        self.x_width = bbox[2]
        self.y_height = bbox[3]

        self.x_left = bbox[0]
        self.x_right = self.x_left + self.x_width

        self.y_upper = im_height - bbox[1]
        self.y_lower = self.y_upper - self.y_height

        self.x_center = self.x_left + 0.5 * self.x_width
        self.y_center = self.y_lower + 0.5 * self.y_height

        self.coco_bbox = bbox

    def __str__(self):
        return "center : {0:5.2f}/{1:5.2f} - size: {2:5.2f}/{3:5.2f}" \
            .format(self.x_center, self.y_center, self.x_width, self.y_height)


class Object:
    def __init__(self, crop_id, category, category_id, bbox, area, segment, crop_builder, image):
        self.id = crop_id
        self.category = category
        self.category_id = category_id
        self.bbox = bbox
        self.area = area
        self.segment = segment

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        if type(segment) == dict or type(segment[0]) == list:  # polygon
            self.rle_mask = cocoapi.frPyObjects(segment, h=image.height, w=image.width)
        else:
            self.rle_mask = segment

        if crop_builder is not None:
            self.crop_loader = crop_builder.build(crop_id, filename=image.filename, bbox=bbox)
            self.crop_scale = crop_builder.scale

    def get_mask(self):
        assert self.rle_mask is not None, "Mask option are not available, please compile and link cocoapi (cf. cocoapi/PythonAPI/setup.py)"
        tmp_mask = cocoapi.decode(self.rle_mask)
        if len(tmp_mask.shape) > 2:  # concatenate several mask into a single one
            tmp_mask = np.sum(tmp_mask, axis=2)
            tmp_mask[tmp_mask > 1] = 1

        return tmp_mask.astype(np.float32)

    def get_crop(self, **kwargs):
        assert self.crop_loader is not None, "Invalid crop loader"
        return self.crop_loader.get_image(**kwargs)


class ReferitDataset(AbstractDataset):
    """Loads the dataset."""

    def __init__(self, folder, which_set, dataset, split_by, image_builder=None, crop_builder=None, games_to_load=float("inf")):

        # Load referit data
        self.refer = REFER(data_root=folder, dataset=dataset, splitBy=split_by)

        if which_set == "all":
            which_set = []

        games = []
        for id in self.refer.getRefIds(split=which_set):

            ref_game = self.refer.loadRefs(id)[0]

            image_id = ref_game["image_id"]
            object_id = ref_game['ann_id']

            # Create Image
            img_ref = self.refer.Imgs[image_id]

            # Extract the coco folder name from the image name
            image_filename = img_ref["file_name"]
            if "coco" in dataset:
                img_dir = re.search('COCO_([^_]*)_.*', img_ref["file_name"]).group(1)
                image_filename = os.path.join(img_dir, image_filename)

            image = Image(
                image_id=image_id,
                filename=image_filename,
                width=img_ref["width"],
                height=img_ref["height"],
                url=img_ref.get("coco_url", ""),
                image_builder=image_builder)

            # Create objects in the image/game
            target_object, objects = None, []
            for annotation in self.refer.imgToAnns[image_id]:

                object_category_id = ref_game['category_id']
                object_category = self.refer.Cats[object_category_id]

                obj = Object(
                    crop_id=annotation["id"],
                    category=object_category,
                    category_id=object_category_id,
                    bbox=Bbox(annotation["bbox"], img_ref["width"], img_ref["height"]),
                    area=annotation["area"],
                    segment=annotation["segmentation"],
                    crop_builder=crop_builder,
                    image=image)

                if annotation["id"] == object_id:
                    target_object = obj

                objects.append(obj)

            assert target_object is not None

            # Create a game for each referit sentence
            for sentence in ref_game["sentences"]:
                index_sentence = sentence["sent_id"]

                id_game = (index_sentence+1) * 1000000 + id  # as there is no game_id, we compute one from the sentence_index nd referit index

                game = Game(id_game,
                            image=image,
                            target_object=target_object,
                            objects=objects,
                            sentence=sentence["sent"])

                games.append(game)

            if len(games) >= games_to_load:
                break

        print("{} games were loaded...".format(len(games)))
        super(ReferitDataset, self).__init__(games)


class CropDataset(AbstractDataset):
    """
    Each game contains no question/answers but a new object
    """

    def __init__(self, dataset, expand_objects):
        old_games = dataset.get_data()
        new_games = []

        for g in old_games:
            if expand_objects:
                new_games += self.split(g)
            else:
                new_games += self.update_ref(g)

        super(CropDataset, self).__init__(new_games)

    @classmethod
    def load(cls, folder, which_set, dataset, split_by, image_builder=None, crop_builder=None, expand_objects=False, games_to_load=float("inf")):
        return CropDataset(ReferitDataset(folder, which_set, dataset, split_by, image_builder, crop_builder, games_to_load=games_to_load),
                           expand_objects=expand_objects)

    def split(self, game):
        games = []
        for obj in game.objects:
            new_game = copy.copy(game)

            # select new object
            new_game.object = obj
            new_game.object_id = obj.id

            # Hack the image id to differentiate objects
            new_game.image = copy.copy(game.image)
            new_game.image.id = obj.id

            games.append(new_game)

        return games

    def update_ref(self, game):

        new_game = copy.copy(game)

        # Hack the image id to differentiate objects
        new_game.image = copy.copy(game.image)
        new_game.image.id = game.object_id

        return [new_game]


if __name__ == '__main__':

    coco1 = ReferitDataset(folder="/media/datas1/dataset/referit/",
                   which_set="all",
                   dataset="refcoco",
                   split_by="unc", image_builder=None, crop_builder=None)

    coco2 = ReferitDataset(folder="/media/datas1/dataset/referit/",
                           which_set="all",
                           dataset="refcoco+",
                           split_by="unc", image_builder=None, crop_builder=None)

    coco3 = ReferitDataset(folder="/media/datas1/dataset/referit/",
                           which_set="all",
                           dataset="refcocog",
                           split_by="google", image_builder=None, crop_builder=None)

    img1 = set(g.image.id for g in coco1.games)
    img2 = set(g.image.id for g in coco2.games)
    img3 = set(g.image.id for g in coco3.games)

    print("test")



