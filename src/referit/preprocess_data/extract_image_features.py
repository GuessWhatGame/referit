#!/usr/bin/env python
import os
import tensorflow as tf
from distutils.util import strtobool
import numpy as np
import argparse

from generic.data_provider.image_loader import RawImageBuilder, RawCropBuilder
from generic.preprocess_data.extract_img_features import extract_features
from generic.data_provider.dataset import CropDataset

from neural_toolbox import resnet

from referit.data_provider.referit_dataset import ReferitDataset

from referit.data_provider.referit_batchifier import ReferitBatchifier


parser = argparse.ArgumentParser('Feature extractor! ')

parser.add_argument("-img_dir", type=str, required=True, help="Input Image folder")
parser.add_argument("-data_dir", type=str, required=True,help="Dataset folder")
parser.add_argument("-out_dir", type=str, required=True, help="Output directory for h5 files")

parser.add_argument("-ckpt", type=str, required=True, help="Path for network checkpoint: ")
parser.add_argument("-resnet_version", type=int, default=152, choices=[50, 101, 152], help="Pick the resnet version [50/101/152]")
parser.add_argument("-feature_name", type=str, default="block3", help="Pick the name of the network features")

parser.add_argument("-dataset", type=str, choices=["refclef", "refcoco", "refcoco+", "refcocog"], help="Selec")
parser.add_argument("-mode", type=str, choices=["img", "crop", "crop_all"], help="Select to either dump the img/crop feature")

parser.add_argument("-subtract_mean", type=lambda x: bool(strtobool(x)), default="True", help="Preprocess the image by substracting the mean")
parser.add_argument("-img_size", type=int, default=448, help="image size (pixels)")
parser.add_argument("-batch_size", type=int, default=64, help="Batch size to extract features")
parser.add_argument("-crop_scale", type=float, default=1.1, help="crop scale around the bbox")

parser.add_argument("-gpu_ratio", type=float, default=0.45, help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")

args = parser.parse_args()

# define image
if args.subtract_mean:
    channel_mean = np.array([123.68, 116.779, 103.939])
else:
    channel_mean = None

split_by = "unc"
if args.dataset == "refcocog":
    split_by = "google"

# define the image loader
dataset_args = {"folder": args.data_dir,
                "dataset": args.dataset,
                "split_by": split_by}

if args.mode == "img":
    images = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3], name='image')
    source = 'image'
    dataset_cstor = ReferitDataset
    image_builder = RawImageBuilder(args.img_dir,
                                    height=args.img_size,
                                    width=args.img_size,
                                    channel=channel_mean)

    dataset_args["image_builder"] = image_builder

elif "crop" in args.mode:
    images = tf.placeholder(tf.float32, [None, args.img_size, args.img_size, 3], name='crop')
    source = 'crop'
    dataset_cstor = CropDataset.load
    crop_builder = RawCropBuilder(args.img_dir,
                                  height=args.img_size,
                                  width=args.img_size,
                                  scale=args.crop_scale,
                                  channel=channel_mean)

    dataset_args["dataset_cls"] = ReferitDataset
    dataset_args["expand_objects"] = (args.mode == "crop_all")
    dataset_args["crop_builder"] = crop_builder

else:
    assert False, "Invalid mode: {}".format(args.mode)


# create network
print("Create network...")
ft_output = resnet.create_resnet(images,
                                 resnet_out=args.feature_name,
                                 resnet_version=args.resnet_version,
                                 is_training=False)

extract_features(
    img_input=images,
    ft_output=ft_output,
    dataset_cstor=dataset_cstor,
    dataset_args=dataset_args,
    batchifier_cstor=ReferitBatchifier,
    out_dir=args.out_dir,
    set_type=["all"],
    network_ckpt=args.ckpt,
    batch_size=args.batch_size,
    no_threads=args.no_thread,
    gpu_ratio=args.gpu_ratio)

