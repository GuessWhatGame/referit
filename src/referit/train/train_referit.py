import argparse
import logging
import tensorflow as tf
from distutils.util import strtobool

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.utils.config import load_config
from generic.utils.thread_pool import create_cpu_pool
from generic.data_provider.image_loader import get_img_builder
from generic.data_provider.nlp_utils import GloveEmbeddings

from referit.models.referit_network_factory import create_network
from referit.data_provider.referit_dataset import ReferitDataset
from referit.data_provider.referit_batchifier import ReferitBatchifier
from referit.data_provider.referit_tokenizer import ReferitTokenizer
from referit.train.listener import ReferitAccuracyListener


if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Referit network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-out_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-img_dir", type=str, help="Directory with image")
    parser.add_argument("-crop_dir", type=str, help='Directory with crops')

    parser.add_argument("-dataset", type=str, choices=["refclef", "refcoco", "refcoco+", "refcocog"], help="Select referit dataset")
    parser.add_argument("-split_by", type=str, choices=["unc", "berkeley", "umd", "google"], help="Select referit split (google not included as there is no testing set)")

    parser.add_argument("-config", type=str, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-glove_file", type=str, default="glove_dict.pkl", help="Glove file name")

    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", type=lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")

    parser.add_argument("-no_thread", type=int, default=2, help="No thread to load batch")
    parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_games_to_load", type=int, default=float("inf"), help="No games to use during training Default : all")

    args = parser.parse_args()

    config, xp_manager = load_config(args)
    logger = logging.getLogger()

    # Load config
    finetune = config["model"]["image"].get('finetune', list())
    no_epoch = config["optimizer"]["no_epoch"]
    batch_size = config["optimizer"]["batch_size"]

    # Load images
    logger.info('Loading images..')
    image_builder = get_img_builder(config['model']['image'], args.img_dir, is_crop=False)

    logger.info('Loading crops..')
    crop_builder = get_img_builder(config['model']['crop'], args.crop_dir, is_crop=True)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = ReferitTokenizer(args.dict_file, dataset=args.dataset)

    # Load glove
    glove = None
    if config["model"]["question"]["glove"]:
        logger.info('Loading glove..')
        glove = GloveEmbeddings(args.glove_file)

    # Load data
    logger.info('Loading data..')
    trainset = ReferitDataset(args.data_dir, which_set="train", dataset=args.dataset, split_by=args.split_by, image_builder=image_builder, crop_builder=crop_builder, games_to_load=args.no_games_to_load)
    validset = ReferitDataset(args.data_dir, which_set="val", dataset=args.dataset, split_by=args.split_by, image_builder=image_builder, crop_builder=crop_builder, games_to_load=args.no_games_to_load)
    testset = ReferitDataset(args.data_dir, which_set="test", dataset=args.dataset, split_by=args.split_by, image_builder=image_builder, crop_builder=crop_builder, games_to_load=args.no_games_to_load)

    # Build Network
    logger.info('Building network..')
    network = create_network(config=config["model"], no_words=tokenizer.no_words)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimize, outputs = create_optimizer(network, config["optimizer"], finetune=finetune)

    ###############################
    #  START  TRAINING
    #############################

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # retrieve incoming sources
        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))

        # Load checkpoints or pre-trained networks
        sess.run(tf.global_variables_initializer())
        if args.continue_exp or args.load_checkpoint is not None:
            start_epoch, _ = xp_manager.load_checkpoint(sess, saver)
        else:
            start_epoch = 0

        # Create evaluation tools
        evaluator = Evaluator(sources, scope=network.scope_name, network=network, tokenizer=tokenizer)
        batchifier = ReferitBatchifier(tokenizer, sources, glove=glove, split_by_objects=True)

        # define listener
        listener = ReferitAccuracyListener(require=network.softmax)

        for t in range(start_epoch, no_epoch):

            # CPU
            cpu_pool = create_cpu_pool(args.no_thread, use_process=image_builder.require_multiprocess())

            logger.info('Epoch {}/{}..'.format(t + 1, no_epoch))

            train_iterator = Iterator(trainset,
                                      batch_size=batch_size,
                                      batchifier=batchifier,
                                      shuffle=True,
                                      pool=cpu_pool)
            [train_loss, train_accuracy_fake] = evaluator.process(sess, train_iterator, outputs=outputs + [optimize], listener=listener)
            train_accuracy = listener.evaluate()

            valid_iterator = Iterator(validset,
                                      batch_size=batch_size*2,
                                      batchifier=batchifier,
                                      shuffle=False,
                                      pool=cpu_pool)

            [valid_loss, valid_accuracy_fake] = evaluator.process(sess, valid_iterator, outputs=outputs, listener=listener)
            valid_accuracy = listener.evaluate()

            logger.info("Training loss: {}".format(train_loss))
            logger.info("Training accuracy : {}".format(train_accuracy))
            logger.info("Training accuracy (Fake): {}".format(train_accuracy_fake))

            logger.info("Validation loss: {}".format(valid_loss))
            logger.info("Validation accuracy: {}".format(valid_accuracy))
            logger.info("Validation accuracy (Fake): {}".format(valid_accuracy_fake))

            xp_manager.save_checkpoint(sess, saver,
                                       epoch=t,
                                       train_loss=train_loss,
                                       valid_loss=valid_loss,
                                       extra_losses=dict(
                                           train_accuracy=train_accuracy,
                                           valid_accuracy=valid_accuracy,
                                       ))

        # Load early stopping
        xp_manager.load_checkpoint(sess, saver, load_best=True)
        cpu_pool = create_cpu_pool(args.no_thread, use_process=False)

        test_names = ["test"]
        if args.split_by == "unc":
            test_names = ["test", "testA", "testB"]

        for test_name in test_names:

            test_batchifier = ReferitBatchifier(tokenizer, sources, glove=glove, split_type=test_name, split_by_objects=True)
            test_iterator = Iterator(testset, pool=cpu_pool,
                                     batch_size=batch_size,
                                     batchifier=test_batchifier,
                                     shuffle=False)

            [test_loss, test_accuracy_fake] = evaluator.process(sess, test_iterator, outputs, listener=listener)
            test_accuracy = listener.evaluate()

            logger.info("======== {} ========".format(test_name))
            logger.info("Testing loss: {}".format(test_loss))
            logger.info("Testing accuracy (Fake): {}".format(test_accuracy_fake))
            logger.info("Testing accuracy: {}".format(test_accuracy))

            # Save the test scores
            xp_manager.update_user_data(
                user_data={
                    "{}_loss".format(test_name): test_loss,
                    "{}_accuracy".format(test_name): test_accuracy,
                }
            )
