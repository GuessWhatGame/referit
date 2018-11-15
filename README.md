# ReferIt
Code to run ReferIt models with Multi-Hop Generator [1]

The code was developed by Florian Strub (University of Lille) and Mathieu Seurin (University of Lille)

Note: the code is still under cleaning. Having said that, we already reproduced the baseline of the paper

## Requirement

### Python package
```pip install \
    tensorflow-gpu \
    nltk \
    tqdm \
    cython \
``` 

### File architecture
In the following, we assume that the following file/folder architecture is respected:

```
referit
├── config         # store the configuration file to create/train models
|   └── referit
|
├── out            # store the output experiments (checkpoint, logs etc.)
|   ├── referit_refclef
|   ├── referit_coco
|   ├── referit_coco+
|   └── referit_cocog
|
├── data          # contains the VQA data
|   ├── refclef
|   ├── coco
|   ├── coco+
|   ├── cocog
|   └── images
|       ├── mscoco
|       └── saiapr_tc-12
|
└── src            # source files
```

To complete the git-clone file architecture, you can do:

```
cd guesswhat
mkdir data;
mkdir out; mkdir out/vqa
```


### Compile Coco API
Go to the src/cocoapi/PythonAPI folder to compile ms coco utilities
```
cd src/cocoapi/PythonAPI 
make install 
```

Note:
 - You may have to use the command sudo
 - the Makefile is using your default python interpreter. You may have to change the Makefile to enforce a specific python interpreter. 


## Data

### Referit dataset
Four Referit Dataset co-exist in the literature, lichengunc formatted the dataset into a single format to ease research.  
Thus, download the cleaned data from https://github.com/lichengunc/refer 

```
dataset=( refclef refcoco refcoco+ refcocog)
for d in "${dataset[@]}"; do
    wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/${d}.zip -P data/
    unzip data/${d}.zip -d data/
done
```

Do not forget to cite their paper ;) [2]

### Images
Referit dataaset either relies on the [ImageClef](http://imageclef.org/SIAPRdata) or [MS Coco](http://cocodataset.org/) dataset.

Download ImageClef 
```
wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip -P data/images
unzip data/images/saiapr_tc-12.zip -d data/images
```

Download MS Coco
```
wget http://images.cocodataset.org/zips/train2014.zip -P data/images/
unzip data/images/train2014.zip -d data/images/mscoco/
ln -s data/images/mscoco/train2014 data/images/mscoco/train 
``` 

### Glove
Our model can use GLOVE vectors (pre-computed word embedding).
To create the GLOVE dictionary, please download the original glove file.

```
wget http://nlp.stanford.edu/data/glove.42B.300d.zip -P data/
unzip data/glove.42B.300d.zip -d data/
```

## Preprocessing

### Extract image/crop features
You need to extract the ResNet features for both the image and the object crop.

First, you need to download the resnet checkpoint

```
wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz -P data/
tar zxvf data/resnet_v1_152_2016_08_28.tar.gz -C data/
```
For each referit dataset, you need to extract the image/crop features.
To do so, you can use the python script referit/src/guesswhat/preprocess_data/extract_img_features.py

```
array=( img crop_all )
for mode in "${array[@]}"; do
   python src/referit/preprocess_data/extract_img_features.py \
        -img_dir data/mscoco/images \
        -data_dir data/ \
        -out_dir data/${dataset}/${mode}_ft \
        -ckpt datas/resnet_v1_152.ckpt \
        -resnet_version 152 \
        -feature_name block3 \
        -mode ${mode} \
        -dataset ${dataset} \
done
```

Note: 
 - The crop are very heavy ~(150 Go)
 - You can use a symlink between refcoco and refcoco+ as they are using the same images

### Extract vocabulary

To create the ReferIt vocabulary, you need to use the python script referit/src/guesswhat/preprocess_data/create_dictionary.py

```
python src/referit/preprocess_data/create_dictionary.py \
        -data_dir data/referit \
        -dict_file data/referit/dict.json
```

Note that we are using a single vocabulary file for all Referit dataset for technical reasons. 
However, the vocabulary are clearly separated inside the final json file.


```
python src/referit/preprocess_data/create_gloves.py \ 
        -data_dir data/referit \
        -glove_in data/glove.42B.300d.txt \
        -glove_out data/referit/glove_dict.pkl
```

## Training

You only have to run the following script

```
python src/referit/train/train_referit.py \ 
        -data_dir data/referit \
        -out_dir out/referit_refclef \
        -img_dir data/refclef/img_ft \
        -crop_dir data/refclef/cop_all_ft \ 
        -config config/referit/config.json \
        -dict_file data/dict.json \ 
        -glove_file data/glove_dict.pkl \
        -dataset refclef \
        -split_by berkeley \
        -dataset refcoco \
```

## Config

You can configure your models by updating the config file in config/referit. 
Note that it is easy to implement an extra baseline by changing the type flag

```
{
  "name" : "Multi-Hop FiLM",            # some config description 

  "model": {

    "type" : "film",                    # Select the name of your network (for multiple baselines)

    "inputs": {             
      "crop": true,                     # use crop input 
      "image": true                     # use image input
    },

    "question": {                       # Define language model
      "word_embedding_dim": 200,        # size of the word embedding 
      "rnn_state_size": 512,            # number of units in the GRU
      "glove" : false,                  # append glove (concatenated to word embedding)
      "bidirectional" : true,           # use bidirectional GRU
      "layer_norm" : true,              # use layer norm on GRU
      "max_pool" : false                # use max-pooling over states as a final state
    },

    "spatial": {                        # define spatial feature
      "no_mlp_units": 30                # upsampling of the bbox information
    },

    "classifier":                       # define classifier model 
    {
      "inputs" :                        # define the inputs of the classifier input embedding
      {
       "question": true,                # re-inject language embedding
       "category": false,               # re-inject category embedding (not recommended for ReferIt)
       "spatial": true                  # re-inject spatial embedding
      },

      "no_mlp_units": 512               # hidden layer of the classifier
    },

    "film_input":                       # Define the FiLM generator
    {
      "category": false,                # re-inject category embedding in FiLM embedding
      "spatial": false,                 # re-inject spatial embedding in FiLM embedding
      "mask": false,                    # re-inject mask embedding in FiLM embedding

      "reading_unit": {                 # Define Multi-hop FiLM generator
        "reading_unit_type" : "basic",  # Select your favorite generator

        "attention_hidden_units": 0,    # hidden layer of attention
        "shared_attention": false,      # use shared attention over hops

        "inject_img_before": true,      # inject img-mean pooling in context cell
        "inject_img_after": false,      # inject img-mean pooling in FiLM embedding
   
        "sum_memory" : true             # sum context cell at each hop (cf memory network)

      }

    },

    "image": {                          # Define the image setting
      "image_input": "conv",            # input type (raw/feature/conv)
      "dim": [14, 14, 1024],            # input dimension
      "normalize": false,               # Normalize image by L2 norm
    },

    "crop": {                           # Define the crop setting
      "image_input": "conv",               
      "dim": [14, 14, 1024],
      "normalize": false,
      "scale" : 1.1,                    # Define the the % crop outside the initial bounding box
    },

    "film_block":                       # Define the modulated pipeline
    {
      "stem" : {                        # Define the Stem
        "spatial_location" : true,      # Inject spatial mesh 
        "mask" : true,                  # Inject resized mask 
        "conv_out": 256,                # number of channels of the stem
        "conv_kernel": [3,3]            # kernel size of the stem
      },

      "resblock" : {                    # Define the modulated pipeline
        "feature_size" : [128, 128, 128, 128],  # Define the number of Modulated resblock and the number of channels
        "spatial_location" : true,      # Inject spatial mesh 
        "mask" : true,                  # Inject resized mask  
        "kernel1" : [1,1],              # kernel size of conv1 in resblock (v2)
        "kernel2" : [3,3]               # kernel size of conv2 in resblock (v2)
      },

      "head" : {
        "spatial_location" : true,      # Inject spatial mesh 
        "mask" : true,                  # Inject resized mask  
        "conv_out": 512,                # number of channels of the stem
        "conv_kernel": [1,1],           # kernel size of the stem
      }
    },

    "pooling" : {                   # Define pooling or attention method at the end
      "mode": "glimpse",            # mean/max/classic/glimpse(MLB)
      "no_attention_mlp": 256,      # projection size for MLB 
      "no_glimpses": 1              # number of glimpse  
    }

    "dropout_keep_prob" : 1.0             # number of glimpse

  },

  "optimizer": {                        # define optimizer
    "no_epoch": 10,                     # number of training epoch
    "learning_rate": 2e-4,              # learning rate
    "batch_size": 64,                   # batch-size
    "clip_val": 10.0,                   # global clip values 
    "weight_decay": 5e-6,               # weight decay applied to "weight_decay_add" - "weight_decay_remove"
    "weight_decay_add": ["film_stack"], 
    "weight_decay_remove": ["feedback_loop", "FiLM_layer" ,"pooling" ,"basic_reading_unit"]
  },

  "seed": -1
}
```

## Reference
 - [1] Strub, F., Seurin, M., Perez, E., de Vries, H., Preux, P., Courville, A., & Pietquin, O. (2018). Visual Reasoning with Multi-hop Feature Modulation. In European Conference on Computer Vision 2018. [pdf](https://arxiv.org/abs/1808.04446)
 - [2] Yu, L., Poirson, P., Yang, S., Berg, A. C., & Berg, T. L. (2016). Modeling context in referring expressions. In European Conference on Computer Vision. [pdf](https://arxiv.org/abs/1608.00272) - [github](https://github.com/lichengunc/refer)

## Acknowledgment
 - SequeL Team
 - Mila Team

We would also like people that help us along the project: Harm de Vries.

The project is part of the CHISTERA - IGLU Project.
