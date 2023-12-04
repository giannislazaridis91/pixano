#!/usr/bin/env python
# coding: utf-8

# # Active Learning with Pixano - MNIST Dataset

# ***!NOTE : Before running this notebook, set the value for the following variables***

# ### Configuration variables

# In[2]:


# The root dir name of the current repo (i.e. pixano or pixano-main etc.)
ROOTDIR='pixano'
# name of the dataset
DATASET_NAME="MNIST_pixano_v2"
# directory where the raw mnist dataset will be saved to be transformed latter (images), and also to be used by the active learning auto-annotator (labels)
datasets_dir="/home/melissap/Desktop/LAGO/3.githubs/integration/datasets/MNIST"
# the pixano datasets dir. It is the directory in which the transformed mnist dataset will be saved to be used by Pixano
library_dir="/home/melissap/_pixano_datasets_"
# conda env name builded for running the active learning module as a separate program
customLearnerCondaEnv="customLearner"
# ActiveLearning module's directory
ALModule="ActiveLearning"


# ### internal experimental variables (that could be defined by the user)

# In[3]:


labels_per_round=100
numInitLabels = labels_per_round
learning_rate=0.001
max_epochs_per_round=100
model_name="mlp" 
strategy="AlphaMixSampling" #i.e. other alternatives may be : EntropySampling #RandomSampling
alpha_opt=True


# ### External experimental variables

# In[4]:


num_rounds = 10


# ***... the rest of the notebook should run without any code adjustments/modifications.***

# # 1.Create the dataset

# In[ ]:


"""
In this section we will convert the MNIST dataset into Pixano Format
!Note: For running, activate the pixano env
"""


# In[5]:


# func for importing ROOT dir to import pixano root module , which is the pixano directory
import os
import sys

def insertRootDir(ROOTDIR='pixano'):
    pardir=os.path.dirname(os.path.realpath('__file__'))

    found = False
    potential_root_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('__file__'))))))

    while(os.path.basename(pardir)!=ROOTDIR):

        # print(pardir)
        pardir=os.path.dirname(pardir)
        

        if (os.path.basename(pardir) == ROOTDIR):
            found = True
            break
        if (pardir == "/" ):
            break
    
    if found:
        print("Inserting parent dir : ",pardir)
        sys.path.insert(0,pardir)
        return pardir
    else:
        print(f"ROOTDIR NOT FOUND. You may have to change ROOTDIR variable from : '{ROOTDIR}' to '{potential_root_dir}'")
        return "_NOT_FOUND_"

ROOTDIR = insertRootDir(ROOTDIR)


# In[6]:


ALModuleDir = os.path.join(ROOTDIR,ALModule)

from pathlib import Path
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import datasets
import lancedb
import pyarrow as pa
from ActiveLearning.ALearner import (
    Learner,
    BaseAnnotator,
    BaseSampler,
    BaseTrainer,
    getLabels,
    getLabelledIds,
    getUnlabelledIds,
    getTaggedIds,
    getLastRound,
    ddb_str,
    custom_update,
    importTestLabels
)
from pixano.apps import Explorer
from pixano.data import ImageImporter
from pixano.utils import natural_key
from ActiveLearning.customTrainer import customTrainer
from ActiveLearning.customSampler import customSampler


# #### method for downloading the dataset

# In[7]:


def get_MNIST(data_dir):
    """
    function for downloading and storing the dataset to the data_dir
    """
    framecounter = 0 
    # downloads mnist and convert it to an image dataset

    image_dir = os.path.join(data_dir,"images")
    annotation_dir = os.path.join(data_dir,"annotations")
    train_imdir = os.path.join(image_dir,"train")
    val_imdir = os.path.join(image_dir,"val")
    test_imdir = os.path.join(image_dir,"test")

    raw_downloaDir = os.path.join(data_dir,"raw_dataset")

    train_anfile = os.path.join(annotation_dir,"train.csv")
    val_anfile = os.path.join(annotation_dir,"val.csv") # not used
    test_anfile = os.path.join(annotation_dir,"test.csv")
    
    if os.path.isdir(image_dir) and os.path.isdir(annotation_dir):
       pass
    else: 
        try:
            os.makedirs(image_dir)
            os.makedirs(annotation_dir)
            os.makedirs(train_imdir)
            os.makedirs(val_imdir)
            os.makedirs(test_imdir)
        except:
            pass

        raw_tr = datasets.MNIST(raw_downloaDir, train=True, download=True)
        raw_te = datasets.MNIST(raw_downloaDir, train=False, download=True)
        X_tr = raw_tr.data
        Y_tr = raw_tr.targets
        X_te = raw_te.data
        Y_te = raw_te.targets

        for i in range(len(X_tr)):
            x=X_tr[i].numpy()
            im = Image.fromarray(x)
            savepath = os.path.join(train_imdir,str(framecounter)+".jpeg")
            im.save(savepath)
            framecounter+=1

        for i in range(len(X_te)):
            x=X_te[i].numpy()
            im = Image.fromarray(x)
            savepath = os.path.join(test_imdir,str(framecounter)+".jpeg")
            im.save(savepath)
            framecounter+=1

        y=pd.DataFrame(Y_tr.numpy())
        y.to_csv(train_anfile,index=False)
            
        y=pd.DataFrame(Y_te.numpy())
        y.to_csv(test_anfile,index=False)


        # remove raw data
        shutil.rmtree(raw_downloaDir)
        print(f'Dataset succesfull downoladed within {data_dir}, framecounter = {framecounter}')
    
    get_MNIST(Path(datasets_dir))


# In[ ]:


# uncomment for help 
# help(ImageImporter.import_dataset)


# #### convert the dataset into pixano format

# In[8]:


# Dataset information
name = "Mnist dataset"
description = "http://yann.lecun.com/exdb/mnist/"
splits = ["train", "test"] # "val",

# Input information
input_dirs = {
    "image": Path(datasets_dir) / "images" #,
    # "objects": library_dir / "annotations",
}

library_dir=Path(library_dir)
import_dir = library_dir / DATASET_NAME #("MNIST_pixano"+"_"+str(dt))


# In[10]:


importer = ImageImporter(name, description, splits)
importer.import_dataset(input_dirs, import_dir, portable=True)


# #### !ERROR: Here we found and issue. explorer doesn't return a localhost port for opening Pixano GUI. A fix is required.

# In[11]:


# explorer = Explorer(library_dir)
# explorer.display()


# # 2. Active Learning

# In[12]:


# utility function to convert id (format "<index>.png") to index
def id_to_idx(id: str) -> int:
    return int(id.split(".")[0])
    # return int(id[0:-4])  #remove the last 4 chars (".png")


# ### Connect to Pixano DB

# In[13]:


mnist_db = lancedb.connect(import_dir)


# ## Model Trainer Object
# 
# We will get raw x_train, x_test, y_test data directly from MNIST.
# 
# 2 proposed Model Trainer Objects, with same model: SimpleTrainer and IncrementalTrainer

# In[14]:


# Overload function for reading the dataset from storage.
def get_MNIST(data_dir):

    image_dir = os.path.join(data_dir,"images")
    train_imdir = os.path.join(image_dir,"train")
    test_imdir = os.path.join(image_dir,"test")

    annotation_dir = os.path.join(data_dir,"annotations")
    train_anfile = os.path.join(annotation_dir,"train.csv")
    test_anfile = os.path.join(annotation_dir,"test.csv")

    # read sorted file names     
    X_train = sorted(os.listdir(train_imdir), key=lambda x: int(x.split('.', 1)[0])) 
    X_test = sorted(os.listdir(test_imdir), key=lambda x: int(x.split('.', 1)[0]))
    
    # X_train = np.array([Image.open(os.path.join(train_imdir,x)) for x in X_train])
    # X_test = np.array([Image.open(os.path.join(test_imdir,x)) for x in X_test])

    # # read labels
    Y_train = pd.read_csv(train_anfile).values #.to_numpy(dtype=np.uint8)
    Y_test = pd.read_csv(test_anfile).values #.to_numpy(dtype=np.uint8)

    Y_train = np.array([y[0] for y in Y_train], dtype=np.uint8)
    Y_test = np.array([y[0] for y in Y_test], dtype=np.uint8)

    return (X_train,Y_train),(X_test,Y_test)

# reminder : datasets_dir is the  local directory sto store the raw dataset
(X_train, Y_train), (X_test, Y_test) = get_MNIST(datasets_dir)


# ## Query Sampler Object
# <!-- RandomSampler or SequentialSampler -->

# #### Custom Trainer

# > prepare the directories for data exchange between pixano and annotation tool

# In[15]:


import shutil

# TEMPORARY SOLUTION FOR EXCHANGING DATA BETWEEN PIXANO AND AL
def create_dir(path):
    try:
        if (os.path.basename(path)== "temp_data" and os.path.exists(path)):
            shutil.rmtree(path) # erase the previous results
        os.makedirs(path)
    except:
        print(f'Dir {path} exists already')
    return path

# here define the paths of exchanging data between pixano and the customLearner
temp_data_exchange_dir = create_dir(os.path.join(ROOTDIR,"temp_data"))                # define a directory for exchanging data
output_queDir = create_dir(os.path.join(temp_data_exchange_dir,"output_queries"))       # [out] query strategy results
output_accDir = create_dir(os.path.join(temp_data_exchange_dir,"output_accuracy"))      # [out] accuracy results 


# ## Labeling Interface Objects
# 
# Human labeling with Pixano Annotator is built-in, here we specify an Auto Annotator

# In[16]:


class AutoAnnotator(BaseAnnotator):
    # custom annotation function
    # as we have ground truth for MNIST, we can autofill
    def annotate(self, round):
        candidates = getTaggedIds(self.db, round)
        db_tbl = mnist_db.open_table("db")
        custom_update(db_tbl, f"id in ({ddb_str(candidates)})", 'label', [str(Y_train[id_to_idx(candidate)]) for candidate in sorted(candidates, key=natural_key)])
        print(f"AutoAnnotator: round {round} annotated.")


# ## Orchestrator

# ### Initial Learning

# In[17]:


myTrainer = customTrainer(mnist_db, 
                            DATASET_NAME = DATASET_NAME,
                            output_accDir = output_accDir,
                            import_dir = import_dir,
                            customLearnerCondaEnv = customLearnerCondaEnv,
                            model_name = model_name, 
                            learning_rate = learning_rate, 
                            max_epochs_per_round = max_epochs_per_round,
                            strategy_name = strategy)

mycustomSampler = customSampler(mnist_db,
                                DATASET_NAME = DATASET_NAME,
                                output_queDir = output_queDir,
                                import_dir = import_dir,
                                customLearnerCondaEnv = customLearnerCondaEnv,
                                model_name = model_name, 
                                strategy_name = strategy, 
                                number_init_labels = numInitLabels, 
                                labels_per_round = labels_per_round,
                                alpha_opt = True)

myTrainer.set_parameter("pixano_root",ROOTDIR)
mycustomSampler.set_parameter("pixano_root",ROOTDIR)

autofillAnnotator = AutoAnnotator(mnist_db)

init_learner = Learner(
    db=mnist_db,
    trainer=myTrainer,
    sampler=mycustomSampler,
    custom_annotator=autofillAnnotator,
    new_al=True,
    verbose=0
)


# In[18]:


importTestLabels(mnist_db,(X_test,Y_test))


# ### Active Learning - Human annotation with Pixano Annotator
# 
# We add some auto-annotation rounds

# In[20]:


for round in range(num_rounds):
    candidates = init_learner.query(round)
    init_learner.annotate(round)
    init_learner.train(round, epochs="nevermind")

