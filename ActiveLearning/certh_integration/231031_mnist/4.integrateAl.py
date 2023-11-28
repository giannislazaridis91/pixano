# Configuration variables
DATASET_NAME="MNIST_pixano_v17"
customLearnerCondaEnv="customLearner"

import os
import sys

def insertRootDir(ROOTDIR='pixano'):
    pardir=os.path.dirname(os.path.realpath('__file__'))

    while(os.path.basename(pardir)!=ROOTDIR):

        print(pardir)
        pardir=os.path.dirname(pardir)
        # print(os.path.basename(pardir))
    print("Inserting parent dir : ",pardir)
    sys.path.insert(0,pardir)
    return pardir

ROOTDIR = insertRootDir()



from pathlib import Path
from pixano.data import ImageImporter




library_dir=Path('/home/melissap/_pixano_datasets_') # directory where we have install the pixano formatted dataset
import_dir = library_dir / DATASET_NAME




import random
import lancedb
import pyarrow as pa
import numpy as np
import pandas as pd
from PIL import Image
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.datasets import mnist
from ALearner import (
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
from pixano.utils import natural_key





# utility function to convert id (format "<index>.png") to index
def id_to_idx(id: str) -> int:
    return int(id.split(".")[0])
    # return int(id[0:-4])  #remove the last 4 chars (".png")




mnist_db = lancedb.connect(import_dir)





datasets_dir="/home/melissap/Desktop/LAGO/3.githubs/integration/datasets/MNIST"

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

# (xt,yt),(xte,yte) = get_MNIST(datasets_dir)
(X_train, Y_train), (X_test, Y_test) = get_MNIST(datasets_dir)

import shutil

# TEMPORARY SOLUTION
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









import subprocess

class customTrainer():

    customLearnerCondaEnv = "customLearner3"

    model_name="mlp" 
    max_epochs_per_round=100
    batch_size=16
    learning_rate=0.001
    
    strategy_name="AlphaMixSampling"

    _mode='train'
    

    #in this trainer we train only on last round
    def __init__(self, db, **kwargs):
        self.db = db
        # self.validation_data = validation_data # ---------------------------> remove later
        self.initial_epoch = 0

        # sets new values to any default arguments passed during construction    
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.set_parameter(key,value)
    
    def set_parameter(self,key,value):
        # change member variable members. Public method that can be used outside the scope of the scope
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            print(f'Argument {key} does not exist. Value of {value} does not set any of the member values of the customTrainer class')

    # training on subset data
    def train(self, round, epochs, batch_size): # discard epochs, batch_size

        curRound = getLastRound(self.db)
        assert curRound == round

        # csvAcc=os.path.join(output_accDir,"accuracy"+str(curRound)+".csv")
        csvAcc=os.path.join(output_accDir,"accuracy.csv")

        arguments = f"--data_name {DATASET_NAME} --data_dir {import_dir} --round {round} --mode {self._mode} --strategy {self.strategy_name} --train_out {csvAcc} --model {self.model_name} --learning_rate {self.learning_rate} --n_epoch {self.max_epochs_per_round}"
        subprocess.run(f"""source ~/miniconda3/etc/profile.d/conda.sh
            conda activate {self.customLearnerCondaEnv} 
            python alpha_mix_active_learning/_main.py {arguments}""", #{customLearner_ROOTDIR}/customLearner_main_3
            shell=True, executable='/bin/bash', check=True)

        trainOut = pd.read_csv(csvAcc,index_col=0)
        return {
            "score": 100 * trainOut.loc["round_"+str(round),"accuracy"]
        }









# here define the implementation for the new sampler
class customSampler():
    
    #add all other dependencies define in https://docs.google.com/document/d/1NlArhWYjePzB43sR4HCUc_4xBU73Up9OI24hIyPx0zY/edit
    # for now only the vital ones

    customLearnerCondaEnv = "customLearner3"

    model_name = "mlp"
    number_init_labels = 100
    labels_per_round = 100 
    strategy_name = "AlphaMixSampling" #EntropySampling #RandomSampling
    alpha_opt = True

    _mode = "query"

    def __init__(self, dataset, **kwargs):
        self.db = dataset
        # super().__init__(dataset)

        # sets new values to any default arguments passed during construction    
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.set_parameter(key,value)

    def set_parameter(self,key,value):
        # change member variable members. Public method that can be used outside the scope of the scope
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            print(f'Argument {key} does not exist. Value of {value} does not set any of the member values of the customSampler class')

    def query(self, round):
        # under active development
        # if (round == -1):                                                   # random sampling when labels are absent
        #     ids = getLabelledIds(self.db, round)
        #     return random.sample(ids, labels_per_round)
        # elif (round >= 0):
        # curRound = getLastRound(self.db)

        # import pdb
        # pdb.set_trace()

        # assert round == curRound

        csvQue=os.path.join(output_queDir,"queries_"+str(round)+".csv")

        print(f"csvQue {csvQue}")

        arguments = f"--data_name {DATASET_NAME} --data_dir {import_dir} --round {round} --mode {self._mode} --query_out {csvQue} --model {self.model_name} --strategy {self.strategy_name} -- {self.number_init_labels} --n_query {self.labels_per_round}"
        if self.alpha_opt and self.strategy_name=="AlphaMixSampling":
            arguments +=" --alpha_opt"
        
        subprocess.run(f"""source ~/miniconda3/etc/profile.d/conda.sh
                    conda activate {self.customLearnerCondaEnv} 
                    python alpha_mix_active_learning/_main.py {arguments}""",
                    shell=True, executable='/bin/bash', check=True)
        

        # import pdb
        # pdb.set_trace()

        queryOut = pd.read_csv(csvQue,index_col=0)
        
        return queryOut["query_results"].tolist()



class AutoAnnotator(BaseAnnotator):
    # custom annotation function
    # as we have ground truth for MNIST, we can autofill
    def annotate(self, round):
        candidates = getTaggedIds(self.db, round)
        db_tbl = mnist_db.open_table("db")

        custom_update(db_tbl, f"id in ({ddb_str(candidates)})", 'label', [str(Y_train[id_to_idx(candidate)]) for candidate in sorted(candidates, key=natural_key)])
        print(f"AutoAnnotator: round {round} annotated.")


# variables that could be defined 
labels_per_round=100
numInitLabels = labels_per_round
learning_rate=0.001
max_epochs_per_round=100
model_name="mlp" 
strategy="AlphaMixSampling" #EntropySampling #RandomSampling
alpha_opt=True

# train on candidates data (without resetting weights obviously)
myTrainer = customTrainer(mnist_db, 
                            customLearnerCondaEnv = customLearnerCondaEnv,
                            model_name = model_name, 
                            learning_rate = learning_rate, 
                            max_epochs_per_round = max_epochs_per_round,
                            strategy_name = strategy)

# randomSampler = RandomSampler(mnist_db)
mycustomSampler = customSampler(mnist_db,
                                customLearnerCondaEnv = customLearnerCondaEnv,
                                model_name = model_name, 
                                strategy_name = strategy, 
                                number_init_labels = numInitLabels, 
                                labels_per_round = labels_per_round,
                                alpha_opt = True)


autofillAnnotator = AutoAnnotator(mnist_db)

init_learner = Learner(
    db=mnist_db,
    trainer=myTrainer,
    sampler=mycustomSampler,
    custom_annotator=autofillAnnotator,
    new_al=True,
    verbose=0
)


importTestLabels(mnist_db,(X_test,Y_test))

num_rounds = 10
for round in range(num_rounds):

    # import pdb
    # pdb.set_trace()

    candidates = init_learner.query(round)
    init_learner.annotate(round)
    init_learner.train(round, epochs="nevermind")