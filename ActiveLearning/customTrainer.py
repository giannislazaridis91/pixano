import os
import subprocess
import pandas as pd
from ActiveLearning.ALearner import getLastRound

relpath_ALmain = "alpha_mix_active_learning/_main.py"

class customTrainer():

    pixano_root = "../../pixano"
    customLearnerCondaEnv = "customLearner"

    model_name="mlp"
    num_classes = None
    max_epochs_per_round=100
    batch_size=16
    learning_rate=0.001
    
    strategy_name="AlphaMixSampling"

    _mode='train'

    #in this trainer we train only on last round
    def __init__(self, db, DATASET_NAME, output_accDir, import_dir, **kwargs):
        self.db = db
        # self.validation_data = validation_data # ---------------------------> remove later
        self.initial_epoch = 0

        # sets new values to any default arguments passed during construction    
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.set_parameter(key,value)

        self.dataset_name = DATASET_NAME
        self.output_dir = output_accDir
        self.import_dir = import_dir

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

        csvAcc=os.path.join(self.output_dir,"accuracy.csv")

        arguments = f"--data_name {self.dataset_name}           \
                        --data_dir {self.import_dir}            \
                        --pixano_root {self.pixano_root}        \
                        --round {round} --mode {self._mode}     \
                        --strategy {self.strategy_name}         \
                        --train_out {csvAcc}                    \
                        --model {self.model_name}               \
                        --n_label {self.num_classes}            \
                        --learning_rate {self.learning_rate}    \
                        --n_epoch {self.max_epochs_per_round}"

        subprocess.run(f"""source ~/miniconda3/etc/profile.d/conda.sh
            conda activate {self.customLearnerCondaEnv} 
            python {relpath_ALmain} {arguments}""", #{customLearner_ROOTDIR}/customLearner_main_3
            shell=True, executable='/bin/bash', check=True)

        trainOut = pd.read_csv(csvAcc,index_col=0)
        return {
            "score": 100 * trainOut.loc["round_"+str(round),"accuracy"]
        }