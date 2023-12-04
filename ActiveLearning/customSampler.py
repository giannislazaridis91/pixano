import os
import subprocess
import pandas as pd

relpath_ALmain = "alpha_mix_active_learning/_main.py"

# here define the implementation for the new sampler
class customSampler():
    
    #add all other dependencies define in https://docs.google.com/document/d/1NlArhWYjePzB43sR4HCUc_4xBU73Up9OI24hIyPx0zY/edit
    # for now only the vital ones

    pixano_root = "../../pixano"
    customLearnerCondaEnv = "customLearner"

    model_name = "mlp"
    num_classes = None
    number_init_labels = 100
    labels_per_round = 100 
    strategy_name = "AlphaMixSampling" #EntropySampling #RandomSampling
    alpha_opt = True

    _mode = "query"

    def __init__(self, dataset, DATASET_NAME ,output_queDir, import_dir, **kwargs):
        self.db = dataset
        # super().__init__(dataset)

        # sets new values to any default arguments passed during construction    
        for key, value in kwargs.items():
            if hasattr(self, key):
                self.set_parameter(key,value)
            
        self.dataset_name = DATASET_NAME
        self.output_dir = output_queDir
        self.import_dir = import_dir


    def set_parameter(self,key,value):
        # change member variable members. Public method that can be used outside the scope of the scope
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            print(f'Argument {key} does not exist. Value of {value} does not set any of the member values of the customSampler class')

    def query(self, round):

        csvQue=os.path.join(self.output_dir,"queries_"+str(round)+".csv")

        print(f"csvQue {csvQue}")

        arguments = f"--data_name {self.dataset_name}           \
                        --data_dir {self.import_dir}            \
                        --pixano_root {self.pixano_root}        \
                        --round {round} --mode {self._mode}     \
                        --query_out {csvQue}                    \
                        --model {self.model_name}               \
                        --strategy {self.strategy_name}         \
                        --n_label {self.num_classes}            \
                        --n_init_lb {self.number_init_labels}   \
                        --n_query {self.labels_per_round}"         
        if self.alpha_opt and self.strategy_name=="AlphaMixSampling":
            arguments +=" --alpha_opt"

        subprocess.run(f"""source ~/miniconda3/etc/profile.d/conda.sh
                    conda activate {self.customLearnerCondaEnv} 
                    python {relpath_ALmain} {arguments}""",
                    shell=True, executable='/bin/bash', check=True)
        

        queryOut = pd.read_csv(csvQue,index_col=0)
        
        return queryOut["query_results"].tolist()