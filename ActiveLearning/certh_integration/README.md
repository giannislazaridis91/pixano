## For the mnist dataset, torchvision will be used (Keras/Tensorflow dependencies are discarded). Therefore to provide MNIST support, please procceed as follows:
```Pixano
conda activate pixano_env
conda install -c pytorch torchvision 
```
.. you may also need to install chardet ..
```
conda install -c conda-forge chardet
```

## instructions to install AL

> in this directory (pixano/ActiveLearning/certh_integration/231031_mnist/) download clone the forked repository by using the following commands
```customLearner
$ git clone https://github.com/pasquale90/alpha_mix_active_learning
$ cd alpha_mix_active_learning
$ git checkout integration-dev
```
then install dependencies into a new conda environment as follows:
```
conda ycreate --name customLearner python=3.10.10
conda activate customLearner
pip install -r requirements.txt
```

## instructions to run customAL demo

1. activate pixano_env

2. open 1.convert_dataset_2_explorer.ipynb and provide values to the following variables : 
- ROOTDIR
- DATASET_NAME
- datasets_dir
- library_dir
Run all cells to install mnist dataset 


3. open 4.integrateAI.ipynb and provide values to the following variables : 
- configuration variables
- internal experimental variables
- external experimental variables
Run all cells to run the customAL scenario using alphamix sampling.


4. To inspect results you may check the following files:
- ROOT/temp_data (accuracy / samples selected)
- console out at the 4.interateAl program
- logs within activeLearning module

5. Check the following folders to validate obtained results for each corresponding dataset supported :
- for the mnist dataset : ROOT/temp_data_Mnist
- for the CIFAR11 dataset : ROOT/temp_data_CIFAR11