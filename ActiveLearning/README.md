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

> in this directory (pixano/ActiveLearning/) download clone the forked repository by using the following commands
```customLearner
$ git clone https://github.com/pasquale90/alpha_mix_active_learning
$ cd alpha_mix_active_learning
$ git checkout integration-dev
```
then install dependencies into a new conda environment as follows:
```
conda create --name customLearner python=3.10.10
conda activate customLearner
pip install -r requirements.txt
```

## instructions to run customAL demo

1. activate pixano_env

2. open _{dataset}.ipynb and provide values to the following variables : 
- configuration variables
    - ROOTDIR
    - DATASET_NAME
    - datasets_dir
    - library_dir
    - customLearnerCondaEnv
    - ALModule
- internal experimental variables:
    - labels_per_round
    - numInitLabels
    - learning_rate
    - max_epochs_per_round
    - model_name
    - strategy
- external experimental variables
    - num_rounds

3. Run all cells to install mnist dataset and to run the customAL scenario using a custom sampling.

4. To inspect results you may check the following files:
- ROOT/temp_data (accuracy / samples selected)
- console out at the 4.interateAl program
- logs within activeLearning module

5. Check the following folders to validate obtained results for each corresponding dataset supported :
- for the mnist dataset : ROOT/temp_data_Mnist
- for the CIFAR11 dataset : ROOT/temp_data_CIFAR11


## instructions to define a new dataset

For supporting a new dataset, the following actions ought to take place:

- create a new file within pixano/ActiveLearning/_{dataset_name}.ipynb : 
    - download the dataset manually or define a method for downloading the dataset (example in _mnist.ipynb/_cifar11.ipynb)
    - create a _get_dataset() method to access the raw data in the following format.
        - X_train: list of len_train_data containing image_names as strings (i.e. ["0.png","1.png",...,"999.png","1000.png"])
        - Y_train: numpy array of shape (len_train_data,) containing labels of the training data (i.e. array([5, 0, 4, ..., 5, 6, 8], dtype=uint8))
        - X_test: list of len_test_data containing image_names as strings (i.e. ["0.png","1.png",...,"199.png","200.png"])
        - Y_test: numpy array of shape (len_test_data,) containing labels of the training data (i.e. array([5, 0, 4, ..., 5, 6, 8], dtype=uint8))
- within **pixano/ActiveLearning/alpha_mix_learning/_main.py**, add a new if statement and parse data accordingly. Currently there are 3 different TODOS defined, so as to: 
    - train_params = train_params_pool["dataset_name"]
    - the method to load the dataset from the pixano_dir:
        - X_tr = torch.stack(....
        - Y_tr = torch.tensor( ....
        - ....
    - handler = get_handler("dataset_name")

     --> *  HINT: the parsing of the dataset in _main.py should be aligned with the current dataset (i.e. if dataset consist of RGB of Grayscale images)

![Pixano and Active Learning integration process](https://github.com/giannislazaridis91/pixano/blob/main/ActiveLearning/ActiveLearning.png)
