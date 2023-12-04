# The root dir name of the current repo (i.e. pixano or pixano-main etc.)
ROOTDIR='pixano'
# name of the dataset
DATASET_NAME="MNIST_pixano_v1"
# directory where the raw mnist dataset will be saved to be transformed latter (images), and also to be used by the active learning auto-annotator (labels)
datasets_dir="/home/melissap/Desktop/LAGO/3.githubs/integration/datasets/MNIST"
# the pixano datasets dir. It is the directory in which the transformed mnist dataset will be saved to be used by Pixano
library_dir="/home/melissap/_pixano_datasets_"
# conda env name builded for running the active learning module as a separate program
customLearnerCondaEnv="customLearner"
# ActiveLearning module's directory
ALModule="ActiveLearning"