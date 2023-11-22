## instructions to install AL

> in this directory (pixano/ActiveLearning/certh_integration/231031_mnist/) download clone the forked repository by using the following commands
```
$ git clone https://github.com/pasquale90/alpha_mix_active_learning
$ cd alpha_mix_active_learning
$ git checkout integration-dev
```
then install dependencies into a new conda environment as follows:
```
conda env create --name customLearner python=3.10.10
conda activate customLearner
pip install -r requiremets.txt
```

## instructions to run customAL demo

- run the 1.convert_dataset_2_explorer.ipynb to import MNIST into Pixano
- run the 4.integrateAI.ipynb to run the customAL scenario.
