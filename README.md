## Tensor Decomposition-Based Temporal Knowledge Graph Embedding
This repository contains code for the reprsentation proposed in Tensor Decomposition-Based Temporal Knowledge Graph Embedding paper.
## Installation
- Create a conda environment:
```
$ conda create -n T_simple python=3.6 anaconda
```
- Run
```
$ source activate T_simple
```
- Change directory to T_simple folder
- Run
```
$ pip install -r requirements.txt
```
## How to use?
After installing the requirements, run the following command to reproduce results for T_SimplE:
```
$ python main.py -dropout 0.4 -se_prop 0.68 -model T_simple
```
To reproduce the results for T_distmult and T_complex, specify **model** as T_distmult/T_complex as following.
```
$ python main.py -dropout 0.4 -se_prop 0.36 -model T_distmult
$ python main.py -dropout 0.4 -se_prop 0.36 -model T_complex
```
