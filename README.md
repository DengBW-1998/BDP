# BDP

This repository contains the demo code of the paper: 

> BDP: Bipartite Graph Adversarial Defense Algorithm Based on Graph Purification. 

If you have any question or you find any bug about codes, please contact me at wy727100600@163.com


## Environment settings

- python==3.6.13
- numpy==1.19.2
- deeprobust==0.2.4
- pytorch==1.10.1
- tensorflow==2.6.0
- gensim==3.8.3


## Basic Usage

**Usage**

You can run the following different main codes file to set adversarial attack enviroment
If you want to run BBAI:
- python BBAI.py
- python cln.py
- python rndAttack.py
- python test_dice.py

If you want to set abversarial defense enviroment, you can change hyperparameters "defense" and "dp":
- nodefense with defense='nodefense' and dp=False
- bdp with defense='bdp' and dp=False
- prognn with defense='prognn' and dp=False
- stable with defense='stable' and dp=False
- supergat(dp) with defense='nodefense' and dp=True

You can know other hyperparameters' intruductions in the file BBAI.py