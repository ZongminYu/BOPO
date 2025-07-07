# BOPO

Official Implementation of paper: **BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization**

## JSP

Implementation of BOPO for JSP.

First, unzip the training set and validation set.

```shell
unzip JSP/dataset5k/dataset5k.zip
unzip JSP/benchmarks/validation/validation.zip
```

Then, you can start the project quickly:

```shell
# Quick Testing
python JSP/test.py -B 32 -benchmark LA 

# Training
python JSP/train.py -B 256 -K 16 -tag test -epochs 20 
```

## TSP

Implementation of BOPO for TSP.

You can start the project directly:

```shell
# Quick Testing
python TSP/test_n100.py

# Training
python TSP/train_n100.py 
```

## FJSP

Implementation of BOPO for FJSP.

First, unzip the training set and validation set.

```shell
unzip FJSP/dataset5k/dataset5k.zip
unzip FJSP/benchmarks/validation/validation.zip
```

Then, you can start the project quickly:

```shell
# Quick Testing
python FJSP/test.py -B 32 -benchmark LA-e 

# Training
python FJSP/train.py -B 256 -K 16 -tag test -epochs 20 
```

## Requirements

Please see the `requirements.txt`

## Cite

> @inproceedings{
> liao2025bopo,
> title={{BOPO}: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization},
> author={Zijun Liao and Jinbiao Chen and Debing Wang and Zizhen Zhang and Jiahai Wang},
> booktitle={Forty-second International Conference on Machine Learning},
> year={2025},
> }
