# BOPO

Official Implementation of paper: **BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization**

## JSP

Implementation of BOPO for JSP.

```shell
# Quick Testing
python JSP/test.py -B 32 -benchmark LA 

# Training
python JSP/train.py -B 256 -K 16 -tag test -epochs 20 
```

## TSP

Implementation of BOPO for TSP.

```shell
# Quick Testing
python TSP/test_n100.py

# Training
python TSP/train_n100.py 
```


## FJSP

Implementation of BOPO for FJSP.

```shell
# Quick Testing
python FJSP/test.py -B 32 -benchmark LA-e 

# Training
python FJSP/train.py -B 256 -K 16 -tag test -epochs 20 
```