# ResRGAT
This repository contains the code for the ICML 2021 paper "Improving Breadth-Wise Backpropagation in Graph Neural Networks Helps Learning Long-Range Dependencies." by Denis Lukovnikov and Asja Fischer.

The code for the "Conditional Recall" experiments is located in the script `ldgnn/scripts/sggnn_recall_grad.py`.

The code for the "Tree Max" experiment is located in the script `ldgnn/scripts/sggnn_treemax.py`.

The code for the experiment on Ogbg-Code2 is based on the example code from OGB and is located in `ldgnn/scripts/ogbg_code2`, where the script `main_resrgat.py` contains the script that we modified from the provided script.

The code for the experiment on ZINC is based on the code from https://github.com/graphdeeplearning/benchmarking-gnns. Our code is contained in our fork of here: https://github.com/lukovnikov/benchmarking-gnns . 