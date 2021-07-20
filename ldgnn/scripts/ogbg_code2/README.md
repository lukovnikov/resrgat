We obtained the reported results by using dropout 0.1 and 10 layers with weight sharing between every two layers:

```main_resrgat.py --lr 0.00025 --numlayers 5 --numreps 2 --use_sgru --batch_size 1000 --max_nodes 18000 --epochs 20 --trainevalinter 5 --seed 87646464 --drop_ratio 0.1 --device $GPU```