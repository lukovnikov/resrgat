python main_pyg.py --gnn gcn-virtual --filename code2_gcn_vn_out --removelarge --trainevalinter 5 --use15 --seed 87646464
python main_pyg.py --gnn gcn-virtual --filename code2_gcn_vn_out --removelarge --trainevalinter 5 --use15 --seed 159753
python main_pyg.py --gnn gcn-virtual --filename code2_gcn_vn_out --removelarge --trainevalinter 5 --use15 --seed 42
python main_resrgat.py --gnn resrgat --filename code2_resrgat_out --removelarge --trainevalinter 5 --use15 --drop_ratio 0.1 --lr 0.00025 --batch_size 100 --numreps 2 --numlayers 3 --use_sgru --seed 159753
python main_resrgat.py --gnn resrgat --filename code2_resrgat_out --removelarge --trainevalinter 5 --use15 --drop_ratio 0.1 --lr 0.00025 --batch_size 50 --numreps 2 --numlayers 5 --use_sgru --seed 42
python main_resrgat.py --gnn resrgat --filename code2_resrgat_out --removelarge --trainevalinter 5 --use15 --drop_ratio 0.1 --lr 0.00025 --batch_size 50 --numreps 2 --numlayers 5 --use_sgru --seed 87646464
python main_resrgat.py --gnn resrgat --filename code2_resrgat_out --removelarge --trainevalinter 5 --use15 --drop_ratio 0.1 --lr 0.00025 --batch_size 50 --numreps 2 --numlayers 5 --use_sgru --seed 159753