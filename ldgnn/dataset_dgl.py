import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_graph_dgl
from tqdm import tqdm


class MyDglGraphPropPredDataset(DglGraphPropPredDataset):

    def pre_process(self):
        processed_dir = osp.join(self.root, 'processed')
        raw_dir = osp.join(self.root, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'dgl_data_processed')

        if self.task_type == 'subtoken prediction':
            target_sequence_file_path = osp.join(processed_dir, 'target_sequence')

        if os.path.exists(pre_processed_file_path):

            if self.task_type == 'subtoken prediction':
                self.graphs, _ = load_graphs(pre_processed_file_path)
                self.labels = torch.load(target_sequence_file_path)

            else:
                self.graphs, label_dict = load_graphs(pre_processed_file_path)
                self.labels = label_dict['labels']

        else:
            ### check download
            if self.binary:
                # npz format
                has_necessary_file = osp.exists(osp.join(self.root, 'raw', 'data.npz'))
            else:
                # csv file
                has_necessary_file = osp.exists(osp.join(self.root, 'raw', 'edge.csv.gz'))

            ### download
            if not has_necessary_file:
                url = self.meta_info['url']
                if decide_download(url):
                    path = download_url(url, self.original_root)
                    extract_zip(path, self.original_root)
                    os.unlink(path)
                    # delete folder if there exists
                    try:
                        shutil.rmtree(self.root)
                    except:
                        pass
                    shutil.move(osp.join(self.original_root, self.download_name), self.root)
                else:
                    print('Stop download.')
                    exit(-1)

            ### preprocess
            add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

            if self.meta_info['additional node files'] == 'None':
                additional_node_files = []
            else:
                additional_node_files = self.meta_info['additional node files'].split(',')

            if self.meta_info['additional edge files'] == 'None':
                additional_edge_files = []
            else:
                additional_edge_files = self.meta_info['additional edge files'].split(',')

            graphs = read_graph_dgl(raw_dir, add_inverse_edge=add_inverse_edge,
                                    additional_node_files=additional_node_files,
                                    additional_edge_files=additional_edge_files, binary=self.binary)

            _graphs = []
            for graph in tqdm(graphs):
                _graph = dgl.DGLGraph()
                _graph.add_nodes(graph.number_of_nodes(), data=dict(graph.ndata))
                _graph.add_edges(graph.all_edges()[0], graph.all_edges()[1], data=dict(graph.edata))
                _graphs.append(_graph)
            graphs = _graphs

            if self.task_type == 'subtoken prediction':
                # the downloaded labels are initially joined by ' '
                labels_joined = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                            header=None).values
                # need to split each element into subtokens
                labels = [str(labels_joined[i][0]).split(' ') for i in range(len(labels_joined))]

                print('Saving...')
                save_graphs(pre_processed_file_path, graphs)
                torch.save(labels, target_sequence_file_path)

                ### load preprocessed files
                self.graphs, _ = load_graphs(pre_processed_file_path)
                self.labels = torch.load(target_sequence_file_path)

            else:
                if self.binary:
                    labels = np.load(osp.join(raw_dir, 'graph-label.npz'))['graph_label']
                else:
                    labels = pd.read_csv(osp.join(raw_dir, 'graph-label.csv.gz'), compression='gzip',
                                         header=None).values

                has_nan = np.isnan(labels).any()

                if 'classification' in self.task_type:
                    if has_nan:
                        labels = torch.from_numpy(labels).to(torch.float32)
                    else:
                        labels = torch.from_numpy(labels).to(torch.long)
                else:
                    labels = torch.from_numpy(labels).to(torch.float32)

                print('Saving...')
                save_graphs(pre_processed_file_path, graphs, labels={'labels': labels})

                ### load preprocessed files
                self.graphs, label_dict = load_graphs(pre_processed_file_path)
                self.labels = label_dict['labels']


