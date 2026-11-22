import numpy as np
import pandas as pd
import argparse
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()

# parser.add_argument('--edge_ver', type=int, default=1, help='edge version.')
# parser.add_argument('--edge_path', type=str, default='/home/lmy/data/scATAC/testing', help='edge file path.')
# parser.add_argument('--root_file_path', type=str, default='/home/lmy/data/scATAC/testing/TF_for_tree.txt', help='root list file path.')
# parser.add_argument('--max_d', type=int, default=2, help='maximun height of the tree.')
# parser.add_argument('--result_path', type=str, default='/home/lmy/data/scATAC/testing', help='the location where the results saved')
#
parser.add_argument('--edge_ver', type=int, default=1, help='edge version.')
parser.add_argument('--edge_path', type=str, default='../example_data/', help='edge file path.')
parser.add_argument('--root_file_path', type=str, default='../example_data/TF_for_tree.txt', help='root list file path.')
parser.add_argument('--max_d', type=int, default=2, help='maximun height of the tree.')
parser.add_argument('--result_path', type=str, default='../example_data/', help='the location where the results saved')



class My_node():
    def __init__(self, name, par, dep):
        self.name = name
        self.par = par
        self.dep = dep
    def __str__(self):
        return "node :%s, node parent:%s, node depth:%s" % (self.name, self.par, self.dep)
    def __repr__(self):
        return "node :%s, node parent:%s, node depth:%s" % (self.name, self.par, self.dep)


def edge_loading(edge_path, edge_ver):
    smesg2name = {}
    with open('../map_data/adata_var_data.csv','r') as file:
        data_lists = file.readlines()
    for line in data_lists:
        line = line.strip('\n').split(',')
        if('SMED' not in line[9]):
            smesg2name[line[0]] =line[9]
    if(edge_ver == 1):
        edge_pos = np.load(edge_path+'/edge_pos_1.npy', allow_pickle=True).item()
        edge_neg = np.load(edge_path+'/edge_neg_1.npy', allow_pickle=True).item()
    else:
        edge_pos = np.load(edge_path+'/edge_pos_2.npy', allow_pickle=True).item()
        edge_neg = np.load(edge_path+'/edge_neg_2.npy', allow_pickle=True).item()
    # 加 smesg000077821对smesg000035181 NFYC对IDH1
    edge_pos['NFYC'].append('IDH1')
    edge_pos_new = {}
    miss_genes =[]
    for gene in edge_pos:
        new_genes = []
        for gene_id in edge_pos[gene]:
            if gene_id in smesg2name:
                new_genes.append(smesg2name[gene_id])
            else:
                new_genes.append(gene_id)
        edge_pos_new[gene] = new_genes
    edge_neg_new = {}
    for gene in edge_neg:
        new_genes = []
        for gene_id in edge_neg[gene]:
            if gene_id in smesg2name:
                new_genes.append(smesg2name[gene_id])
            else:
                new_genes.append(gene_id)
        edge_neg_new[gene] = new_genes
    miss_genes = list(set(miss_genes))
    print(miss_genes)
    return edge_pos_new, edge_neg_new
def root_loading(root_list_path):

    root_list = ['NFYC']

    # root_list_path = '/home/lmy/data/scATAC/testing/TF_for_tree.txt'

    # with open(root_list_path, "r") as f:
    #     for line in f.readlines():
    #         line = line.strip('\n')  #去掉列表中每一个元素的换行符
    #         root_list.append(line)

    return root_list


def run_tree_building(root_list, edge_pos, edge_neg, max_d, result_path):

    for root in root_list:
        if((root not in edge_neg) and (root not in edge_pos)):
            print(root not in edge_neg)

            print(root,'not in the list')
            continue
        
        tree_pos_edge = []
        tree_neg_edge = []

        # bfs with max_depth
        vis_set = set()
        vis_list = []

        # root = 'SOX9'

        root_node = My_node(name=root, par=root, dep=0)

        vis_list.append(root_node)
        vis_set.add(root_node.name)

        while(len(vis_list)>0):
            cur_node = vis_list.pop(0)
            if(cur_node.dep < max_d):
                source = cur_node.name
                # pos weight
                if(edge_pos.get(source) != None):
                    pos_target = edge_pos[source]
                    for target in pos_target:
                        if(target not in vis_set):
                            target_node = My_node(name=target, par=source, dep=cur_node.dep+1)
                            vis_list.append(target_node)
                            vis_set.add(target)
                            tree_pos_edge.append([source, target])

                # neg weight            
                if(edge_neg.get(source) != None):
                    neg_target = edge_neg[source]
                    for target in neg_target:
                        if(target not in vis_set):
                            target_node = My_node(name=target, par=source, dep=cur_node.dep+1)
                            vis_list.append(target_node)
                            vis_set.add(target)
                            tree_neg_edge.append([source, target])

        all_nodes = np.array(list(vis_set))
        tree_neg_edge = np.array(tree_neg_edge)
        tree_pos_edge = np.array(tree_pos_edge)
        
        nodes_info = pd.DataFrame(index=all_nodes, columns=['label'])
        node_label = []
        for node in all_nodes:
            if node.startswith('SMESG'):
                node_label.append('')
            else:
                node_label.append(node)
        nodes_info['label'] = node_label

        nodes_info.to_csv(result_path+'/'+root+'_nodes.csv')

        neg_edge_info = pd.DataFrame(index=np.array(range(len(tree_neg_edge))))
        neg_edge_info['from'] = tree_neg_edge[:,0]
        neg_edge_info['to'] = tree_neg_edge[:,1]
        neg_edge_info.to_csv(result_path+'/'+root+'_neg_edge.csv')

        pos_edge_info = pd.DataFrame(index=np.array(range(len(tree_pos_edge))))
        pos_edge_info['from'] = tree_pos_edge[:,0]
        pos_edge_info['to'] = tree_pos_edge[:,1]
        pos_edge_info.to_csv(result_path+'/'+root+'_pos_edge.csv')
        
        
        
        print(root, 'done!')



#
args = parser.parse_args()
edge_ver = args.edge_ver
root_list_path = args.root_file_path
max_d = args.max_d
edge_path = args.edge_path
result_path = args.result_path

root_list = root_loading(root_list_path=root_list_path)
edge_pos, edge_neg = edge_loading(edge_path=edge_path, edge_ver=edge_ver)
run_tree_building(root_list=root_list, edge_pos=edge_pos, edge_neg=edge_neg, max_d=max_d, result_path=result_path)


data_path_list = ['../example_data/edge_neg_1.npy', '../example_data/edge_neg_2.npy', '../example_data/edge_pos_1.npy', '../example_data/edge_pos_2.npy']

for data_path in data_path_list:
    edge_data = np.load(data_path, allow_pickle=True).item()
    new_path = data_path.replace('.npy','.csv')
    with open(new_path, 'w') as f:
        for gene in edge_data:
            f.write(gene)
            f.write(',')
            f.write(str(edge_data[gene]))
            f.write('\n')



