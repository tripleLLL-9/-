import numpy as np
import argparse
import glob
import time
import os
import sys
import pickle
import torch
from ExpAndro import ExpAndro, DatasetProcess, PrintColor
import warnings
import networkx as nx
import matplotlib.pyplot as plt

#python main.py --id default-CIC-2 --datapath /home/gongjiacheng/code/CICMalDroid2020 --dataset CIC_8000 --explain

parser = argparse.ArgumentParser(description='The proposed method.')
parser.add_argument('--id', type=str, default='default-CIC-2', help='Marking experiments.')
parser.add_argument('--datapath', type=str, help='Dataset folder location.')
parser.add_argument('--model', type=str, default='ModelAP', help='Choose a specific model.')
parser.add_argument('--dataset', type=str, default='CIC_8000', help='Choose a specific dataset.')
parser.add_argument('--ratio', type=float, default=0.7, help='Ratio of training set.')
parser.add_argument('--epochs', type=int, default=300, help='Train epochs.')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--random_seed', type=int, default=319, help='Random seed.')
parser.add_argument('--retrain_ASTModel', type=bool, default=False, help='Retrain ASTModel or not.')
parser.add_argument('--recollect_AST', type=bool, default=False, help='Recollect AST or not.')
parser.add_argument('--regenerate_dataset', type=bool, default=False, help='Regenerate Dataset or not.')
parser.add_argument('--feature', type=str, default='Full', help='Select Node Features.')
parser.add_argument('--explain', action="store_true", default=False, help='Use GNNExplainer or not.')
args = parser.parse_args()

class Logger():
    def __init__(self, file_name='Default.log',stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, 'a')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass

def Printinfo():
    print('=============================================================')
    print(f'Run_ID: {args.id}')
    print(f'Data_Path: {args.datapath}')
    print(f'Model: {args.model}')
    print(f'Ratio: {args.ratio}')
    print(f'Model Parameters:')
    print(f'\tEpochs: {args.epochs}')
    print(f'\tHidden_dim: {args.hidden_dim}')
    print(f'\tLearning rate: {args.lr}')
    print(f'\tRandom seed: {args.random_seed}')
    print('=============================================================')

def ablation(feature,dataset):
    for data in dataset:
        x = data.x
        x1,x2,x3,x4,x5 = torch.split(x,[1,17,200,402,10],dim=1)
        if feature == 'Null':
            x = torch.zeros_like(x)
        elif feature == 'AST':
            x = torch.hstack([x1,x2,x3])
        elif feature == 'API':
            x = torch.hstack([x1,x2,x4])
        elif feature == 'Network':
            x = torch.hstack([x1,x2,x5])
        elif feature == 'SocialNetwork':
            x = x5
        elif feature == 'Code':
            x = torch.hstack([x1,x2,x3,x4])
        elif feature == 'AST+Network':
            x = torch.hstack([x1,x2,x3,x5])
        elif feature == 'API+Network':
            x = torch.hstack([x1,x2,x4,x5])
        else: print('Wrong feature, use Full feature default')
        data.x = x
    return dataset

def ablation_data(feature,data):
    x = data.x
    x1,x2,x3,x4,x5 = torch.split(x,[1,17,200,402,10],dim=1)
    if feature == 'Null':
        x = torch.zeros_like(x)
    elif feature == 'AST':
        x = torch.hstack([x1,x2,x3])
    elif feature == 'API':
        x = torch.hstack([x1,x2,x4])
    elif feature == 'Network':
        x = torch.hstack([x1,x2,x5])
    elif feature == 'Code':
        x = torch.hstack([x1,x2,x3,x4])
    elif feature == 'AST+Network':
        x = torch.hstack([x1,x2,x3,x5])
    elif feature == 'API+Network':
        x = torch.hstack([x1,x2,x4,x5])
    else: print('Wrong feature, use Full feature default')
    data.x = x
    return data


def SaveGraph(global_g, explanation, path):
    #node_mask = explanation.get('node_mask').cpu()
    edge_mask = explanation.get('edge_mask').cpu()
    print(f'Edge mask\n{edge_mask}')
    #print(f'Node mask\n{node_mask}')
    #g_exp = to_networkx(data_exp)
    plt.figure(figsize=(32, 30)) 
    pos = nx.spring_layout(global_g, k=0.5, iterations=200, scale=2, seed=42) 
    node_incident = 1
    node_colors = np.full(len(global_g.nodes()),'b')
    '''
    node_colors = []
    for i in node_mask:
        if i.item() > 0.5:
            node_colors.append('r')
        else:
            node_colors.append('b')
    #print(type(node_colors))
    '''
    
    edge_incident = 10
    edge_widths = [mask * edge_incident for mask in edge_mask]
    cmap = plt.cm.coolwarm
    #print(type(edge_widths))
    node_size = 500
    nx.draw_networkx_nodes(global_g, pos, node_color=node_colors, cmap=cmap, node_size=node_size)
    #nx.draw_networkx_labels(global_g, pos)
    nx.draw_networkx_edges(global_g, pos, width=edge_widths, edge_color='#D3D3D3')
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap))
    fig_path = path
    plt.savefig(fig_path, dpi=640)
    plt.show()

def ExpPerformance(global_g, exp_g, path):
    node_color = []
    for node in global_g.nodes():
        if node in exp_g.nodes():
            node_color.append('#FCA311')
        else: node_color.append('#13213C')
    edge_color = []
    for edge in global_g.edges():
        if edge in exp_g.edges():
            edge_color.append('#BF1E2E')
        else: edge_color.append('#D3D3D3')
    # 绘制图形
    plt.figure(figsize=(9, 9))
    pos = nx.spring_layout(global_g,k=0.8,seed=314) 
    #pos = nx.shell_layout(global_g)
    nx.draw_networkx_nodes(global_g, pos, node_color=node_color)
    nx.draw_networkx_edges(global_g, pos, edge_color=edge_color, width=3)
    #nx.draw_networkx_labels(global_g, pos)  # 可选：显示节点标签
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(path,dpi=300)  # 保存图像到指定路径
    plt.show()
    



def main():
    warnings.filterwarnings("ignore")
    pc = PrintColor()
    localtime = time.localtime(time.time())
    working_path = '/home/gongjiacheng/code/ExpAndro2.0/'
    if not os.path.exists(working_path):
        os.makedirs(working_path)
    Logpath = working_path + 'Log/'
    if not os.path.exists(Logpath):
        os.makedirs(Logpath)
    Graphpath = working_path + 'Graph/'
    if not os.path.exists(Graphpath):
        os.makedirs(Graphpath)
    Logname = Logpath+f'/{args.id}_{args.model}_{args.dataset}_{localtime.tm_year}y_{localtime.tm_mon:02d}m_{localtime.tm_mday:02d}d_{localtime.tm_hour:02d}h_{localtime.tm_min:02d}m.log'
    sys.stdout = Logger(Logname)
    sys.stderr = Logger(Logname)
    print('=============================================================')
    print(pc.YELLOW+f'Working Path: {working_path}'+pc.END)
    print(pc.YELLOW+f'Log Path: {Logname}'+pc.END)
    print('=============================================================')
    print(torch.cuda.device_count())
    Printinfo()
    ea = ExpAndro(model=args.model,dataset=args.dataset,run_id=args.id,data_path=args.datapath,working_path=working_path)
    dp = DatasetProcess()
    if args.id.endswith('2'):
        num_classes = 2
    elif args.id.endswith('5'):
        num_classes = 5
    else:
        print('Wrong id')
        return
    if args.explain == False:
        print(f'===============================================================')
        print(pc.HIGHLIGHT+"Classification Model"+pc.END)
        if args.retrain_ASTModel == True or ea._astmodel == None:
            APKlist = ea.Get_APKlist()
            ea.ASTModel_generate(APKlist,'Xref',args.recollect_AST)
        if args.regenerate_dataset == True or os.path.exists(ea.dataset_path) == False:
            APKlist, APKlabel = dp.List_byid(args.id,args.datapath)
            ea.ParseAPK(APKlist,APKlabel)
            datalist = dp.Datalist_byid(args.id,ea._datafile_path)
            ea.Dataset_generate(datalist)
        train, test = dp.Loader_byratio(args.id,ea._datafile_path,ea.dataset_path,args.ratio)

        time_dataset = []
        if 'AZ' in args.id:
            time_dataset = dp.Get_AZlist(ea._datafile_path)
        if args.feature != 'Full':
            train = ablation(args.feature,train)
            test = ablation(args.feature,test)
            if time_dataset != []:
                time_dataset = ablation(args.feature,time_dataset)
        model = ea.Model_train_test(args.id,train_dataset=train,test_dataset=test,model=args.model,epochs=args.epochs,hidden_dim=args.hidden_dim,lr=args.lr,random_seed=args.random_seed,ratio=args.ratio,num_classes=num_classes,time_dataset=time_dataset)

    else:
        print(f'===============================================================')
        print(pc.HIGHLIGHT+"Explanation Model"+pc.END)
        filepath = "/home/gongjiacheng/code/ObfusCIC/Original/Benign/0a37f44ab538fce5885680975b2711e276d6dc0cb4f2b1f23b177ae79e5549a9.apk"
        filename = "XMLerror"
        p_id = f'{args.model}_{args.hidden_dim}_{args.lr}_{args.ratio}'
        global_g, data_exp = ea._get_data_exp(filepath, CGType="Xref")
        if args.feature != 'Full':
            data_exp = ablation_data(args.feature,data_exp)
        num_features = data_exp.x.size(1)
        print(f'Num_features: {num_features}')
        init_model = ea._get_init_model(args.model,num_features,args.hidden_dim,num_classes)
        model = ea._load_model_dict(init_model,p_id)
        nodenumber = []
        explanation_times = 1
        final_g = nx.DiGraph()
        for i in range(explanation_times):
            print(pc.GREEN+f'Explanation {i} phase'+pc.END)
            ExpAlgorithm = "GNNExplainer"
            #ExpAlgorithm = "BaseGNNExplainer"
            #ExpAlgorithm = "PGExplainer"
            #ExpAlgorithm = "GraphMaskExplainer"
            explanation = ea.explain(model,global_g,data_exp,ExpAlgorithm=ExpAlgorithm)
            SaveGraph(global_g, explanation, Graphpath+f'{args.id}_{args.feature}_{filename}_{ExpAlgorithm}{i}.png')
            if ExpAlgorithm == "BaseGNNExplainer" or ExpAlgorithm == "GraphMaskExplainer":
                sub_g = ea.exp_subgraph_hard(global_g,explanation,0.5)
            else:
                sub_g = ea.exp_subgraph(global_g,explanation)
            final_g.add_nodes_from(sub_g.nodes())
            final_g.add_edges_from(sub_g.edges())
            nodenumber.append(len(final_g.nodes()))
            print(f'Exp_graph nodes: {len(final_g.nodes())}')
        ea.gml_generate(final_g,f'{args.id}_{args.feature}_{filename}_{ExpAlgorithm}.graphml')
        ea.dot_generate(final_g,f'{args.id}_{args.feature}_{filename}_{ExpAlgorithm}.dot')

        ExpPerformance(global_g, final_g, Graphpath+f'{args.id}_{args.feature}_{filename}_{ExpAlgorithm}.png')

        print(nodenumber)
        for node in global_g.nodes():
            if node.is_external():
                print(pc.YELLOW+f'{node.full_name}'+pc.END)
        #edge_mask = explanation.get('edge_mask')
    
    
if __name__ == '__main__':
    main()