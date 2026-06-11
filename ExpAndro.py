import os
from AnalyzeModule import Analyzer
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from GNNMoudle import GAT,GATv2,GCN,GCNv2,ModelA,ModelB,ModelAP,ModelBP,ModelAPP,ModelBPP,GIN,GraphSAGE,MGNN
#from torch_geometric.nn import GraphSAGE
import networkx as nx
import numpy as np
import time
import sys
import multiprocessing as mp
import glob
import jieba
from sklearn import metrics
import matplotlib.pyplot as plt
from colorama import Fore, Style
from colorama import init
import traceback
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer, CaptumExplainer, PGExplainer, GraphMaskExplainer
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from captum import attr
import re
from torch_geometric.nn.conv import MessagePassing


class DatasetProcess():
    def __init__(self):
        pass

    def List_byid(self,runid,FilePath):
        Benignlist = []
        Malwarelist = []
        if "CIC" in runid:
            Benignlist = glob.glob(FilePath+'/Benign/*',recursive=True)
            Adwarelist = glob.glob(FilePath+'/Adware/*',recursive=True)
            Bankinglist = glob.glob(FilePath+'/Banking/*',recursive=True)
            Riskwarelist = glob.glob(FilePath+'/Riskware/*',recursive=True)
            SMSlist = glob.glob(FilePath+'/SMS/*',recursive=True)
            if runid.endswith('2'):
                Malwarelist = Adwarelist + Bankinglist + Riskwarelist + SMSlist
                apk_list = Benignlist + Malwarelist
                apk_label = np.concatenate((np.full((1,len(Benignlist)),0),np.full((1,len(Malwarelist)),1)),axis=1).squeeze()
            elif runid.endswith('5'):
                apk_list = Benignlist + Adwarelist + Bankinglist + Riskwarelist + SMSlist
                apk_label = np.concatenate(np.full(((1,len(Benignlist)),0),np.full((1,len(Adwarelist)),1),np.full((1,len(Bankinglist)),2),np.full((1,len(Riskwarelist)),3),np.full((1,len(SMSlist)),4)),axis=1).squeeze()
            else:
                print('Wrong id')  
        elif "AZ" in runid:
            benlist10 = glob.glob(FilePath+'/Benign/2010/*',recursive=True)
            mallist10 = glob.glob(FilePath+'/Malware_10/2010/*',recursive=True)
            benlist11 = glob.glob(FilePath+'/Benign/2011/*',recursive=True)
            mallist11 = glob.glob(FilePath+'/Malware_10/2011/*',recursive=True)
            benlist12 = glob.glob(FilePath+'/Benign/2012/*',recursive=True)
            mallist12 = glob.glob(FilePath+'/Malware_10/2012/*',recursive=True)
            benlist13 = glob.glob(FilePath+'/Benign/2013/*',recursive=True)
            mallist13 = glob.glob(FilePath+'/Malware_10/2013/*',recursive=True)
            benlist14 = glob.glob(FilePath+'/Benign/2014/*',recursive=True)
            mallist14 = glob.glob(FilePath+'/Malware_10/2014/*',recursive=True)
            benlist15 = glob.glob(FilePath+'/Benign/2015/*',recursive=True)
            mallist15 = glob.glob(FilePath+'/Malware_10/2015/*',recursive=True)
            benlist16 = glob.glob(FilePath+'/Benign/2016/*',recursive=True)
            mallist16 = glob.glob(FilePath+'/Malware_10/2016/*',recursive=True)
            benlist17 = glob.glob(FilePath+'/Benign/2017/*',recursive=True)
            mallist17 = glob.glob(FilePath+'/Malware_10/2017/*',recursive=True)
            benlist18 = glob.glob(FilePath+'/Benign/2018/*',recursive=True)
            mallist18 = glob.glob(FilePath+'/Malware_10/2018/*',recursive=True)
            benlist19 = glob.glob(FilePath+'/Benign/2019/*',recursive=True)
            mallist19 = glob.glob(FilePath+'/Malware_10/2019/*',recursive=True)
            benlist20 = glob.glob(FilePath+'/Benign/2020/*',recursive=True)
            mallist20 = glob.glob(FilePath+'/Malware_10/2020/*',recursive=True)
            benlist21 = glob.glob(FilePath+'/Benign/2021/*',recursive=True)
            mallist21 = glob.glob(FilePath+'/Malware_10/2021/*',recursive=True)
            benlist22 = glob.glob(FilePath+'/Benign/2022/*',recursive=True)
            mallist22 = glob.glob(FilePath+'/Malware_10/2022/*',recursive=True)
            
            Benignlist = benlist10+benlist11+benlist12+benlist13+benlist14+benlist15+benlist16+benlist17+benlist18+benlist19+benlist20+benlist21+benlist22
            Malwarelist = mallist10+mallist11+mallist12+mallist13+mallist14+mallist15+mallist16+mallist17+mallist18+mallist19+mallist20+mallist21+mallist22
            
            apk_list = Benignlist + Malwarelist
            apk_label = np.concatenate((np.full((1,len(Benignlist)),0),np.full((1,len(Malwarelist)),1)),axis=1).squeeze()
        else:
            print('Wrong id')
        return apk_list, apk_label

    def Datalist_byid(self,runid,FilePath):
        if "CIC" in runid:
            Benignlist = glob.glob(FilePath+'_Benign_*.pkl')
            Adwarelist = glob.glob(FilePath+'_Adware_*.pkl')
            Bankinglist = glob.glob(FilePath+'_Banking_*.pkl')
            Riskwarelist = glob.glob(FilePath+'_Riskware_*.pkl')
            SMSlist = glob.glob(FilePath+'_SMS_*.pkl')
            datalist = Benignlist+Adwarelist+Bankinglist+Riskwarelist+SMSlist
        elif "AZ" in runid:
            benlist10 = glob.glob(FilePath+'_Benign_2010_*.pkl')
            benlist11 = glob.glob(FilePath+'_Benign_2011_*.pkl')
            benlist12 = glob.glob(FilePath+'_Benign_2012_*.pkl')
            benlist13 = glob.glob(FilePath+'_Benign_2013_*.pkl')
            benlist14 = glob.glob(FilePath+'_Benign_2014_*.pkl')
            benlist15 = glob.glob(FilePath+'_Benign_2015_*.pkl')
            benlist16 = glob.glob(FilePath+'_Benign_2016_*.pkl')
            benlist17 = glob.glob(FilePath+'_Benign_2017_*.pkl')
            benlist18 = glob.glob(FilePath+'_Benign_2018_*.pkl')
            mallist10 = glob.glob(FilePath+'_Malware_10_2010_*.pkl')
            mallist11 = glob.glob(FilePath+'_Malware_10_2011_*.pkl')
            mallist12 = glob.glob(FilePath+'_Malware_10_2012_*.pkl')
            mallist13 = glob.glob(FilePath+'_Malware_10_2013_*.pkl')
            mallist14 = glob.glob(FilePath+'_Malware_10_2014_*.pkl')
            mallist15 = glob.glob(FilePath+'_Malware_10_2015_*.pkl')
            mallist16 = glob.glob(FilePath+'_Malware_10_2016_*.pkl')
            mallist17 = glob.glob(FilePath+'_Malware_10_2017_*.pkl')
            mallist18 = glob.glob(FilePath+'_Malware_10_2018_*.pkl')
            datalist = benlist10+benlist11+benlist12+benlist13+benlist14+benlist15+benlist16+benlist17+benlist18+mallist10+mallist11+mallist12+mallist13+mallist14+mallist15+mallist16+mallist17+mallist18
        else:
            print('Wrong id')    
        return datalist

    def Loader_byratio(self,runid,FilePath,dataset_path,ratio):
        f = open(dataset_path,'rb')
        dataset = pickle.load(f)
        f.close()
        train_dataset = []
        test_dataset = []
            
        if "CIC" in runid:
            Benignlist = glob.glob(FilePath+'_Benign_*.pkl')
            Adwarelist = glob.glob(FilePath+'_Adware_*.pkl')
            Bankinglist = glob.glob(FilePath+'_Banking_*.pkl')
            Riskwarelist = glob.glob(FilePath+'_Riskware_*.pkl')
            SMSlist = glob.glob(FilePath+'_SMS_*.pkl')
            len1 = len(Benignlist)
            offset1 = int(len1*ratio)
            len2 = len(Adwarelist)
            offset2 = int(len2*ratio)
            len3 = len(Bankinglist)
            offset3 = int(len3*ratio)
            len4 = len(Riskwarelist)
            offset4 = int(len4*ratio)
            len5 = len(SMSlist)
            offset5 = int(len5*ratio)
            if runid.endswith('5'):
                for data in dataset[len1:len1+len2]:
                    data.y = 1
                for data in dataset[len1+len2:len1+len2+len3]:
                    data.y = 2
                for data in dataset[len1+len2+len3:len1+len2+len3+len4]:
                    data.y = 3
                for data in dataset[len1+len2+len3+len4:]:
                    data.y = 4
            train_dataset = dataset[:offset1]+dataset[len1:len1+offset2]+dataset[len1+len2:len1+len2+offset3]+dataset[len1+len2+len3:len1+len2+len3+offset4]+dataset[len1+len2+len3+len4:len1+len2+len3+len4+offset5]
            test_dataset = dataset[offset1:len1]+dataset[len1+offset2:len1+len2]+dataset[len1+len2+offset3:len1+len2+len3]+dataset[len1+len2+len3+offset4:len1+len2+len3+len4]+dataset[len1+len2+len3+len4+offset5:]
            print(f'Total Samples: {len(train_dataset)+len(test_dataset):05d}')
            print(f'\tBenign Samples: {len(Benignlist):05d}')
            print(f'\tAdware Samples: {len(Adwarelist):05d}')
            print(f'\tBanking Samples: {len(Bankinglist):05d}')
            print(f'\tRiskware Samples: {len(Riskwarelist):05d}')
            print(f'\tSMS Samples: {len(SMSlist):05d}')
            print(f'Train Samples: {len(train_dataset):05d}')
            print(f'Test Samples:  {len(test_dataset):05d}')
        elif "AZ" in runid:
            benlist10 = glob.glob(FilePath+'_Benign_2010_*.pkl')
            benlist11 = glob.glob(FilePath+'_Benign_2011_*.pkl')
            benlist12 = glob.glob(FilePath+'_Benign_2012_*.pkl')
            benlist13 = glob.glob(FilePath+'_Benign_2013_*.pkl')
            benlist14 = glob.glob(FilePath+'_Benign_2014_*.pkl')
            benlist15 = glob.glob(FilePath+'_Benign_2015_*.pkl')
            benlist16 = glob.glob(FilePath+'_Benign_2016_*.pkl')
            benlist17 = glob.glob(FilePath+'_Benign_2017_*.pkl')
            benlist18 = glob.glob(FilePath+'_Benign_2018_*.pkl')
            mallist10 = glob.glob(FilePath+'_Malware_10_2010_*.pkl')
            mallist11 = glob.glob(FilePath+'_Malware_10_2011_*.pkl')
            mallist12 = glob.glob(FilePath+'_Malware_10_2012_*.pkl')
            mallist13 = glob.glob(FilePath+'_Malware_10_2013_*.pkl')
            mallist14 = glob.glob(FilePath+'_Malware_10_2014_*.pkl')
            mallist15 = glob.glob(FilePath+'_Malware_10_2015_*.pkl')
            mallist16 = glob.glob(FilePath+'_Malware_10_2016_*.pkl')
            mallist17 = glob.glob(FilePath+'_Malware_10_2017_*.pkl')
            mallist18 = glob.glob(FilePath+'_Malware_10_2018_*.pkl')
            #Benignlist = benlist10+benlist11+benlist12+benlist13+benlist14+benlist15+benlist16+benlist17+benlist18
            #Malwarelist = mallist10+mallist11+mallist12+mallist13+mallist14+mallist15+mallist16+mallist17+mallist18
            len1 = len(benlist10)
            off1 = int(len1*ratio)
            stt1 = len1
            len2 = len(benlist11)
            off2 = stt1 + int(len2*ratio)
            stt2 = stt1 + len2
            len3 = len(benlist12)
            off3 = stt2 + int(len3*ratio)
            stt3 = stt2 + len3
            len4 = len(benlist13)
            off4 = stt3 + int(len4*ratio)
            stt4 = stt3 + len4
            len5 = len(benlist14)
            off5 = stt4 + int(len5*ratio)
            stt5 = stt4 + len5
            len6 = len(benlist15)
            off6 = stt5 + int(len6*ratio)
            stt6 = stt5 + len6
            len7 = len(benlist16)
            off7 = stt6 + int(len7*ratio)
            stt7 = stt6 + len7
            len8 = len(benlist17)
            off8 = stt7 + int(len8*ratio)
            stt8 = stt7 + len8
            len9 = len(benlist18)
            off9 = stt8 + int(len9*ratio)
            stt9 = stt8 + len9
            
            len10 = len(mallist10)
            off10 = stt9 + int(len10*ratio)
            stt10 = stt9 + len10
            len11 = len(mallist11)
            off11 = stt10 + int(len11*ratio)
            stt11 = stt10 + len11
            len12 = len(mallist12)
            off12 = stt11 + int(len12*ratio)
            stt12 = stt11 + len12
            len13 = len(mallist13)
            off13 = stt12 + int(len13*ratio)
            stt13 = stt12 + len13
            len14 = len(mallist14)
            off14 = stt13 + int(len14*ratio)
            stt14 = stt13 + len14
            len15 = len(mallist15)
            off15 = stt14 + int(len15*ratio)
            stt15 = stt14 + len15
            len16 = len(mallist16)
            off16 = stt15 + int(len16*ratio)
            stt16 = stt15 + len16
            len17 = len(mallist17)
            off17 = stt16 + int(len17*ratio)
            stt17 = stt16 + len17
            len18 = len(mallist18)
            off18 = stt17 + int(len18*ratio)
            stt18 = stt17 + len18
            train_dataset = dataset[:off1]+dataset[stt1:off2]+dataset[stt2:off3]+dataset[stt3:off4]+dataset[stt4:off5]+dataset[stt5:off6]+dataset[stt6:off7]+dataset[stt7:off8]+dataset[stt8:off9]+dataset[stt9:off10]+dataset[stt10:off11]+dataset[stt11:off12]+dataset[stt12:off13]+dataset[stt13:off14]+dataset[stt14:off15]+dataset[stt15:off16]+dataset[stt16:off17]+dataset[stt17:off18]
            
            test_dataset = dataset[off1:stt1]+dataset[off2:stt2]+dataset[off3:stt3]+dataset[off4:stt4]+dataset[off5:stt5]+dataset[off6:stt6]+dataset[off7:stt7]+dataset[off8:stt8]+dataset[off9:stt9]+dataset[off10:stt10]+dataset[off11:stt11]+dataset[off12:stt12]+dataset[off13:stt13]+dataset[off14:stt14]+dataset[off15:stt15]+dataset[off16:stt16]+dataset[off17:stt17]+dataset[off18:]
            print(f'Total Samples: {len(train_dataset)+len(test_dataset):05d}')
            print(f'\tYear\tBenign\tMalware')
            print(f'\t2010\t{len(benlist10):05d}\t{len(mallist10):05d}')
            print(f'\t2011\t{len(benlist11):05d}\t{len(mallist11):05d}')
            print(f'\t2012\t{len(benlist12):05d}\t{len(mallist12):05d}')
            print(f'\t2013\t{len(benlist13):05d}\t{len(mallist13):05d}')
            print(f'\t2014\t{len(benlist14):05d}\t{len(mallist14):05d}')
            print(f'\t2015\t{len(benlist15):05d}\t{len(mallist15):05d}')
            print(f'\t2016\t{len(benlist16):05d}\t{len(mallist16):05d}')
            print(f'\t2017\t{len(benlist17):05d}\t{len(mallist17):05d}')
            print(f'\t2018\t{len(benlist18):05d}\t{len(mallist18):05d}')
            print(f'Train Samples: {len(train_dataset):05d}')
            print(f'Test Samples:  {len(test_dataset):05d}')   
        else:
            print('Wrong id')
        return train_dataset, test_dataset

    def Get_AZlist(self,FilePath):
        datalist = []
        #benlist17 = glob.glob(FilePath+'_Benign_2017_*.pkl')
        #benlist18 = glob.glob(FilePath+'_Benign_2018_*.pkl')
        benlist19 = glob.glob(FilePath+'_Benign_2019_*.pkl')
        benlist20 = glob.glob(FilePath+'_Benign_2020_*.pkl')
        benlist21 = glob.glob(FilePath+'_Benign_2021_*.pkl')
        benlist22 = glob.glob(FilePath+'_Benign_2022_*.pkl')
        #mallist17 = glob.glob(FilePath+'_Malware_10_2017_*.pkl')
        #mallist18 = glob.glob(FilePath+'_Malware_10_2018_*.pkl')
        mallist19 = glob.glob(FilePath+'_Malware_10_2019_*.pkl')
        mallist20 = glob.glob(FilePath+'_Malware_10_2020_*.pkl')
        mallist21 = glob.glob(FilePath+'_Malware_10_2021_*.pkl')
        mallist22 = glob.glob(FilePath+'_Malware_10_2022_*.pkl')
        #list17 = benlist17+mallist17
        #list18 = benlist18+mallist18
        list19 = benlist19+mallist19
        list20 = benlist20+mallist20
        list21 = benlist21+mallist21
        list22 = benlist22+mallist22
        #datalist.append(list17)
        #datalist.append(list18)
        datalist.append(list19)
        datalist.append(list20)
        datalist.append(list21)
        datalist.append(list22)
        return datalist

class PrintColor():
    def __init__(self):
        self.END = "\033[0m"
        self.RED = "\033[0;31;40m"
        self.GREEN = "\033[0;32;40m"
        self.YELLOW = "\033[0;33;40m"
        self.BLUE = "\033[0;34;40m"
        self.PURPLERED = "\033[0;35;40m"
        self.GREENBLUE = "\033[0;36;40m"
        self.WHITE = "\033[0;37;40m"
        self.HIGHLIGHT = "\033[0;30;47m"


class ExpAndro():
    def __init__(self, model="default", dataset="default", run_id="default", data_path = '', working_path="./"):
        '''
        model: name of choosen GNN model
        dataset: id of dataset, dataset-num_classes
        run_id: id of experiment
        data_path: source APK data path
        working_path: basic working path
        '''
        #init tool
        pc = PrintColor()
        #init device
        #self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda')
        #list of method descriptor -> str
        self.flagslist = ['public','private','protected','static','final','synchronized','bridge','varargs','native','interface','abstract','strictfp','synthetic','enum','unused','constructor','declared_synchronized']
        #name of GNN -> str
        self._model_name = model
        #id of dataset -> str
        self._datasetstr = dataset
        #id of experiments -> str
        self.runid = run_id
        #check working path
        if not working_path.endswith('/'):
            working_path += '/'
        self._basic_wp = working_path
        #APK analyzer
        self._analyzer = Analyzer(self._basic_wp)
        #check source data path
        self._data_path = data_path
        #working path of ExpAndro
        self._working_path = working_path + "Output/ExpAndro_Data/"
        if not os.path.exists(self._working_path):
            os.makedirs(self._working_path)
        #path to save related file
        self._relatedfile_path = self._working_path + 'RelatedFile/'
        if not os.path.exists(self._relatedfile_path):
            os.makedirs(self._relatedfile_path)
        #path to save related file
        self._relatedgraph_path = self._relatedfile_path + 'Graph/'
        if not os.path.exists(self._relatedgraph_path):
            os.makedirs(self._relatedgraph_path)
        #trained GNN model
        self._model = None
        #path to save temp analysis files
        self._datafile_path = self._working_path + self._datasetstr + '/'
        #path to save AST corups
        self.corups_path = self._relatedfile_path + self._datasetstr + '_corups.pkl'
        #path to save AST2Vec model
        self.astmodel_path = self._relatedfile_path + self._datasetstr + '_astmodel'
        self.fname = get_tmpfile(self.astmodel_path)
        #path to save dataset
        self.dataset_path = self._relatedfile_path + self._datasetstr + '_dataset.pkl'
        #try to load AST2Vec model
        try:
            self._astmodel = Doc2Vec.load(self.fname)
            print(f"AST2Vec model {self.astmodel_path} load succeed!")
        except:
            self._astmodel = None  
            print(f"AST2Vec model {self.astmodel_path} load failed!")
        self.feature = None  

    def set_feature(self,feature):
        self.feature = feature  

    def _ASTCorups_Collect(self, Filelist, CGType='Xref'):
        count = 1
        for file in Filelist:
            try:
                print(f'[{count:04d}] {file}')
                scg = self._analyzer.AnalyzeAPK(file, CGType)
                self._analyzer.Feature_collection(scg)
                print(Fore.GREEN+f'Collect done')
                print(Style.RESET_ALL)
                print('=======================================================')
                count+=1
            except:
                #os.remove(file)
                print(Fore.RED+f'Collect error, wrong file {file}')
                print(Style.RESET_ALL)
                print('=======================================================')
                pass
    
    def ASTModel_generate(self, APKlist, CGtype='Xref', recollect = False):
        if recollect == True or os.path.exists(self.corups_path) == False:
            if os.path.exists(self.corups_path):
                os.remove(self.corups_path)

            print('\nCollect ASTs of APK Files')
            print('=======================================================')
            print('\nCollect start')
            
            self._ASTCorups_Collect(APKlist,CGtype)
            
            print('\nCollect done')
            print('=======================================================')
            with open(self.corups_path,'wb') as f:
                pickle.dump(self._analyzer.corups,f)
        with open(self.corups_path,'rb') as f:
            read_corups = pickle.load(f)
        start_time = time.time()
        Docmodel(self.fname, read_corups, vector_size = 200, epochs = 50, workers = 8)
        end_time = time.time()
        print("Doc2Vec Model training time: ",end_time-start_time)
        self._astmodel = Doc2Vec.load(self.fname)
           
    def _data_generate(self, scg, class_label = 0):
        #if SCG has no node, then the feature vector will be set as 1*623[every element is 0]
        vector_len = 630
        if len(scg.nodes()) == 0:
            scg.add_node(1)
            adj = nx.to_scipy_sparse_array(scg).tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row,col],dim = 0)
            y = class_label
            x = torch.zeros(vector_len,dtype = torch.float32)
            x = torch.tensor([x.tolist()])
        else:
            coder = self._astmodel
            #Computing graph structure features via the graph algorithm library in networkx
            degree_centrality = nx.degree_centrality(scg)
            in_degree_centrality = nx.in_degree_centrality(scg)
            out_degree_centrality = nx.out_degree_centrality(scg)
            katz_centrality = nx.katz_centrality(nx.DiGraph(scg))
            closeness_centrality = nx.closeness_centrality(scg)
            betweenness_centrality = nx.betweenness_centrality(scg)
            harmonic_centrality = nx.harmonic_centrality(scg)
            #trophic_levels = nx.trophic_levels(scg)
            clustering = nx.clustering(nx.DiGraph(scg))
            square_clustering = nx.square_clustering(scg)
            pagerank = nx.pagerank(scg)

            adj = nx.to_scipy_sparse_array(scg).tocoo()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row,col],dim = 0)

            y = class_label

            x = None
            for node in scg.nodes():
                if node.is_external() == False:
                    AST = self._analyzer.get_ast_method(node)
                    ASTText = str(AST)
                    ASTText_processed = cut_doc(ASTText)
                    coder.random.seed(0)
                    vector = coder.infer_vector(ASTText_processed)
                    vector = torch.from_numpy(vector)
                    vector = torch.cat((torch.tensor([0]),vector),0)    #0 is internal method
                    flag = AST['flags']
                    for i in self.flagslist:
                        if i in flag:
                            vector = torch.cat((vector,torch.tensor([1])),0)
                        else:
                            vector = torch.cat((vector,torch.tensor([0])),0)
                    vector = torch.cat((vector,torch.zeros(len(self._analyzer.sapi)+1,dtype = torch.float32)),0)
                else:
                    vector = torch.zeros(217,dtype = torch.float32)
                    vector = torch.cat((torch.tensor([1]),vector),0)    #1 is external method
                    sapicount = 0
                    for i in self._analyzer.sapi:
                        classname, apiname = self._analyzer.api_getname(i)
                        apiname = apiname[1:-1]
                        if classname == node.get_class_name() and apiname == node.name:
                            vector = torch.cat((vector,torch.tensor([1])),0)
                            sapicount = sapicount + 1
                        else:
                            vector = torch.cat((vector,torch.tensor([0])),0)
                    if sapicount == 0:
                        vector = torch.cat((vector,torch.tensor([1])),0)
                    else:
                        vector = torch.cat((vector,torch.tensor([0])),0)
                #Adding graph structure features
                vector = torch.cat((vector,torch.tensor([degree_centrality[node]])),0)
                vector = torch.cat((vector,torch.tensor([in_degree_centrality[node]])),0)
                vector = torch.cat((vector,torch.tensor([out_degree_centrality[node]])),0)
                vector = torch.cat((vector,torch.tensor([katz_centrality[node]])),0)
                vector = torch.cat((vector,torch.tensor([closeness_centrality[node]])),0)
                vector = torch.cat((vector,torch.tensor([betweenness_centrality[node]])),0)
                vector = torch.cat((vector,torch.tensor([harmonic_centrality[node]])),0)
                #vector = torch.cat((vector,torch.tensor([trophic_levels[node]])),0)    #Existing Error
                vector = torch.cat((vector,torch.tensor([clustering[node]])),0)
                vector = torch.cat((vector,torch.tensor([square_clustering[node]])),0)
                vector = torch.cat((vector,torch.tensor([pagerank[node]])),0)
                if x == None:
                    x = vector
                    x = torch.tensor([vector.tolist()])
                else:
                    x = torch.vstack((x,vector))
        x = x.to(torch.float32)
        data = Data(x=x, edge_index=edge_index, y=y)
        return data

    def _writeData(self, file, label, out_dir, CGType='Xref'):
        out_dir = self._datafile_path
        path = file
        path = path.replace(self._data_path, '').replace('../', '')
        path = path.replace('/', '_')
        path = path[:-4]
        write_path = out_dir + path + ".pkl"
        if os.path.isfile(write_path):
            return
        try:
            scg = scg = self._analyzer.AnalyzeAPK(file, CGType)
            data = self._data_generate(scg,label)
            with open(write_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(Fore.RED+f'ERROR: {str(e)}')
            print(Style.RESET_ALL)

    def _filterAPKList (self, apk_list, out_dir):
        new_list = []
        for idx in range(len(apk_list)):
            path = apk_list[idx]
            path = path.replace(self._data_path, '').replace('../', '')
            path = path.replace('/', '_')
            path = path[:-4]
            write_path = out_dir + path + ".pkl"
            if os.path.isfile(write_path): continue
            new_list.append(apk_list[idx])
        return new_list

    def Get_APKlist(self,overwrite=False):
        files = glob.glob(self._data_path+'/**/*.*', recursive=True)
        n_files = len(files)
        print("Found", n_files, " in directory")
        if not overwrite:
            files = self._filterAPKList(files, self._datafile_path)
            n_files = len(files)
            print("Remaining", n_files)
        return files  

    def Get_APKlist_label(self,apk_list,label_list):
        files = glob.glob(self._data_path+'/**/*.*', recursive=True)
        n_files = len(files)
        print("Found", n_files, " in directory")
        new_list = []
        new_label = []
        for idx in range(len(apk_list)):
            path = apk_list[idx]
            path = path.replace(self._data_path, '').replace('../', '')
            path = path.replace('/', '_')
            path = path[:-4]
            write_path = self._datafile_path + path + ".pkl"
            if os.path.isfile(write_path): continue
            new_list.append(apk_list[idx])
            new_label.append(label_list[idx])
        n_files = len(new_list)
        print("Remaining", n_files)
        return new_list, new_label

    def ParseAPK(self, APKlist, Labellist, CGType='Xref', pcount=20):
        datafile_path = self._working_path + self._datasetstr + '/'
        if not os.path.exists(datafile_path):
            os.makedirs(datafile_path)
        
        t1 = time.time()
        n_files = len(APKlist)
        n = 0
        if pcount == 1:
            while n < n_files:
                self._writeData(APKlist[n], Labellist[n].item(), datafile_path, CGType)
                n += 1
        else: 
            mpp = mp.Pool(pcount)
            while n < n_files:
                mpp.apply_async(MultiParseAPK, args=(self._datasetstr,self._basic_wp, self._data_path, APKlist[n], Labellist[n].item(), datafile_path, CGType), error_callback=on_task_error)
                n+=1
            mpp.close()
            mpp.join()         

        t2 = time.time()
        
        if n_files != 0:
            print("Finished in: ", t2-t1, ", avg time: ", (t2-t1)/(n_files), "s/apk" )  

    def Dataset_generate(self, filelist):
        dataset = []
        
        print('\nGenerate dataset')
        print('=======================================================')
        for file in filelist:
            try:
                with open(file,'rb') as f:
                    data = pickle.load(f)
                dataset.append(data)
            except Exception as e:
                print(str(e))
                pass
        print(f'DataSet Length: {len(dataset):05d}')
        print('Generate done')
        print('=======================================================')
        with open(self.dataset_path,'wb') as f:
            pickle.dump(dataset,f)
        print('Save Dataset Done!')
        return dataset
    
    def _train(self, model, optimizer, criterion, train_loader, device):
        model = model.to(device)
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.batch = data.batch.to(device)
            data.y = data.y.to(device)
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def _test(self, model, criterion, test_loader, device):
        model = model.to(device)
        model.eval()
        Y = []
        y_pred = []
        loss_sum = 0
        with torch.no_grad():
            for data in test_loader:  # Iterate in batches over the training/test dataset.
                data.x = data.x.to(device)
                data.edge_index = data.edge_index.to(device)
                data.batch = data.batch.to(device)
                data.y = data.y.to(device)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)  # Compute the loss.
                loss_sum += loss
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                Y.extend(data.y.cpu())
                y_pred.extend(pred.cpu())
        accuracy = metrics.accuracy_score(Y,y_pred)
        conf = metrics.confusion_matrix(Y,y_pred)
        if len(conf) == 2:
            average = 'binary'
        else:
            average = 'weighted'
        f1 = metrics.f1_score(Y, y_pred,average=average)
        recall = metrics.recall_score(Y,y_pred,average=average)
        precision = metrics.precision_score(Y,y_pred,average=average)
        return loss_sum/len(test_loader), accuracy, recall, precision, f1  # Derive ratio of correct predictions.
    
    def _predict(self, model, file_path, device, CGType="Xref"):
        SCG = self._analyzer.AnalyzeAPK(file_path, CGType)
        data = self._data_generate(SCG)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        model = model.to(device)
        model.eval()
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        out = model(data.x, data.edge_index, data.batch)
        Y = out.argmax(dim=1)  # Use the class with highest probability.
        return SCG, Y.item()
    
    def data2predict(self,model,data,device):
        model = model.to(device)
        model.eval()
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        out = model(data.x, data.edge_index, data.batch)
        Y = out.argmax(dim=1)  # Use the class with highest probability.
        return Y.item()
    
    def _get_data_exp(self, file_path, CGType="Xref"):
        SCG = self._analyzer.AnalyzeAPK(file_path, CGType)
        if CGType == "All":
            return SCG, None
        data = self._data_generate(SCG)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        return SCG, data
    
    def _label2class(self,label,num_classes):
        if num_classes == 2:
            label_map = {0:"benign",1:"malicious"}
        elif num_classes == 5:
            label_map = {0:"benign",1:"adware",2:"banking",3:"riskware",4:"sms"}
        return label_map[label]

    def _show_acc(self,train,test,p_id):
        plt.figure()
        epoch = range(len(train))
        plt.plot(epoch,train,'b',label='Train Acc')
        plt.plot(epoch,test,'r',label='Test Acc')
        plt.legend()
        plt.savefig(fname=self._relatedfile_path + self.runid + self._datasetstr + f'{p_id}_figure.pdf',dpi=256)
    
    def _save_model(self,model,p_id):
        torch.save(model,self._relatedfile_path + self.runid + self._datasetstr + f'{p_id}_model.pt')
    
    def _load_model(self,p_id):
        load_model = torch.load(self._relatedfile_path + self.runid + self._datasetstr + f'{p_id}_model.pt')
        return load_model
    
    def _filelist_datalist(self,filelist):
        datalist = []
        for file in filelist:
            try:
                with open(file,'rb') as f:
                    data = pickle.load(f)
                datalist.append(data)
            except Exception as e:
                print(str(e))
                pass
        return datalist
      
    def Model_train_test(self,datasetid,train_dataset,test_dataset,model='ModelAP',epochs=300,hidden_dim=128,lr=0.0001,random_seed=319,ratio=0.9,num_classes=2,time_dataset=[]):
        device = self.device
        print(f'Device: {device}')
        torch.cuda.manual_seed(random_seed)
        torch.manual_seed(random_seed)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print('Dataset')
        print('=============================================================')
        print(f'Total Train APKs: {len(train_dataset):05d}')
        print(f'Total Test APKs: {len(test_dataset):05d}')
        
        print('GNN Model')
        print('=============================================================')
        num_features = train_dataset[0].num_features
        if model == 'ModelAP':
            m = ModelAP(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'ModelBP':
            m = ModelBP(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'ModelA':
            m = ModelA(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'ModelB':
            m = ModelB(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'GCN':
            m = GCN(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'GCNv2':
            m = GCNv2(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'GAT':
            m = GAT(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'GATv2':
            m = GATv2(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'ModelAPP':
            m = ModelAPP(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'ModelBPP':
            m = ModelBPP(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'GIN':
            m = GIN(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'GraphSAGE':
            m = GraphSAGE(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model == 'MGNN':
            m = MGNN(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        else:
            print('Model not exists')
            return
        print(m)
        optimizer = torch.optim.Adam(m.parameters(), lr = lr, weight_decay = 5e-4)
        criterion = torch.nn.CrossEntropyLoss()
        p_id = f'{model}_{hidden_dim}_{lr}_{ratio}_{self.feature}'
        print('Train & Test Process')
        best = [0,0,0,0]
        #[acc,recall,precision,f1]
        trainlist = []
        testlist = []
        print('=============================================================')
        train_loss, train_acc, train_recall, train_precision, train_f1 = self._test(m, criterion, train_loader, device)
        test_loss, test_acc, test_recall, test_precision, test_f1 = self._test(m, criterion, test_loader, device)
        if test_acc>best[0]:
            best = [test_acc,test_recall,test_precision,test_f1,train_acc,train_recall,train_precision,train_f1]
            self._save_model(m,p_id)
        trainlist.append(train_acc)
        testlist.append(test_acc)
        print(f'Epoch: {0:03d} || Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Train Precision:{train_precision:.4f}, Train Recall:{train_recall:.4f}, Train F1-score:{train_f1:.4f}')
        print(f'              Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}, Test Precision:{test_precision:.4f}, Test Recall:{test_recall:.4f}, Test F1-score:{test_f1:.4f}')
        for epoch in range(1, epochs):
            self._train(m, optimizer, criterion, train_loader, device)
            train_loss, train_acc, train_recall, train_precision, train_f1 = self._test(m, criterion, train_loader, device)
            test_loss, test_acc, test_recall, test_precision, test_f1 = self._test(m, criterion, test_loader, device)
            if test_acc>best[0]:
                best = [test_acc,test_recall,test_precision,test_f1,train_acc,train_recall,train_precision,train_f1]
                self._save_model(m,p_id)
                self._save_model_dict(m,p_id)
            trainlist.append(train_acc)
            testlist.append(test_acc)
            print(f'Epoch: {epoch:03d} || Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Train Precision:{train_precision:.4f}, Train Recall:{train_recall:.4f}, Train F1-score:{train_f1:.4f}')
            print(f'              Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}, Test Precision:{test_precision:.4f}, Test Recall:{test_recall:.4f}, Test F1-score:{test_f1:.4f}')
        print('=============================================================')
        print(f'Best Performance:')
        print(f'\tTest\tAccuracy={best[0]:.4f} Recall={best[1]:.4f} Precision={best[2]:.4f} F1-score={best[3]:.4f}')
        print(f'\tTrain\tAccuracy={best[4]:.4f} Recall={best[5]:.4f} Precision={best[6]:.4f} F1-score={best[7]:.4f}')
        self._show_acc(trainlist,testlist,p_id)
        
        if 'AZ' in datasetid:
            print('=============================================================')
            m = self._load_model(p_id)
            time = 2019
            for datalist in time_dataset:
                data_loader = self._filelist_datalist(datalist)
                data_loader = ablation(self.feature,data_loader)              
                time_loader = DataLoader(data_loader, batch_size=32, shuffle=False)
                print(f'{time} APKs: {len(data_loader)}')
                test_loss, test_acc, test_recall, test_precision, test_f1 = self._test(m, criterion, time_loader, device)
                print(f'\tTest Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}, Test Precision:{test_precision:.4f}, Test Recall:{test_recall:.4f}, Test F1-score:{test_f1:.4f}')
                time += 1
                print('=============================================================')
        return m
    
    def _get_init_model(self,model_str,num_features,hidden_dim,num_classes):
        if model_str == 'ModelAP':
            m = ModelAP(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'ModelBP':
            m = ModelBP(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'ModelA':
            m = ModelA(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'ModelB':
            m = ModelB(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'GCN':
            m = GCN(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'GCNv2':
            m = GCNv2(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'GAT':
            m = GAT(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'GATv2':
            m = GATv2(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'ModelAPP':
            m = ModelAPP(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'ModelBPP':
            m = ModelBPP(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'GIN':
            m = GIN(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'GraphSAGE':
            m = GraphSAGE(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        elif model_str == 'MGNN':
            m = MGNN(hidden_channels = hidden_dim, num_nodes_features = num_features,num_classes=num_classes)
        
        return m

    def _save_model_dict(self,model,p_id):
        torch.save(model.state_dict(),self._relatedfile_path + self.runid + self._datasetstr + f'{p_id}_state.pt')

    def _load_model_dict(self,model,p_id):
        load_dict = torch.load(self._relatedfile_path + self.runid + self._datasetstr + f'{p_id}_state.pt',map_location=self.device)
        model.load_state_dict(load_dict)
        return model

    def get_explainer(self, model, exp_algorithm):
        explainer = Explainer(
            model=model,
            algorithm=exp_algorithm,
            explanation_type='phenomenon',
            #node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='binary_classification',
                task_level='graph',
                return_type='raw',
            ),
        )
        return explainer
    
    def explain(self, model, global_g, data_exp, ExpAlgorithm="GNNExplainer"):
        if ExpAlgorithm == "PGExplainer":
            device = torch.device("cpu")
        else:
            device = self.device
        model.to(device)
        if ExpAlgorithm == "GNNExplainer" or ExpAlgorithm == "BaseGNNExplainer":
            exp_algorithm = GNNExplainer(epochs=1000,lr=0.001)
        elif ExpAlgorithm == "PGExplainer":
            exp_algorithm = PGExplainer(epochs=30)
        elif ExpAlgorithm == "GraphMaskExplainer":
            #print("--- 诊断模型结构 ---")
            mp_layer_count = 0
            for i, module in enumerate(model.modules()):
                # 检查模块是否是 MessagePassing 的实例
                if isinstance(module, MessagePassing):
                    mp_layer_count += 1
                    #print(f"找到 MessagePassing 层 #{mp_layer_count}: {module}")

            #print(f"\n模型中总共找到了 {mp_layer_count} 个 MessagePassing 层。")
            #print("请确保 GraphMaskExplainer 中的 'num_layers' 参数设置为这个值。")
            #print("------------------------\n")
            exp_algorithm = GraphMaskExplainer(num_layers=mp_layer_count)
        explainer = self.get_explainer(model, exp_algorithm)
        data_exp.to(device)
        prection = explainer.get_prediction(x=data_exp.x, edge_index=data_exp.edge_index, batch = data_exp.batch)
        target = explainer.get_target(prection)
        Y = self.data2predict(model,data_exp,device)
        print(f'Predict: {prection} {target} {Y}')
        if ExpAlgorithm == "PGExplainer":
            epochs = 30
            for epoch in range(epochs):
                explainer.algorithm.train(epoch, model, x=data_exp.x, edge_index=data_exp.edge_index, target=target, batch = data_exp.batch)
        explanation = explainer(x=data_exp.x, edge_index=data_exp.edge_index, target=target, batch = data_exp.batch)
        return explanation
    
    def exp_subgraph(self,scg,exp):
        max_nodes = 50
        topK_edges = 10
        #node_mask = exp.get('node_mask').cpu()
        edge_mask = exp.get('edge_mask').cpu()

        #no_zero_mask = edge_mask[edge_mask>0]

        mean_value = torch.mean(edge_mask)
        std_value = torch.std(edge_mask)
        edge_threshold = mean_value + std_value

        print(f'Mean value calculate: {mean_value}')    
        print(f'Std value calculate: {std_value}')
        print(f'Edge threshold calculate: {edge_threshold}')
        #print(f'Edge mask: {edge_mask}')

        for i in range(2):
            sub_g = nx.DiGraph()
            '''
            for node, mask in zip(scg.nodes(), node_mask):
                if mask.item() > node_threshold:
                    sub_g.add_node(node)
            '''
            for edge, mask in zip(scg.edges(), edge_mask):
                if mask.item() >= edge_threshold:
                    sub_g.add_edge(edge[0], edge[1])
            if len(sub_g.nodes()) == 0:
                edge_threshold = mean_value
            else:
                break
        
        if len(sub_g.nodes()) > max_nodes:
            sorted_lst = sorted(edge_mask, reverse=True)
            edge_threshold = sorted_lst[topK_edges-1] if topK_edges < len(edge_mask) else sorted_lst[-1]
            sub_g = nx.DiGraph()
            '''
            for node, mask in zip(scg.nodes(), node_mask):
                if mask.item() > node_threshold:
                    sub_g.add_node(node)
            '''
            for edge, mask in zip(scg.edges(), edge_mask):
                if mask.item() >= edge_threshold:
                    sub_g.add_edge(edge[0], edge[1])

        plt.figure()
        node_size = 400
        nx.draw(sub_g,node_size=node_size)
        plt.savefig(self._relatedfile_path + "temp_subgraph.png", dpi=640)
        plt.show()
        return sub_g
    
    def exp_subgraph_hard(self,scg,exp,hard_value):
        max_nodes = 50
        topK_edges = 10
        #node_mask = exp.get('node_mask').cpu()
        edge_mask = exp.get('edge_mask').cpu()

        edge_threshold = hard_value

        sub_g = nx.DiGraph()
        '''
        for node, mask in zip(scg.nodes(), node_mask):
            if mask.item() > node_threshold:
                sub_g.add_node(node)
        '''
        for edge, mask in zip(scg.edges(), edge_mask):
            if mask.item() >= edge_threshold:
                sub_g.add_edge(edge[0], edge[1])

        plt.figure()
        node_size = 400
        nx.draw(sub_g,node_size=node_size)
        plt.savefig(self._relatedfile_path + "temp_subgraph.png", dpi=640)
        plt.show()
        return sub_g
    
    def clean_xml(self,code):
        # Remove all NULL bytes and control characters except common whitespace
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', code)   

    def gml_generate(self,scg,path="explanation_subgraph.graphml"):
        filepath = self._relatedgraph_path + path
        sub_g = nx.DiGraph()
        for node in scg.nodes():
            code = node.get_method().get_source() if not node.is_external() else "External Method"
            code = self.clean_xml(code)
            sub_g.add_node(node, source_code=code)
        for edge in scg.edges():
            sub_g.add_edge(edge[0],edge[1])
        nx.write_graphml(sub_g,filepath)

    def dot_generate(self,g,path="explanation_subgraph.dot"):
        filepath = self._relatedgraph_path + path
        dot_g = nx.DiGraph()
        for node in g.nodes():
            dot_g.add_node(node2id(node))
        for edge in g.edges():
            dot_g.add_edge(node2id(edge[0]),node2id(edge[1]))
        write_dot(dot_g,filepath)

    def gml_generate2path(self,scg,path="explanation_subgraph.graphml"):
        filepath = path
        sub_g = nx.DiGraph()
        for node in scg.nodes():
            code = node.get_method().get_source() if not node.is_external() else "External Method"
            code = self.clean_xml(code)
            sub_g.add_node(node, source_code=code)
        for edge in scg.edges():
            sub_g.add_edge(edge[0],edge[1])
        nx.write_graphml(sub_g,filepath)

    def dot_generate2path(self,g,path="explanation_subgraph.dot"):
        filepath = path
        dot_g = nx.DiGraph()
        for node in g.nodes():
            dot_g.add_node(node2id(node))
        for edge in g.edges():
            dot_g.add_edge(node2id(edge[0]),node2id(edge[1]))
        write_dot(dot_g,filepath)

                
    
def cut_doc(text):
    stop_list = ["'",',']
    text_cut = jieba.cut(text)
    text_split = ' '.join(text_cut).split()
    final_doc = [word for word in text_split if word not in stop_list]
    return final_doc

def read_corups(corups, tokens_only = False):
        for i, line in enumerate(corups):
            tokens = cut_doc(line)
            if tokens_only:
                yield tokens
            else:
                yield TaggedDocument(tokens,[i])

def Docmodel(fname, corups, vector_size = 100, min_count = 2, epochs = 10, workers = 1):
    train_corups = list(read_corups(corups))
    print('\nTrain Doc2Vec model')
    print('=======================================================')
    model = Doc2Vec(train_corups, vector_size = vector_size, seed = 0, min_count = min_count, epochs = epochs, workers = workers)
    model.save(fname)
    print('\nTrain Doc2Vec model done')
    print('=======================================================')

def MultiParseAPK(dataset,working_path,datapath,APKfile,label,save_path,cgtype):
    ea = ExpAndro(data_path=datapath,dataset=dataset,working_path=working_path)
    ea._writeData(APKfile, label, save_path, cgtype)

def on_task_error(exc):
    exc_traceback = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(f"Traceback Info:\n{exc_traceback}")

def node2id(node):
    class_name = node.class_name
    method_name = node.name
    return class_name + '\n' + method_name

def ablation(feature,dataset):
    for idx, data in enumerate(dataset):
        #print(idx,type(data))
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