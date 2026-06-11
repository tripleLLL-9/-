import pickle
import networkx as nx
import queue as q
from androguard.misc import AnalyzeAPK, AnalyzeDex
from androguard.decompiler.dad import decompile

class Analyzer():
    def __init__(self, working_path):
        #Load sensitive api list from pkl file
        sapi_path = working_path + "new_sensitive_api_list_data.pkl"
        with open(sapi_path,'rb') as f:
            sapi=pickle.load(f)
        self.sapi = sapi    #Sensitive API list
        self.corups = []    #AST corups

    def get_ast_method(self, m):
        z = decompile.DvMethod(m)
        z.process(doAST=True)
        return z.get_ast()

    def api_getname(self, api):
        classname,dot,apiname=api.rpartition('.')
        classname='L'+classname.replace('.','/')
        apiname='^'+apiname+'$'
        return classname+';',apiname

    def SourceCompleteCG(self, als,classname,methodname):
        source=als.find_methods(classname,methodname)
        sourcenode=list(source)[0]
        #print("Source_Method_full_name: ",sourcenode.full_name)
        tempgraph=als.get_call_graph(sourcenode.get_class_name(),sourcenode.name) 
        nodesq=q.Queue()
        for i in tempgraph.nodes():
            nodesq.put(i)
        nodeslist=[]
        nodeslist.append(sourcenode.full_name)
        while nodesq.empty()== 0:
            node=nodesq.get()
            if node.full_name not in nodeslist:
               #print("正在分析节点",node.full_name)
               temp=als.get_call_graph(node.get_class_name(),node.name)
               tempgraph.add_nodes_from(temp.nodes)
               tempgraph.add_edges_from(temp.edges)
               nodeslist.append(node.full_name)
               for i in temp.nodes():
                   if (i.full_name not in nodeslist):
                       nodesq.put(i)
        return tempgraph

    def SourceInternalCG(self, als,classname,methodname):
        source=als.find_methods(classname,methodname)
        sourcenode=list(source)[0]
        #print("Source_Method_full_name: ",sourcenode.full_name)
        tempgraph=als.get_call_graph(sourcenode.get_class_name(),sourcenode.name) 
        nodesq=q.Queue()
        for i in tempgraph.nodes():
            nodesq.put(i)
        nodeslist=[]
        nodeslist.append(sourcenode.full_name)
        while nodesq.empty()== 0:
            node=nodesq.get()
            if node.full_name not in nodeslist:
               #print("正在分析节点",node.full_name)
               temp=als.get_call_graph(node.get_class_name(),node.name)
               tempgraph.add_nodes_from(temp.nodes)
               tempgraph.add_edges_from(temp.edges)
               nodeslist.append(node.full_name)
               for i in temp.nodes():
                   if (i.full_name not in nodeslist):
                       nodesq.put(i)
        checklist=list(tempgraph.nodes())
        for i in checklist:
            if i.is_external()==True and i!=sourcenode:
                tempgraph.remove_node(i)
        return tempgraph

    def SourceXrefCG(self, als,classname,methodname):
        source=als.find_methods(classname,methodname)
        sourcenode=list(source)[0]
        #print("Source_Method_full_name: ",sourcenode.full_name)
        tempgraph=als.get_call_graph(sourcenode.get_class_name(),sourcenode.name) 
        nodesq=q.Queue()
        for i in tempgraph.nodes():
            nodesq.put(i)
        nodeslist=[]
        nodeslist.append(sourcenode.full_name)
        while nodesq.empty()== 0:
            node=nodesq.get()
            if node.full_name not in nodeslist:
               #print("正在分析节点",node.full_name)
               xrefnodes=node.get_xref_from()
               for i in xrefnodes:
                   ma=i[1]
                   mnode=als.get_method(ma)
                   tempgraph.add_edge(ma,node)
                   if (ma.full_name not in nodeslist):
                       nodesq.put(ma)
               nodeslist.append(node.full_name) 
        return tempgraph

    def AnalyzeAPK(self, filepath, graphtype = 'Complete'):
        '''
            graphtype:
                'All': whole CG
                'Complete': SourceCompleteCG; 
                'Internal': SourceInternalCG; 
                'Xref'    : SourceXrefCG; 
        '''
        filename = (filepath.rpartition('/'))[2]
        print('Start analyzing the APK file: ',filename)
        apk, dvm, als = AnalyzeAPK(filepath)

        if graphtype == "All":
            CG = als.get_call_graph()
            return CG

        SCG = nx.MultiDiGraph()
        sapicount = 0
        for api in self.sapi:
            tempclass,tempapi=self.api_getname(api)
            #print(tempclass,tempapi)
            mx=als.find_methods(tempclass,tempapi)
            xx=list(mx)
            if(len(xx)!= 0):
                sapicount += 1
                if graphtype == 'Xref':
                    tempcg = self.SourceXrefCG(als,tempclass,tempapi)
                elif graphtype == 'Internal':
                    tempcg = self.SourceInternalCG(als,tempclass,tempapi)
                else:   #graphtype == 'Complete'
                    graphtype = 'Complete'
                    tempcg = self.SourceCompleteCG(als,tempclass,tempapi)
                SCG.add_nodes_from(tempcg.nodes())
                SCG.add_edges_from(tempcg.edges())
        if sapicount == 0:
            SCG = als.get_call_graph()
        print(f'Sensitive API Counts: {sapicount:03d}')
        print(f'Sensitive Call Graph Nodes Total Number: {SCG.number_of_nodes():05d}')
        #nx.write_gexf(SCG,os.getcwd()+'/APKSCG/'+filename+graphtype+'SCG.gexf')
        print('Analysis completed')
        print('*******************************************************')

        return SCG

    def Feature_collection(self, SCG):
        for node in SCG.nodes():
            if node.is_external() == False:
                ast = self.get_ast_method(node)
                list = self.corups
                list.append(str(ast))



