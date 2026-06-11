import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader as TensorDataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GraphConv, SAGPooling

from tqdm import tqdm

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast
import torch_geometric.transforms as T
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
from sklearn.model_selection import train_test_split
from GCL.augmentors.augmentor import Graph
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 自定义语义增强器
class AugmentorA(A.Augmentor):
    def __init__(self, pf: float, num_seeds: int, walk_length: int):
        super(AugmentorA, self).__init__()
        self.augmentor = A.FeatureMasking(pf=pf)

    def augment(self, g: Graph) -> Graph:
        # 1. 解包，得到三个位置组件
        x, edge_index, edge_attr = g.unfold()

        # 2. 将解包后的组件传递给内部的 augmentor
        x_aug, edge_index_aug, edge_attr_aug = self.augmentor(x, edge_index, edge_attr)

        # 3. 使用位置参数 (Positional Arguments) 重新创建 Graph 对象
        return Graph(x_aug, edge_index_aug, edge_attr_aug)
    
class AugmentorB(A.Augmentor):
    def __init__(self, pf: float, num_seeds: int, walk_length: int):
        super(AugmentorB, self).__init__()
        self.augmentor = A.RWSampling(num_seeds=num_seeds, walk_length=walk_length)

    def augment(self, g: Graph) -> Graph:
        # 1. 解包，得到三个位置组件
        x, edge_index, edge_attr = g.unfold()

        # 2. 将解包后的组件传递给内部的 augmentor
        x_aug, edge_index_aug, edge_attr_aug = self.augmentor(x, edge_index, edge_attr)

        # 3. 使用位置参数 (Positional Arguments) 重新创建 Graph 对象
        return Graph(x_aug, edge_index_aug, edge_attr_aug)

# 2. GNN 编码器
class ModelAP(torch.nn.Module):
    def  __init__(self,hidden_channels, num_nodes_features, dropout = 0.5, keep_ratio = 0.5):
        super(ModelAP, self).__init__()
        self.dropout = dropout
        self.keep_ratio = keep_ratio
        self.gcn1 = GraphConv(num_nodes_features, hidden_channels)
        self.gcn2 = GraphConv(hidden_channels, hidden_channels)
        self.gcn3 = GraphConv(hidden_channels, hidden_channels)
        self.sagp = SAGPooling(hidden_channels*3, self.keep_ratio, multiplier=8)

    def forward(self, x, edge_index, batch):
        gcn1 = F.relu(self.gcn1(x, edge_index))
        gcn2 = F.relu(self.gcn2(gcn1, edge_index))
        gcn3 = F.relu(self.gcn3(gcn2, edge_index))

        pre_gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim = 1)
        px, __, __, pbatch, __, __ = self.sagp(pre_gcn_feature, edge_index, batch = batch)
        readout = torch.cat((global_mean_pool(px, pbatch),global_max_pool(px, pbatch)),dim = 1) 
        return px, readout
    
# 3. 下游任务分类器
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# --- 修改后的训练和评估函数 ---
def pretrain_epoch(encoder_model, contrast_model, dataloader, optimizer, augmentor1, augmentor2): # 添加 augmentor 参数
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()

        # ---- 这是修改的关键部分 ----
        # 从 data 对象中直接获取 x 和 edge_index
        x, edge_index = data.x, data.edge_index

        # 解包传递给 augmentor
        x1, edge_index1, _ = augmentor1(x, edge_index)
        x2, edge_index2, _ = augmentor2(x, edge_index)
        # ---------------------------
        
        # Get embeddings
        # 注意：这里需要传递 batch 信息，它在两次增强中保持不变
        _, g1_emb = encoder_model(x1, edge_index1, data.batch)
        _, g2_emb = encoder_model(x2, edge_index2, data.batch)
        
        loss = contrast_model(g1=g1_emb, g2=g2_emb)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def extract_embeddings(encoder_model, dataset, batch_size):
    """使用冻结的编码器提取所有图的嵌入"""
    encoder_model.eval()
    embeddings = []
    labels = []
    loader = GraphDataLoader(dataset, batch_size=batch_size) # Use a loader for batching
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _, g_emb = encoder_model(data.x, data.edge_index, data.batch)
            embeddings.append(g_emb.cpu())
            labels.append(data.y.cpu())
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)

def train_classifier(classifier, train_loader, optimizer, criterion):
    classifier.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = classifier(x_batch)
        loss = criterion(output, y_batch.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_classifier(classifier, test_loader):
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = classifier(x_batch)
            _, predicted = torch.max(output.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    # --- Hyperparameters ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = '.'
    dataset_name = 'PROTEINS'
    
    # Model params
    input_dim, hidden_dim = 1, 256
    # Pre-training params
    batch_size, pretrain_epochs, pretrain_lr = hidden_dim, 500, 0.001
    # Classifier params
    classifier_epochs, classifier_lr, classifier_hidden_dim = 200, 0.0001, hidden_dim*6
    
    # Augmentation params
    pf, num_seeds, walk_length = 0.3, 10, 20

    # --- Setup ---
    dataset = TUDataset(root=path, name=dataset_name, transform=T.Constant(1.0))
    input_dim = max(dataset.num_features, 1)
    num_classes = dataset.num_classes

    # --- PHASE 1: SELF-SUPERVISED PRE-TRAINING ---
    print("="*20 + " Phase 1: Pre-training Encoder " + "="*20)
    pretrain_loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=True)
    augmentor1 = AugmentorA(pf=pf, num_seeds=num_seeds, walk_length=walk_length)
    augmentor2 = AugmentorB(pf=pf, num_seeds=num_seeds, walk_length=walk_length)
    
    encoder = ModelAP(hidden_dim, input_dim).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    optimizer = Adam(encoder.parameters(), lr=pretrain_lr)

    with tqdm(total=pretrain_epochs, desc='(Pre-training)') as pbar:
        for epoch in range(1, pretrain_epochs + 1):
            loss = pretrain_epoch(encoder, contrast_model, pretrain_loader, optimizer, augmentor1, augmentor2)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    # --- PHASE 2: DOWNSTREAM CLASSIFICATION ---
    print("\n" + "="*20 + " Phase 2: Training Downstream Classifier " + "="*20)
    
    # 1. Freeze the encoder and extract embeddings
    print("Step 1: Freezing encoder and extracting embeddings...")
    for param in encoder.parameters():
        param.requires_grad = False
    
    all_embeddings, all_labels = extract_embeddings(encoder, dataset, batch_size)
    
    # 2. Split data for classifier training
    X_train, X_test, y_train, y_test = train_test_split(
        all_embeddings, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = TensorDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = TensorDataLoader(test_dataset, batch_size=batch_size)
    
    # 3. Initialize and train the MLP classifier
    print("Step 2: Training the MLP classifier...")
    classifier = MLPClassifier(
        input_dim=classifier_hidden_dim, 
        hidden_dim=hidden_dim, 
        output_dim=num_classes
    ).to(device)
    
    classifier_optimizer = Adam(classifier.parameters(), lr=classifier_lr)
    criterion = nn.CrossEntropyLoss()

    best_score = 0
    for epoch in range(1, classifier_epochs + 1):
        loss = train_classifier(classifier, train_loader, classifier_optimizer, criterion)
        print(f'Epoch{epoch}\t loss: {loss}')
        test_accuracy = test_classifier(classifier, test_loader)
        print(f'\t Acc:{test_accuracy}')
        print(f'\t Best:{best_score}')
        if test_accuracy > best_score:
            best_score = test_accuracy

    print(f"\nBest Test Accuracy of MLP Classifier: {best_score:.2f}%")

    # 4. Evaluate the classifier
    print("Step 3: Evaluating the classifier...")
    test_accuracy = test_classifier(classifier, test_loader)
    print(f"\nFinal Test Accuracy of MLP Classifier: {test_accuracy:.2f}%")