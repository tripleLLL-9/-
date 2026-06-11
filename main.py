import os
import torch
import glob
import pickle
import numpy as np
import time
import multiprocessing as mp
import jieba
from sklearn import metrics
import matplotlib.pyplot as plt
from colorama import init, Fore, Style
import traceback
import warnings
import networkx as nx
import random
from collections import defaultdict
import shutil
import hashlib
from functools import partial
from torch.utils.data import Subset
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
try:
    from obfuscapk import main as ob
except ImportError:
    print(Fore.YELLOW + "[警告] obfuscapk 未安装" + Style.RESET_ALL)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn import GATConv, global_mean_pool, GraphConv, SAGPooling, global_max_pool
from torch_geometric.utils import subgraph, dropout_adj
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from AnalyzeModule import Analyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

import seaborn as sns
from sklearn.manifold import TSNE
import umap
@torch.no_grad()
def evaluate_view_invariance(encoder, pair_loader, device, num_pairs=500):
    """评估同一APK不同视图之间的相似度（正样本对）以及不同APK之间的相似度（负样本对）"""
    encoder.eval()
    pos_sims = []
    neg_sims = []
    
    for batch1, batch2, _ in pair_loader:
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)
        z1, _ = encoder(batch1.x, batch1.edge_index, batch1.batch)
        z2, _ = encoder(batch2.x, batch2.edge_index, batch2.batch)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 正样本对相似度（对角线）
        pos_sim = (z1 * z2).sum(dim=1)  # [batch_size]
        pos_sims.append(pos_sim)
        
        # 负样本对：随机打乱 batch 内配对（或计算跨 batch 的负样本）
        # 简化：使用同一 batch 内不同索引的配对
        batch_size = z1.size(0)
        indices = torch.randperm(batch_size, device=device)
        z2_shuffled = z2[indices]
        neg_sim = (z1 * z2_shuffled).sum(dim=1)
        neg_sims.append(neg_sim)
        
        if len(pos_sims) * batch_size >= num_pairs:
            break
    
    pos_sim_mean = torch.cat(pos_sims).mean().item()
    neg_sim_mean = torch.cat(neg_sims).mean().item()
    return pos_sim_mean, neg_sim_mean

def make_pair_filter(mode: str, allowed_strategies: list = None):
    """
    生成一个视图对过滤器
    
    Args:
        mode: 过滤模式
            - 'all' : 所有可能的视图对（默认行为）
            - 'original_only' : 只包含原始图与变种图的对
            - 'variant_only' : 只包含不同变种之间的对（两个都是变种）
            - 'specific' : 使用 allowed_strategies 列表指定允许的策略组合
        allowed_strategies: 当 mode='specific' 时，格式为 [(strategy1, strategy2), ...]
            例如 [('Reflection', 'ConstStringEncryption'), (None, 'Reflection')] 
            None 表示任意策略，第一个元素为 None 表示原始图。
    """
    if mode == 'all':
        return lambda v1, v2: True
    elif mode == 'original_only':
        return lambda v1, v2: (v1[0] == 'original' and v2[0] == 'variant')
    elif mode == 'variant_only':
        return lambda v1, v2: v1[0] == 'variant' and v2[0] == 'variant'
    elif mode == 'specific':
        if not allowed_strategies:
            raise ValueError("mode='specific' 必须提供 allowed_strategies")
        # 将每个规则转换为更易匹配的形式
        rules = []
        for rule in allowed_strategies:
            if len(rule) == 2:
                s1, s2 = rule
                rules.append((s1, s2))
            else:
                raise ValueError("每条规则必须是 (strategy1, strategy2) 格式")
        
        def filter_func(v1, v2):
            t1, s1, _ = v1
            t2, s2, _ = v2
            if t1 == 'original': s1 = None
            if t2 == 'original': s2 = None
            for rs1, rs2 in rules:
                if (s1 == rs1 and s2 == rs2) or (s1 == rs2 and s2 == rs1):
                    return True
            return False
        return filter_func
    else:
        raise ValueError(f"未知模式: {mode}")
    
    
def load_paths_from_pkl(pkl_path):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    return None

def scan_original_paths(graph_root):
    """扫描 original/ 目录"""
    paths = []
    original_dir = os.path.join(graph_root, "original")
    if not os.path.exists(original_dir): return []
    for cat in ["benign", "malicious"]:
        cat_dir = os.path.join(original_dir, cat)
        if os.path.exists(cat_dir):
            paths.extend(glob.glob(os.path.join(cat_dir, "*.pt")))
    return paths

def scan_variant_paths(graph_root):
    """扫描 variants/ 目录，返回 dict {strategy: [paths]}"""
    variant_dict = {}
    variants_dir = os.path.join(graph_root, "variants")
    if not os.path.exists(variants_dir): return variant_dict
    
    for strategy in os.listdir(variants_dir):
        strat_dir = os.path.join(variants_dir, strategy)
        if not os.path.isdir(strat_dir) or strategy.startswith('.'):
            continue
        
        paths = []
        for cat in ["benign", "malicious"]:
            cat_dir = os.path.join(strat_dir, cat)
            if os.path.exists(cat_dir):
                paths.extend(glob.glob(os.path.join(cat_dir, "*.pt")))
        
        if paths:
            variant_dict[strategy] = paths
    return variant_dict
class SupConLoss(nn.Module):
    """
    监督对比损失 (Supervised Contrastive Loss)
    核心思想：
    - 正样本：1. 同一个APK的不同视图；2. 所有同类别的其他APK
    - 负样本：所有不同类别的APK
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, hidden_dim]
            labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        
        # 掩码：label相同则为1 (包括自己)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # 为了数值稳定性，减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 排除自己与自己的对比
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask
        
        # 计算指数和
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # 计算平均对数似然
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # 计算最终损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss

def visualize_embeddings(encoder, dataset, device, sample_size=1000, method='tsne', save_path='embedding_vis.png'):
    """
    可视化图嵌入
    encoder: 预训练编码器 (ContrastEncoder 实例)
    dataset: GraphFileDataset 实例 (包含 .pt 文件路径)
    device: torch device
    sample_size: 随机采样多少个图（太多 t-SNE 会很慢）
    method: 'tsne' 或 'umap'
    save_path: 保存图片路径
    """
    encoder.eval()
    encoder.to(device)
    
    # 随机采样
    indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
    sampled_paths = [dataset.file_paths[i] for i in indices]
    
    all_embs = []
    all_labels = []
    
    # 逐个加载（避免内存爆炸，可分批）
    with torch.no_grad():
        for path in tqdm(sampled_paths, desc="Extracting embeddings"):
            data = torch.load(path)  # 每个 .pt 文件包含一个图
            data = data.to(device)
            # 编码器返回 (projected, graph_emb)
            _, graph_emb = encoder(data.x, data.edge_index, data.batch)
            all_embs.append(graph_emb.cpu().numpy()[0])  # 注意 graph_emb 是 2D: [batch_size, dim]
            all_labels.append(data.y.item())
    
    all_embs = np.array(all_embs)
    all_labels = np.array(all_labels)
    
    # 降维
    print(f"Performing {method.upper()} dimensionality reduction...")
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    elif method == 'umap':
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=8,  # 降低邻居数，聚类更紧凑
            min_dist=0.05,  # 降低最小距离，点更集中
            metric='cosine'  # 使用余弦距离，和对比学习的相似度度量一致
        )
    else:
        raise ValueError("method must be 'tsne' or 'umap'")
    
    emb_2d = reducer.fit_transform(all_embs)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    labels_name = ['Benign', 'Malicious']
    for label in [0, 1]:
        mask = (all_labels == label)
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=colors[label], label=labels_name[label], alpha=0.6, s=20)
    plt.legend()
    plt.title(f"Graph Embeddings Visualization using {method.upper()}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Figure saved to {save_path}")
from itertools import combinations

class PairDataset(Dataset):
    def __init__(self, mv_dataset, original_paths, num_pairs=3, random_seed=42, pair_filter=None):
        """
        Args:
            mv_dataset: MultiVariantDataset 实例
            original_paths: 原始图路径映射（未使用，保留兼容）
            num_pairs: 每个 APK 最多选取多少对
            random_seed: 随机种子
            pair_filter: 可调用对象，接收 (view1, view2) 返回 bool，决定是否保留该对
                         view 格式为 (type, strategy, path)
                         默认 None 表示保留所有对（等价于 'all'）
        """
        random.seed(random_seed)
        self.pairs = []
        self.original_paths = original_paths
        
        if pair_filter is None:
            pair_filter = lambda v1, v2: True  # 保留所有对
        
        print("Building positive pairs with custom filter...")
        
        for apk in tqdm(mv_dataset.valid_apks, desc="APK"):
            views = mv_dataset.view_paths[apk]  # list of (type, strategy, path)
            if len(views) < 2:
                continue
            
            # 生成所有可能的视图索引对 (i, j) 且 i < j
            indices = list(range(len(views)))
            all_pairs = list(combinations(indices, 2))
            
            # 应用过滤器
            filtered_pairs = [(i, j) for i, j in all_pairs if pair_filter(views[i], views[j])]
            
            if len(filtered_pairs) > num_pairs:
                sampled_pairs = random.sample(filtered_pairs, num_pairs)
            else:
                sampled_pairs = filtered_pairs
            
            for i, j in sampled_pairs:
                path1 = views[i][2]
                path2 = views[j][2]
                self.pairs.append((path1, path2))
        
        print(f"Total positive pairs after filter: {len(self.pairs)}")

    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        path1, path2 = self.pairs[idx]
        data1 = torch.load(path1, weights_only=False)
        data2 = torch.load(path2, weights_only=False)
        # 返回数据以及它们的标签 (两个视图标签是一样的，返回data1.y即可)
        if idx == 0:
            print(path1, path2)
            print(path1 == path2)
        return data1, data2, data1.y

def collate_pairs(batch):
    data1_list, data2_list, label_list = zip(*batch)
    batch1 = Batch.from_data_list(data1_list)
    batch2 = Batch.from_data_list(data2_list)
    labels = torch.cat(label_list, dim=0) # 拼接标签
    return batch1, batch2, labels

def scan_existing_graph_data(graph_root):
    """
    扫描已有图数据目录，返回 (original_paths, variant_paths)
    目录结构应为：
        graph_root/original/benign/*.pt
        graph_root/original/malicious/*.pt
        graph_root/variants/{strategy}/benign/*.pt
        graph_root/variants/{strategy}/malicious/*.pt
    """
    original_paths = {}
    variant_paths = defaultdict(dict)

    # 扫描原始图
    original_dir = os.path.join(graph_root, "original")
    if os.path.exists(original_dir):
        for cat in ["benign", "malicious"]:
            cat_dir = os.path.join(original_dir, cat)
            if not os.path.exists(cat_dir):
                continue
            for pt_file in glob.glob(os.path.join(cat_dir, "*.pt")):
                apk_hash = os.path.basename(pt_file).replace('.pt', '')
                original_paths[apk_hash] = pt_file

    # 扫描变种图
    variants_dir = os.path.join(graph_root, "variants")
    if os.path.exists(variants_dir):
        for strategy in os.listdir(variants_dir):
            strat_dir = os.path.join(variants_dir, strategy)
            if not os.path.isdir(strat_dir):
                continue
            for cat in ["benign", "malicious"]:
                cat_dir = os.path.join(strat_dir, cat)
                if not os.path.exists(cat_dir):
                    continue
                for pt_file in glob.glob(os.path.join(cat_dir, "*.pt")):
                    apk_hash = os.path.basename(pt_file).replace('.pt', '')
                    variant_paths[strategy][apk_hash] = pt_file

    return original_paths, dict(variant_paths)

def _generate_data_from_scg(scg, class_label, analyzer, astmodel, flagslist):
    """从SCG生成图数据（供多进程worker使用）"""
    vector_len = 1104
    if len(scg.nodes()) == 0:
        scg.add_node(1)
        adj = nx.to_scipy_sparse_array(scg).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        x = torch.zeros((1, vector_len), dtype=torch.float32)
    else:
        centrality = {
            'degree': nx.degree_centrality(scg),
            'in': nx.in_degree_centrality(scg),
            'out': nx.out_degree_centrality(scg),
            'katz': nx.katz_centrality_numpy(nx.DiGraph(scg)),
            'close': nx.closeness_centrality(scg),
            'between': nx.betweenness_centrality(scg),
        }
        harmonic = nx.harmonic_centrality(scg)
        clustering = nx.clustering(nx.DiGraph(scg))
        square = nx.square_clustering(scg)
        pagerank = nx.pagerank(scg)

        adj = nx.to_scipy_sparse_array(scg).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        x = None

        for node in scg.nodes():
            if not node.is_external():
                AST = analyzer.get_ast_method(node)
                ASTText = cut_doc(str(AST))
                astmodel.random.seed(0)
                ast_vec = torch.from_numpy(astmodel.infer_vector(ASTText))
                flag_vec = torch.tensor([1 if f in AST['flags'] else 0 for f in flagslist])
                api_vec = torch.zeros(len(analyzer.sapi) + 1, dtype=torch.float32)
                base_vec = torch.cat([torch.tensor([0]), ast_vec, flag_vec, api_vec], dim=0)
            else:
                ast_vec = torch.zeros(200, dtype=torch.float32)
                flag_vec = torch.zeros(17, dtype=torch.float32)
                api_vec = torch.zeros(len(analyzer.sapi) + 1, dtype=torch.float32)
                for idx, api in enumerate(analyzer.sapi):
                    cls_name, api_name = analyzer.api_getname(api)
                    if cls_name == node.get_class_name() and api_name[1:-1] == node.name:
                        api_vec[idx] = 1
                api_vec[-1] = 1 if api_vec[:-1].sum() == 0 else 0
                base_vec = torch.cat([torch.tensor([1]), ast_vec, flag_vec, api_vec], dim=0)

            def get_feat(cent_dict, node, default=0.0):
                try:
                    return cent_dict.get(node, default)
                except:
                    return default

            struct_vec = torch.tensor([
                get_feat(centrality['degree'], node),
                get_feat(centrality['in'], node),
                get_feat(centrality['out'], node),
                get_feat(centrality['katz'], node),
                get_feat(centrality['close'], node),
                get_feat(centrality['between'], node),
                get_feat(harmonic, node, 0.0),
                get_feat(clustering, node),
                get_feat(square, node, 0.0),
                get_feat(pagerank, node)
            ], dtype=torch.float32)

            node_vec = torch.cat([base_vec, struct_vec], dim=0)
            if node_vec.shape[0] < vector_len:
                node_vec = torch.cat([node_vec, torch.zeros(vector_len - node_vec.shape[0])])
            elif node_vec.shape[0] > vector_len:
                node_vec = node_vec[:vector_len]

            if x is None:
                x = node_vec.unsqueeze(0)
            else:
                x = torch.vstack([x, node_vec])

    data = Data(x=x.to(torch.float32), edge_index=edge_index, y=torch.tensor([class_label], dtype=torch.long))
    
    assert data.x.dim() == 2, f"[错误] 节点特征 x 必须是 2D 张量，当前维度: {data.x.dim()}"
    assert data.edge_index.dtype == torch.long, f"[错误] 边索引 edge_index 必须是 LongTensor"
    return data


def _parse_original_worker(args):
    """多进程worker：解析原始APK，返回 (apk_hash, data)"""
    apk_path, apk_hash_name, label, basic_wp, astmodel_path, flagslist, sapi, CGType = args
    try:
        from AnalyzeModule import Analyzer
        import torch
        import networkx as nx
        import jieba
        from gensim.models.doc2vec import Doc2Vec

        analyzer = Analyzer(basic_wp)
        analyzer.sapi = sapi
        astmodel = Doc2Vec.load(astmodel_path)
        scg = analyzer.AnalyzeAPK(apk_path, CGType)
        data = _generate_data_from_scg(scg, label, analyzer, astmodel, flagslist)
        return (apk_hash_name, data)
    except Exception as e:
        return None


def _parse_variant_worker_full(args):
    """多进程worker：解析变种APK，返回 (apk_hash, data, strategy)"""
    apk_path, apk_hash_name, label, basic_wp, astmodel_path, flagslist, sapi, CGType, strategy = args
    try:
        from AnalyzeModule import Analyzer
        import torch
        import networkx as nx
        import jieba
        from gensim.models.doc2vec import Doc2Vec

        analyzer = Analyzer(basic_wp)
        analyzer.sapi = sapi
        astmodel = Doc2Vec.load(astmodel_path)
        scg = analyzer.AnalyzeAPK(apk_path, CGType)
        data = _generate_data_from_scg(scg, label, analyzer, astmodel, flagslist)
        if strategy.startswith("Simulated"):
            data.x = GraphAugmentor.add_noise_to_features(data.x, 0.02)
        return (apk_hash_name, data, strategy)
    except:
        return None


def integrate_cic_maldroid_dataset(original_dataset_root, target_integrated_path,
                                    benign_count=None, malicious_per_category=None, force_resample=False):
    """
    自动整合 CICMalDroid2020 数据集：
    - 如果 benign_count 或 malicious_per_category 为 None，则复制该类别下的所有文件。
    - 否则按指定数量采样。
    """
    print("\n" + "=" * 70)
    print("【Step 1: 整合 CICMalDroid2020】")
    print("=" * 70)

    integration_flag = os.path.join(target_integrated_path, ".integration_complete")
    if os.path.exists(integration_flag) and not force_resample:
        print(Fore.GREEN + "  [√] 数据集已按要求整合过，跳过此步骤" + Style.RESET_ALL)
        return target_integrated_path

    category_mapping = {
        "Benign": "benign",
        "Adware": "malicious",
        "Banking": "malicious",
        "SMS": "malicious",
        "Riskware": "malicious",
    }

    # 如果数量参数为 None，则表示不限制数量（复制全部）
    sampling_targets = {
        "Benign": benign_count,
        "Adware": malicious_per_category,
        "Banking": malicious_per_category,
        "SMS": malicious_per_category,
        "Riskware": malicious_per_category,
    }

    if force_resample and os.path.exists(target_integrated_path):
        shutil.rmtree(target_integrated_path)

    benign_target_dir = os.path.join(target_integrated_path, "benign")
    malicious_target_dir = os.path.join(target_integrated_path, "malicious")
    os.makedirs(benign_target_dir, exist_ok=True)
    os.makedirs(malicious_target_dir, exist_ok=True)

    stats = {"benign": 0, "malicious": 0, "skipped": 0, "renamed": 0}

    for src_category_name in os.listdir(original_dataset_root):
        src_category_path = os.path.join(original_dataset_root, src_category_name)
        if not os.path.isdir(src_category_path):
            continue
        if src_category_name not in category_mapping:
            print(f"  [跳过] 未知目录: {src_category_name}")
            continue

        target_category_name = category_mapping[src_category_name]
        target_dir = benign_target_dir if target_category_name == "benign" else malicious_target_dir
        
        all_files = [f for f in glob.glob(os.path.join(src_category_path, "*")) if os.path.isfile(f)]
        random.shuffle(all_files)
        target_count = sampling_targets.get(src_category_name, None)  # 可能是 None
        if target_count is None:
            target_count = len(all_files)   # 全部复制
        print(f"\n  处理: {src_category_name:10s} -> {target_category_name:10s} (目标: {target_count} / 可用: {len(all_files)})")

        copied_count = 0
        for file_path in tqdm(all_files, desc=f"  复制 {src_category_name}"):
            if copied_count >= target_count:
                break
            original_filename = os.path.basename(file_path)
            target_filename = original_filename if original_filename.endswith('.apk') else original_filename + '.apk'
            target_file_path = os.path.join(target_dir, target_filename)

            if os.path.exists(target_file_path):
                stats["skipped"] += 1
                copied_count += 1
                continue

            try:
                shutil.copy2(file_path, target_file_path)
                stats[target_category_name] += 1
                copied_count += 1
            except Exception as e:
                print(f"\n  [警告] 复制失败 {original_filename}: {str(e)[:50]}")
                stats["skipped"] += 1

    with open(integration_flag, 'w') as f:
        f.write(f"Integration done. Benign: {stats['benign']}, Malicious: {stats['malicious']}")

    print("\n" + "-" * 60)
    print("  数据集整合完成统计:")
    print(f"    良性样本 (Benign):   {stats['benign']}")
    print(f"    恶意样本 (Malicious): {stats['malicious']}")
    print(f"    重命名文件:           {stats['renamed']}")
    print(f"    整合后数据路径:      {target_integrated_path}")
    print("=" * 70)

    return target_integrated_path


def hash_rename_apks(original_path, working_path, random_seed=42):
    """
    基于文件内容MD5进行重命名，避免文件名冲突，并建立映射
    """
    renamed_path = os.path.join(working_path, "renamed_data")
    rename_flag = os.path.join(renamed_path, ".rename_complete")
    
    if os.path.exists(rename_flag):
        print(Fore.GREEN + "\n[√] APK已Hash重命名过，跳过此步骤" + Style.RESET_ALL)
        mapping_path = os.path.join(renamed_path, "hash_mapping.pkl")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                hash_mapping = pickle.load(f)
            return renamed_path, hash_mapping
            
    os.makedirs(renamed_path, exist_ok=True)
    hash_mapping = {}
    categories = ["benign", "malicious"]
    
    print("\n[APK Hash重命名]")
    for category in categories:
        cat_path = os.path.join(original_path, category)
        if not os.path.exists(cat_path):
            continue
        apk_files = glob.glob(os.path.join(cat_path, "*.apk"))
        for apk_path in tqdm(apk_files, desc=f"  Hash {category}"):
            with open(apk_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:16]
            
            original_name = os.path.basename(apk_path)
            new_name = f"{file_hash}.apk"
            target_dir = os.path.join(renamed_path, category)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, new_name)
            
            if not os.path.exists(target_path):
                shutil.copy2(apk_path, target_path)
            
            hash_mapping[file_hash] = {
                'original_path': apk_path,
                'original_name': original_name,
                'label': 0 if category == "benign" else 1,
                'new_name': new_name
            }
    
    mapping_path = os.path.join(renamed_path, "hash_mapping.pkl")
    with open(mapping_path, 'wb') as f:
        pickle.dump(hash_mapping, f)
    
    with open(rename_flag, 'w') as f:
        f.write("Hash rename completed")
    
    print(f"  共处理 {len(hash_mapping)} 个APK")
    return renamed_path, hash_mapping


# ==================== 变种生成====================
class APKVariantGenerator:
    ALL_OBFUSCATION_STRATEGIES = {
        "ConstStringEncryption": ["ConstStringEncryption", "Rebuild", "NewAlignment", "NewSignature"],
        "Nop": ["Nop", "Rebuild", "NewAlignment", "NewSignature"],
        "CallIndirection": ["CallIndirection", "Rebuild", "NewAlignment", "NewSignature"],
        "Reflection": ["Reflection", "Rebuild", "NewAlignment", "NewSignature"],
        "Rename_CM": ["ClassRename", "MethodRename", "Rebuild", "NewAlignment", "NewSignature"],
        "Rename_M": ["MethodRename", "Rebuild", "NewAlignment", "NewSignature"],
        "Reorder": ["Reorder", "Rebuild", "NewAlignment", "NewSignature"],
    }

    def __init__(self, original_path, variant_path, candidate_strategies=None, random_seed=None, use_fallback=True):
        self.original_path = original_path
        self.variant_path = variant_path
        self.use_fallback = use_fallback
        
        # --- 新增：失败记录管理 ---
        self.failed_apks = []
        self.failed_record_path = os.path.join(self.variant_path, ".failed_apks_record.pkl")
        self.failed_apks_mapping = self._load_failed_records() # 结构: { (apk_base_name, strategy): error_msg }
        
        self.apk_strategies_mapping = defaultdict(list)
        self.strategy_apk_mapping = defaultdict(list)
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.random_seed = random_seed
        if candidate_strategies is not None:
            self.candidate_strategies = [s for s in candidate_strategies if s in self.ALL_OBFUSCATION_STRATEGIES]
        else:
            self.candidate_strategies = list(self.ALL_OBFUSCATION_STRATEGIES.keys())
        
        print(f"\n[变种生成器初始化]")
        print(f"  候选混淆策略池 ({len(self.candidate_strategies)}种): {self.candidate_strategies}")
        print(f"  已记录历史失败组合: {len(self.failed_apks_mapping)} 个")
        self.obfuscapk_available = self._check_obfuscapk()

    # --- 新增：加载失败记录 ---
    def _load_failed_records(self):
        if os.path.exists(self.failed_record_path):
            try:
                with open(self.failed_record_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}

    # --- 新增：保存失败记录 ---
    def _save_failed_records(self):
        try:
            with open(self.failed_record_path, 'wb') as f:
                pickle.dump(self.failed_apks_mapping, f)
        except Exception as e:
            print(Fore.YELLOW + f"  [!] 保存失败记录失败: {e}" + Style.RESET_ALL)

    def _check_obfuscapk(self):
        try:
            from obfuscapk import main as ob
            print(Fore.GREEN + "  [√] obfuscapk工具可用" + Style.RESET_ALL)
            return True
        except ImportError:
            print(Fore.YELLOW + "  [!] obfuscapk未安装，将使用模拟变种" + Style.RESET_ALL)
            return False
        except Exception as e:
            print(Fore.YELLOW + f"  [!] obfuscapk依赖检查失败: {e}" + Style.RESET_ALL)
            return False

    def generate_variants(self, skip_existing=True):
        print(f"\n[开始生成变种]")
        variant_flag = os.path.join(self.variant_path, ".variants_generated")
        if os.path.exists(variant_flag) and skip_existing:
            print(Fore.GREEN + "  [√] 变种已生成过，跳过此步骤" + Style.RESET_ALL)
            self.load_mapping()
            return self.get_variant_info()

        if not self.obfuscapk_available:
            return self._generate_simulated_variants()

        from obfuscapk import main as ob
        stats = defaultdict(int)
        categories = ["benign", "malicious"]

        for category in categories:
            category_path = os.path.join(self.original_path, category)
            if not os.path.exists(category_path):
                continue
            apk_files = glob.glob(os.path.join(category_path, "*.apk"))
            print(f"\n[处理 {category}] 共 {len(apk_files)} 个APK")

            for apk_path in tqdm(apk_files, desc=f"生成{category}变种"):
                apk_name = os.path.basename(apk_path)
                apk_base_name = apk_name.replace('.apk', '')

                for strategy in self.candidate_strategies:
                    # --- 修改：检查是否有历史失败记录 ---
                    fail_key = (apk_base_name, strategy)
                    if fail_key in self.failed_apks_mapping:
                        stats[f"{strategy}_failed_hist"] += 1
                        continue

                    if strategy not in self.apk_strategies_mapping[apk_base_name]:
                        self.apk_strategies_mapping[apk_base_name].append(strategy)

                    output_dir = os.path.join(self.variant_path, strategy, category)
                    os.makedirs(output_dir, exist_ok=True)
                    output_apk = os.path.join(output_dir, apk_name)

                    if skip_existing and os.path.exists(output_apk):
                        stats[f"{strategy}_skipped"] += 1
                        if apk_base_name not in self.strategy_apk_mapping[strategy]:
                            self.strategy_apk_mapping[strategy].append(apk_base_name)
                        continue

                    success = False
                    temp_dir = os.path.join(output_dir, apk_base_name)
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

                    try:
                        ob.perform_obfuscation(
                            input_apk_path=apk_path,
                            working_dir_path=output_dir,
                            obfuscator_list=self.ALL_OBFUSCATION_STRATEGIES[strategy],
                            interactive=False
                        )
                        out_files = glob.glob(os.path.join(output_dir, "*_obfuscated.apk"))
                        if out_files and os.path.exists(out_files[0]):
                            shutil.move(out_files[0], output_apk)
                            success = True
                        elif os.path.exists(output_apk):
                            success = True
                    except Exception as e:
                        error_msg = str(e)[:80]
                        self.failed_apks.append({'apk': apk_name, 'strategy': strategy, 'error': error_msg})
                        # --- 修改：记录失败并保存 ---
                        self.failed_apks_mapping[fail_key] = error_msg
                        self._save_failed_records()

                    if success:
                        stats[f"{strategy}_success"] += 1
                        if apk_base_name not in self.strategy_apk_mapping[strategy]:
                            self.strategy_apk_mapping[strategy].append(apk_base_name)
                    else:
                        stats[f"{strategy}_failed"] += 1
                        if self.use_fallback:
                            try:
                                shutil.copy2(apk_path, output_apk)
                                stats[f"{strategy}_fallback"] += 1
                            except:
                                pass

                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

        self._print_stats(stats)
        self._save_mapping()
        with open(variant_flag, 'w') as f:
            f.write("Done")
        return self.get_variant_info()

    def _generate_simulated_variants(self):
        print(Fore.YELLOW + "\n[模拟变种生成] 直接复制原始APK" + Style.RESET_ALL)
        stats = defaultdict(int)
        categories = ["benign", "malicious"]
        for category in categories:
            category_path = os.path.join(self.original_path, category)
            if not os.path.exists(category_path): continue
            apk_files = glob.glob(os.path.join(category_path, "*.apk"))
            for apk_path in tqdm(apk_files, desc=f"模拟{category}"):
                apk_name = os.path.basename(apk_path)
                apk_base_name = apk_name.replace('.apk', '')
                for strategy in self.candidate_strategies:
                    # --- 修改：模拟模式下也跳过历史失败记录（如果有） ---
                    fail_key = (apk_base_name, strategy)
                    if fail_key in self.failed_apks_mapping:
                        continue
                        
                    output_dir = os.path.join(self.variant_path, strategy, category)
                    os.makedirs(output_dir, exist_ok=True)
                    output_apk = os.path.join(output_dir, apk_name)
                    if not os.path.exists(output_apk):
                        shutil.copy2(apk_path, output_apk)
                    if apk_base_name not in self.strategy_apk_mapping[strategy]:
                        self.strategy_apk_mapping[strategy].append(apk_base_name)
                    if strategy not in self.apk_strategies_mapping[apk_base_name]:
                        self.apk_strategies_mapping[apk_base_name].append(strategy)
                    stats[f"{strategy}_simulated"] += 1
        self._save_mapping()
        return self.get_variant_info()

    def _print_stats(self, stats):
        pass

    def _save_mapping(self):
        mapping_path = os.path.join(self.variant_path, "apk_strategy_mapping.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump({'apk_to_strategies': dict(self.apk_strategies_mapping),
                         'strategy_to_apks': dict(self.strategy_apk_mapping)}, f)

    def load_mapping(self):
        mapping_path = os.path.join(self.variant_path, "apk_strategy_mapping.pkl")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                d = pickle.load(f)
            self.apk_strategies_mapping = defaultdict(list, d['apk_to_strategies'])
            self.strategy_apk_mapping = defaultdict(list, d['strategy_to_apks'])

    def get_variant_info(self):
        info = {}
        if not os.path.exists(self.variant_path): return info
        for s in os.listdir(self.variant_path):
            p = os.path.join(self.variant_path, s)
            if os.path.isdir(p) and not s.startswith('.'):
                cnt = len(glob.glob(os.path.join(p, "*", "*.apk")))
                if cnt>0: info[s] = cnt
        return info


# ==================== 图增强工具 ====================
class GraphAugmentor:
    @staticmethod
    def node_dropping(x, edge_index, batch, p=0.1):
        """
        对每个图独立进行节点丢弃，确保每个图至少保留一个节点。
        """
        device = x.device
        unique_batches = batch.unique()
        new_x_list = []
        new_edge_index_list = []
        new_batch_list = []
        
        for b in unique_batches:
            # 获取当前图的所有节点索引
            node_mask = (batch == b)
            node_idx = torch.where(node_mask)[0]
            x_b = x[node_mask]
            n_nodes = x_b.size(0)
            if n_nodes == 0:
                continue
            
            # 生成当前图的节点保留掩码
            keep = torch.rand(n_nodes, device=device) > p
            if keep.sum() == 0:
                keep[0] = True   # 至少保留一个节点
            
            # 保留的节点在原始图中的索引
            keep_nodes = node_idx[keep]
            
            # 提取子图的边
            # 边两端节点都必须被保留
            src, dst = edge_index
            edge_keep_mask = torch.isin(src, keep_nodes) & torch.isin(dst, keep_nodes)
            edge_index_b = edge_index[:, edge_keep_mask]
            
            # 重映射节点编号：从 0 开始连续编号
            node_map = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
            node_map[keep_nodes] = torch.arange(len(keep_nodes), device=device)
            edge_index_b = node_map[edge_index_b]
            
            # 收集
            new_x_list.append(x_b[keep])
            new_edge_index_list.append(edge_index_b)
            new_batch_list.append(torch.full((len(keep_nodes),), b, dtype=torch.long, device=device))
        
        if not new_x_list:
            # 极端情况：所有图都被删空了，返回原始数据（或保留一个节点）
            return x, edge_index, batch
        
        new_x = torch.cat(new_x_list, dim=0)
        new_edge_index = torch.cat(new_edge_index_list, dim=1)
        new_batch = torch.cat(new_batch_list, dim=0)
        return new_x, new_edge_index, new_batch
    @staticmethod
    def feature_masking(x, p=0.1):
        mask = torch.rand(x.size(), device=x.device) > p
        return x * mask.float()
    @staticmethod
    def edge_removing(edge_index, p=0.1):
        edge_index, _ = dropout_adj(edge_index, p=p, force_undirected=False)
        return edge_index
    @staticmethod
    def random_choice_augment(x, edge_index, batch):
        t = np.random.choice(['node_drop', 'feature_mask', 'edge_remove'])
        if t == 'node_drop': return GraphAugmentor.node_dropping(x, edge_index, batch)
        elif t == 'feature_mask': return GraphAugmentor.feature_masking(x), edge_index, batch
        else: return x, GraphAugmentor.edge_removing(edge_index), batch
    @staticmethod
    def identity(x, edge_index, batch): return x, edge_index, batch
    @staticmethod
    def add_noise_to_features(x, noise_level=0.1):
        return x + torch.randn_like(x) * noise_level


# ==================== 对比学习编码器 ====================
class ModelAP(torch.nn.Module):
    def __init__(self, input_dim, hidden_channels, dropout=0.5, keep_ratio=0.5, proj_dim=256):
        super(ModelAP, self).__init__()
        self.hidden_channels = hidden_channels   # 保存供外部获取图嵌入维度
        self.dropout = dropout
        self.keep_ratio = keep_ratio

        self.gcn1 = GraphConv(input_dim, hidden_channels)
        self.gcn2 = GraphConv(hidden_channels, hidden_channels)
        self.gcn3 = GraphConv(hidden_channels, hidden_channels)
        self.sagp = SAGPooling(hidden_channels * 3, self.keep_ratio, multiplier=8)

        # 图嵌入维度：经过 SAGPooling 后节点特征维度为 hidden_channels*3，
        # 再 mean+max 拼接后为 hidden_channels*6
        self.graph_emb_dim = hidden_channels * 6

        # 投影头（用于对比学习）
        self.projection_head = nn.Sequential(
            nn.Linear(self.graph_emb_dim, hidden_channels * 4),
            nn.ReLU(),
            nn.Linear(hidden_channels * 4, proj_dim)
        )

    def forward(self, x, edge_index, batch):
        gcn1 = F.relu(self.gcn1(x, edge_index))
        gcn2 = F.relu(self.gcn2(gcn1, edge_index))
        gcn3 = F.relu(self.gcn3(gcn2, edge_index))

        pre_gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1)
        px, _, _, pbatch, _, _ = self.sagp(pre_gcn_feature, edge_index, batch=batch)

        mean_pool = global_mean_pool(px, pbatch)
        max_pool = global_max_pool(px, pbatch)
        graph_emb = torch.cat((mean_pool, max_pool), dim=1)   

        projected = self.projection_head(graph_emb)
        return projected, graph_emb   


class InfoNCE(nn.Module):
    def __init__(self, temp=0.2):
        super().__init__()
        self.temp = temp
    def forward(self, z1, z2):
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        sim = torch.matmul(z1, z2.T) / self.temp
        labels = torch.arange(z1.size(0), device=z1.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


PairwiseContrastLoss = InfoNCE


# ==================== 按需加载的数据集类 ====================
class GraphFileDataset(torch.utils.data.Dataset):
    """
    从单个 .pt 文件按需加载图数据，避免内存爆炸。
    file_paths: 路径列表（可以是原始图或变种图）
    """
    def __init__(self, file_paths):
        self.file_paths = list(file_paths)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return torch.load(self.file_paths[idx],weights_only=False)


class MultiVariantDataset:
    """
    多变种数据集：支持从保存的图文件中按需加载同一APK的多个视图。
    original_paths: {apk_hash: file_path}
    variant_paths: {strategy: {apk_hash: file_path}}
    """
    def __init__(self, original_paths, variant_paths, min_required_pairs=3):
        self.original_paths = original_paths
        self.variant_paths = variant_paths
        self.valid_apks = []
        self.view_paths = {}  # apk_hash -> list of (type, strategy, path)
        # 构建有效APK列表和视图索引
        for apk_hash in original_paths.keys():
            views = [('original', None, original_paths[apk_hash])]
            for s, vd in variant_paths.items():
                if apk_hash in vd:
                    views.append(('variant', s, vd[apk_hash]))
            n_views = len(views)
            # 计算组合数 C(n,2) = n*(n-1)/2
            max_possible_pairs = n_views * (n_views - 1) // 2
            
            # 只有当最大对数 >= 要求的最小对数（3）时，才加入有效列表
            if max_possible_pairs >= min_required_pairs:
                self.valid_apks.append(apk_hash)
                self.view_paths[apk_hash] = views
        
        print(f"[MultiVariantDataset] 有效APK (至少能构成{min_required_pairs}对): {len(self.valid_apks)}")
        

    def get_multiple_pairs(self, apk_name, num_pairs=5):
        """返回指定数量的不同视图对 (data_i, data_j) 列表"""
        views = self.view_paths.get(apk_name, [])
        if len(views) < 2:
            return []
        # 生成所有视图索引对
        indices = list(range(len(views)))
        pairs_idx = [(i, j) for i in range(len(indices)) for j in range(i+1, len(indices))]
        if len(pairs_idx) > num_pairs:
            pairs_idx = random.sample(pairs_idx, num_pairs)
        pairs_data = []
        for i, j in pairs_idx:
            # 按需加载图数据
            data_i = torch.load(views[i][2])
            data_j = torch.load(views[j][2])
            pairs_data.append((data_i, data_j))
        return pairs_data


# ==================== 图数据保存与加载 ====================
def save_graph_data(data, save_dir, apk_hash):
    """保存单个图数据到文件，返回文件路径"""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{apk_hash}.pt")
    torch.save(data, file_path)
    return file_path


def load_and_process_graph_data(ea_instance, renamed_apk_path, variant_path, hash_mapping,
                                 CGType='Xref', force_reparse=False):
    """
    统一的图数据处理入口：
    1. 先检查缓存（返回路径映射）
    2. 解析原始APK (并行) -> 保存为.pt文件，返回路径映射
    3. 解析变种APK (并行) -> 保存为.pt文件，返回路径映射
    4. 保存缓存（路径映射）
    """
    cache_path = os.path.join(variant_path, "graph_path_cache.pkl")
    graph_root = os.path.join(variant_path, "graph_data")
    if os.path.exists(cache_path) and not force_reparse:
        print(Fore.GREEN + "\n[√] 加载已缓存的图路径映射" + Style.RESET_ALL)
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"  原始图路径数: {len(cache['original'])}")
            print(f"  变种策略: {list(cache['variants'].keys())}")
            return cache['original'], cache['variants']
        except Exception as e:
            print(Fore.YELLOW + f"  [!] 缓存加载失败，重新解析: {e}" + Style.RESET_ALL)

    print("\n" + "=" * 60)
    print("【解析APK为图数据并保存】")
    print("=" * 60)

    categories = ["benign", "malicious"]

    # ========== 并行解析原始APK ==========
    print("\n[1/2] 解析原始APK（并行）...")
    original_tasks = []
    for cat in categories:
        cat_path = os.path.join(renamed_apk_path, cat)
        if not os.path.exists(cat_path):
            continue
        apks = glob.glob(os.path.join(cat_path, "*.apk"))
        for apk_path in apks:
            apk_hash_name = os.path.basename(apk_path).replace('.apk', '')
            if apk_hash_name not in hash_mapping:
                continue
            label = hash_mapping[apk_hash_name]['label']
            original_tasks.append((apk_path, apk_hash_name, label))

    basic_wp = ea_instance._basic_wp
    astmodel_path = ea_instance.astmodel_path
    flagslist = ea_instance.flagslist
    sapi = ea_instance._analyzer.sapi

    original_args = [(apk_path, apk_hash_name, label, basic_wp, astmodel_path, flagslist, sapi, CGType)
                     for apk_path, apk_hash_name, label in original_tasks]

    original_paths = {}
    num_processes = 8
    if num_processes > 0:
        with mp.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(_parse_original_worker, original_args),
                                total=len(original_args), desc="解析原始APK"))
            for res in results:
                if res is not None:
                    apk_hash_name, data = res
                    label = data.y.item()
                    category = "benign" if label == 0 else "malicious"
                    save_dir = os.path.join(graph_root, "original", category)
                    file_path = save_graph_data(data, save_dir, apk_hash_name)
                    original_paths[apk_hash_name] = file_path

    print(f"  原始APK解析完成: {len(original_paths)} 个")

    # ========== 并行解析变种APK ==========
    variant_paths = defaultdict(dict)
    print("\n[2/2] 解析变种APK（并行）...")

    if os.path.exists(variant_path):
        strategies = [d for d in os.listdir(variant_path) if os.path.isdir(os.path.join(variant_path, d)) and not d.startswith('.')]

        # 收集所有变种解析任务
        variant_tasks = []
        for strategy in strategies:
            strat_path = os.path.join(variant_path, strategy)
            for cat in categories:
                cat_path = os.path.join(strat_path, cat)
                if not os.path.exists(cat_path):
                    continue
                apks = glob.glob(os.path.join(cat_path, "*.apk"))
                for apk_path in apks:
                    apk_hash_name = os.path.basename(apk_path).replace('.apk', '')
                    if apk_hash_name not in original_paths:
                        continue
                    label = hash_mapping[apk_hash_name]['label']
                    variant_tasks.append((apk_path, apk_hash_name, label, basic_wp, astmodel_path, flagslist, sapi, CGType, strategy))

        if variant_tasks:
            with mp.Pool(processes=num_processes) as pool:
                results = list(tqdm(pool.imap_unordered(_parse_variant_worker_full, variant_tasks),
                                    total=len(variant_tasks), desc="解析变种APK"))
                for res in results:
                    if res is not None:
                        apk_hash_name, data, strategy = res
                        label = data.y.item()
                        category = "benign" if label == 0 else "malicious"
                        save_dir = os.path.join(graph_root, "variants", strategy, category)
                        file_path = save_graph_data(data, save_dir, apk_hash_name)
                        variant_paths[strategy][apk_hash_name] = file_path

        # 统计各策略数量
        for strategy, apk_dict in variant_paths.items():
            print(f"  {strategy:20s}: {len(apk_dict)} 个变种")

    # 3. 保存缓存（路径映射）
    print("\n[保存图路径缓存]")
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'original': original_paths,
                'variants': dict(variant_paths)
            }, f)
        print(Fore.GREEN + f"  [√] 缓存已保存: {cache_path}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.YELLOW + f"  [!] 缓存保存失败: {e}" + Style.RESET_ALL)

    return original_paths, dict(variant_paths)


# ==================== 图增强对比学习 ====================
def graph_augment_pretrain(encoder, train_dataset, device, epochs=100, batch_size=8, lr=0.001):
    """
    图增强对比学习预训练
    train_dataset: GraphFileDataset 实例
    """
    print("\n[图增强对比学习预训练]")
    loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    opt = AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    criterion = InfoNCE(temp=0.2)
    best_loss = float('inf')
    encoder.train()
    
    with tqdm(total=epochs, desc='Pre-train') as pbar:
        for epoch in range(epochs):
            total_loss = 0.0
            for data in loader:
                data = data.to(device)
                opt.zero_grad()
                x1, e1, b1 = GraphAugmentor.identity(data.x, data.edge_index, data.batch)
                x2, e2, b2 = GraphAugmentor.random_choice_augment(data.x.clone(), data.edge_index.clone(), data.batch.clone())
                z1, _ = encoder(x1, e1, b1)
                z2, _ = encoder(x2, e2, b2)
                loss = criterion(z1, z2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                opt.step()
                total_loss += loss.item() * data.num_graphs
            sched.step()
            avg_loss = total_loss / len(loader.dataset)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(encoder.state_dict(), './graph_aug_enc.pth')
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            pbar.update()
    return encoder

def variant_contrast_pretrain(encoder, train_loader, device, epochs=100, lr=0.0001,
                              accum_steps=1, temperature=0.2, eval_loader=None,
                              save_path='./var_enc.pth'):
    optimizer = AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = InfoNCE(temp=temperature)
    best_loss = float('inf')
    encoder.train()

    # 使用 tqdm 包装 epoch 循环
    pbar = tqdm(total=epochs, desc='Variant Contrast Pre-train', unit='epoch')
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        for step, (batch1, batch2, labels) in enumerate(train_loader):
            batch1 = batch1.to(device, non_blocking=True)
            batch2 = batch2.to(device, non_blocking=True)

            z1, _ = encoder(batch1.x, batch1.edge_index, batch1.batch)
            z2, _ = encoder(batch2.x, batch2.edge_index, batch2.batch)

            loss = (criterion(z1, z2) + criterion(z2, z1)) / 2
            loss = loss / accum_steps
            loss.backward()

            total_loss += loss.item() * accum_steps
            num_batches += 1

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if (step + 1) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_loss / max(1, num_batches)

        # 可选评估信息
        if eval_loader is not None and (epoch % 5 == 0 or epoch == epochs - 1):
            pos_sim, neg_sim = evaluate_view_invariance(encoder, eval_loader, device, num_pairs=500)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'pos_sim': f'{pos_sim:.4f}',
                'neg_sim': f'{neg_sim:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        else:
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        pbar.update(1)

        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), save_path)

    pbar.close()

    if os.path.exists(save_path):
        encoder.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"加载最佳预训练模型，最终 loss={best_loss:.4f}")

    return encoder

# ==================== 分类器 ====================
class MLPClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2, hidden_dim=256, dropout=0.5, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        input_dim = encoder.graph_emb_dim   # 从编码器获取图嵌入维度
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch):
        _, graph_emb = self.encoder(x, edge_index, batch)
        return self.classifier(graph_emb)
    
# ==================== 工具类 ====================
class PrintColor():
    def __init__(self):
        self.END = "\033[0m"
        self.GREEN = "\033[0;32;40m"
        self.YELLOW = "\033[0;33;40m"


class SimpleEpochProgress(CallbackAny2Vec):
    def __init__(self, total_epochs): self.total = total_epochs
    def on_train_begin(self, model): self.pbar = tqdm(total=self.total, desc="Doc2Vec")
    def on_epoch_end(self, model): self.pbar.update(1)
    def on_train_end(self, model): self.pbar.close()


def cut_doc(text):
    stop_list = ["'", ',']
    return [w for w in jieba.cut(text) if w not in stop_list]


def read_corups(corups):
    for i, line in enumerate(corups):
        yield TaggedDocument(cut_doc(str(line)), [i])


def Docmodel(fname, corups, vector_size=200, epochs=50, workers=4):
    print("预处理文档...")
    from multiprocessing import Pool
    with Pool(processes=workers) as pool:
        tokenized = list(tqdm(pool.imap(cut_doc, (str(line) for line in corups)), total=len(corups), desc="分词"))
    train_corups = [TaggedDocument(tokens, [i]) for i, tokens in enumerate(tokenized)]
    
    print('\n训练Doc2Vec模型')
    epoch_progress = SimpleEpochProgress(epochs)
    model = Doc2Vec(train_corups, vector_size=vector_size, seed=0, min_count=2, epochs=epochs, workers=workers, callbacks=[epoch_progress])
    model.callbacks = []
    model.save(fname)
    print('Doc2Vec模型训练完成')


def on_task_error(exc):
    print(''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


# ==================== 主类 ExpAndro ====================
class ExpAndro():
    def __init__(self, model="default", dataset="default", run_id="default",
                 data_path='', working_path="./"):
        self.pc = PrintColor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flagslist = ['public', 'private', 'protected', 'static', 'final',
                          'synchronized', 'bridge', 'varargs', 'native', 'interface',
                          'abstract', 'strictfp', 'synthetic', 'enum', 'unused',
                          'constructor', 'declared_synchronized']
        self._model_name = model
        self._datasetstr = dataset
        self.runid = run_id
        if not working_path.endswith('/'): working_path += '/'
        self._basic_wp = working_path
        self._working_path = os.path.join(working_path, "Output", "ExpAndro_Data")
        os.makedirs(self._working_path, exist_ok=True)
        self._relatedfile_path = os.path.join(self._working_path, 'RelatedFile')
        os.makedirs(self._relatedfile_path, exist_ok=True)
        self._data_path = data_path
        self._datafile_path = os.path.join(self._working_path, self._datasetstr)
        os.makedirs(self._datafile_path, exist_ok=True)
        self.corups_path = os.path.join(self._relatedfile_path, f'{self._datasetstr}_corups.pkl')
        self.astmodel_path = os.path.join(self._relatedfile_path, f'{self._datasetstr}_astmodel')
        self.fname = self.astmodel_path
        self.dataset_path = os.path.join(self._relatedfile_path, f'{self._datasetstr}_dataset.pkl')
        self._analyzer = Analyzer(self._basic_wp)
        self._astmodel = self._load_astmodel()
        self.pretrained_encoder = None
        self.classifier = None
        self.hidden_dim = 128

    def _load_astmodel(self):
        try:
            model = Doc2Vec.load(self.fname)
            print(f"AST2Vec model loaded: {self.astmodel_path}")
            return model
        except:
            return None

    def _ASTCorups_Collect(self, Filelist, CGType='Xref'):
        for file in tqdm(Filelist, desc="收集AST"):
            try:
                scg = self._analyzer.AnalyzeAPK(file, CGType)
                self._analyzer.Feature_collection(scg)
            except:
                pass

    def ASTModel_generate(self, APKlist, CGtype='Xref', recollect=False):
        if recollect or not os.path.exists(self.corups_path):
            if os.path.exists(self.corups_path): os.remove(self.corups_path)
            print('\nCollect ASTs')
            self._ASTCorups_Collect(APKlist, CGtype)
            with open(self.corups_path, 'wb') as f:
                pickle.dump(self._analyzer.corups, f)
        with open(self.corups_path, 'rb') as f:
            read_corups_data = pickle.load(f)
        Docmodel(self.fname, read_corups_data, vector_size=200, epochs=30, workers=mp.cpu_count())
        self._astmodel = Doc2Vec.load(self.fname)
        return self._astmodel

    def _data_generate(self, scg, class_label=0):
        return _generate_data_from_scg(scg, class_label, self._analyzer, self._astmodel, self.flagslist)

    def Get_APKlist(self):
        files = glob.glob(os.path.join(self._data_path, '**', '*.apk'), recursive=True)
        return files

    class GAT(nn.Module):
        def __init__(self, hidden_channels, num_node_features, num_classes):
            super().__init__()
            #显式指定所有关键参数
            self.conv1 = GATConv(
                in_channels=num_node_features,
                out_channels=hidden_channels,
                heads=4,
                concat=False,
                dropout=0.0,
                add_self_loops=True
            )
            self.conv2 = GATConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=4,
                concat=False,
                dropout=0.0,
                add_self_loops=True
            )
            self.classifier = nn.Linear(hidden_channels, num_classes)
        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index).relu()
            x = global_mean_pool(x, batch)
            return self.classifier(x)

    def _train(self, model, optimizer, criterion, train_loader, device):
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)

    def _test(self, model, criterion, test_loader, device):
        model.eval()
        Y, y_pred = [], []
        total_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device, non_blocking=True)
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                total_loss += loss.item() * data.num_graphs
                pred = out.argmax(dim=1)
                Y.extend(data.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        accuracy = metrics.accuracy_score(Y, y_pred)
        f1 = metrics.f1_score(Y, y_pred, average='binary', zero_division=0)
        return total_loss / len(test_loader.dataset), accuracy, f1

    def Model_train_test(self, train_dataset, test_dataset, hidden_dim=128, lr=0.0001, epochs=150, use_pretrained=False,freeze_encoder=False,plot_curves=True ):
        device = self.device
        print(f'\n[模型训练] 设备：{device}')
        train_loader = GraphDataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16,pin_memory=True)
        test_loader = GraphDataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16,pin_memory=True)
        num_features = train_dataset[0].x.shape[1]
        
        if use_pretrained and self.pretrained_encoder is not None:
            print("[使用预训练编码器]")
            # 注意：这里使用新的 MLPClassifier
            model = MLPClassifier(
                encoder=self.pretrained_encoder,
                num_classes=2,
                hidden_dim=128,        # 可根据需要调整
                dropout=0.6,
                freeze_encoder=freeze_encoder
            ).to(device)
        else:
            print("[使用GAT]")
            model = self.GAT(hidden_dim, num_features, 2).to(device)
            
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_acc = 0.0
        early_stop = 0
        best_model_path = os.path.join(self._relatedfile_path, 'best_model.pt')
        # ---------- 新增：记录历史数据 ----------
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        epoch_list = []
        
        for epoch in range(1, epochs+1):
            self._train(model, optimizer, criterion, train_loader, device)
            train_loss, train_acc, _ = self._test(model, criterion, train_loader, device)
            test_loss, test_acc, test_f1 = self._test(model, criterion, test_loader, device)
            scheduler.step(test_acc)
            # ---------- 记录 ----------
            epoch_list.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(test_loss)
            train_accs.append(train_acc)
            val_accs.append(test_acc)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_f1 = test_f1
                torch.save(model.state_dict(), best_model_path)
                early_stop = 0
            else:
                early_stop += 1
                
            if epoch % 5 == 0:
                print(f'Epoch: {epoch:03d} | Test Acc: {test_acc:.4f}|Best Acc:{best_acc:.4f}')
            if early_stop > 30: break
            
        print(f'最佳 Test Acc: {best_acc:.4f}, F1: {best_f1:.4f}')
        self.classifier = model
        # ---------- 绘制曲线 ----------
        if plot_curves:
            self._plot_training_curves(epoch_list, train_losses, val_losses,
                                    train_accs, val_accs, best_model_path)

        return model

    def _plot_training_curves(self, epochs, train_loss, val_loss, train_acc, val_acc, model_save_path):
        """绘制训练过程中的损失曲线和准确率曲线，并保存图片"""
        plt.figure(figsize=(12, 5))

        # 损失曲线子图
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)

        # 准确率曲线子图
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        # 保存在最佳模型相同目录下
        plot_path = os.path.join(os.path.dirname(model_save_path), 'training_curves.png')
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"训练曲线已保存至: {plot_path}")
    
    def load_classifier(self, model_path):
        """加载已训练的分类器模型，前提是预训练编码器已存在"""
        if self.pretrained_encoder is None:
            print("错误：加载分类器需要预训练编码器已存在")
            return False
        # 创建分类器实例（结构与训练时一致）
        model = MLPClassifier(self.pretrained_encoder, num_classes=2).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.classifier = model
        print(f"分类器已从 {model_path} 加载")
        return True


# ==================== 保存图嵌入 ====================
def save_graph_embeddings_batch(encoder, original_paths, variant_paths, save_path, device, batch_size=32):
    encoder.to(device).eval()
    embeddings = {'original': {}, 'variants': {}}
    
    def process_dict(path_dict, desc):
        items = list(path_dict.items())
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i+batch_size]
            batch_data = [torch.load(path) for _, path in batch_items]
            batch = Batch.from_data_list(batch_data).to(device)
            with torch.no_grad():
                _, graph_embs = encoder(batch.x, batch.edge_index, batch.batch)
            graph_embs_cpu = graph_embs.cpu().numpy()
            for j, (name, _) in enumerate(batch_items):
                target_dict[name] = graph_embs_cpu[j]
            del batch, graph_embs
            torch.cuda.empty_cache()
    
    # 处理原始图
    target_dict = embeddings['original']
    process_dict(original_paths, "Computing original embeddings")
    
    # 处理变种图
    for strategy, apk_dict in variant_paths.items():
        target_dict = embeddings['variants'][strategy] = {}
        process_dict(apk_dict, f"Computing {strategy} embeddings")
    
    with open(save_path, 'wb') as f:
        pickle.dump(embeddings, f)


# ==================== 测试函数 ====================
def evaluate_on_new_dataset(ea, new_dataset_path, new_working_dir, 
                            benign_count=None, malicious_per_category=None,
                            generate_variants=True, use_variants_for_test=False,
                            batch_size=16):
    """
    在新数据集上评估训练好的模型。
    ea: 已训练好的 ExpAndro 实例（包含 classifier 和 pretrained_encoder 等）
    new_dataset_path: 新数据集的根目录，应包含子目录（如 Benign/，Adware/等）或已整合的 benign/malicious
    new_working_dir: 新数据集的工作目录，用于存放处理后的数据（独立于训练数据）
    benign_count, malicious_per_category: 如果新数据集需要整合采样，指定数量
    generate_variants: 是否生成变种（用于测试变种鲁棒性）
    use_variants_for_test: 是否使用变种图进行测试（否则使用原始图）
    """
    print("\n" + "=" * 70)
    print("【在新测试集上评估模型】")
    print("=" * 70)

    # 确保新工作目录存在
    os.makedirs(new_working_dir, exist_ok=True)
    new_integrated_path = os.path.join(new_working_dir, "integrated_data")
    new_renamed_path = os.path.join(new_working_dir, "renamed_data")
    new_variant_path = os.path.join(new_working_dir, "variants") if generate_variants else None

    # 1. 如果需要整合数据集
    if benign_count is not None and malicious_per_category is not None:
        integrated_data_path = integrate_cic_maldroid_dataset(
            original_dataset_root=new_dataset_path,
            target_integrated_path=new_integrated_path,
            benign_count=benign_count,
            malicious_per_category=malicious_per_category,
            force_resample=False
        )
        original_path_for_rename = integrated_data_path
    else:
        # 假设 new_dataset_path 已经是 benign/malicious 结构
        original_path_for_rename = new_dataset_path

    # 2. Hash重命名
    working_data_path, hash_mapping = hash_rename_apks(
        original_path=original_path_for_rename,
        working_path=new_working_dir,
        random_seed=42  # 可固定
    )

    # 3. 生成变种（如果需要）
    if generate_variants:
        variant_generator = APKVariantGenerator(
            original_path=working_data_path,
            variant_path=new_variant_path,
            random_seed=42,
            use_fallback=True
        )
        variant_generator.generate_variants(skip_existing=True)

    # 4. 解析图数据
    # 确保 AST 模型已加载
    if ea._astmodel is None:
        ea._astmodel = ea._load_astmodel()
    original_paths_test, variant_paths_test = load_and_process_graph_data(
        ea_instance=ea,
        renamed_apk_path=working_data_path,
        variant_path=new_variant_path,
        hash_mapping=hash_mapping,
        force_reparse=False
    )

    # 5. 准备测试数据列表
    if use_variants_for_test and variant_paths_test:
        test_file_paths = []
        for strategy, apk_dict in variant_paths_test.items():
            test_file_paths.extend(apk_dict.values())
        print(f"\n使用变种图测试，共 {len(test_file_paths)} 个样本")
    else:
        test_file_paths = list(original_paths_test.values())
        print(f"\n使用原始图测试，共 {len(test_file_paths)} 个样本")

    if not test_file_paths:
        print(Fore.RED + "错误：没有可用的测试数据" + Style.RESET_ALL)
        return

    test_dataset = GraphFileDataset(test_file_paths)
    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 6. 使用训练好的模型进行预测
    device = ea.device
    model = ea.classifier
    if model is None:
        print(Fore.RED + "错误：ea.classifier 为空，请先训练模型" + Style.RESET_ALL)
        return
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    # 计算指标
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    precision = metrics.precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = metrics.recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = metrics.f1_score(all_labels, all_preds, average='binary', zero_division=0)
    conf_matrix = metrics.confusion_matrix(all_labels, all_preds)

    print("\n" + "-" * 50)
    print("评估结果：")
    print(f"  准确率 (Accuracy):  {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1 分数:            {f1:.4f}")
    print("  混淆矩阵:")
    print(conf_matrix)
    print("-" * 50)

    # 保存结果
    result_path = os.path.join(new_working_dir, "evaluation_results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'labels': all_labels
        }, f)
    print(f"评估结果已保存至: {result_path}")


def test_with_embeddings(model, test_data_list, device, batch_size=16):
    loader = GraphDataLoader(test_data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    acc = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds, average='binary', zero_division=0)
    return acc, f1


# ==================== 主流程 ====================
def split_dataset(integrated_path, train_root, test_root, train_ratio=0.8, random_seed=42):
    """
    将 integrated_path 下的 benign 和 malicious 文件夹中的 apk 文件按比例划分为训练集和测试集，
    并分别复制到 train_root 和 test_root 对应的子目录中。
    """
    random.seed(random_seed)
    for category in ["benign", "malicious"]:
        src_dir = os.path.join(integrated_path, category)
        if not os.path.exists(src_dir):
            continue
        apks = [f for f in glob.glob(os.path.join(src_dir, "*.apk")) if os.path.isfile(f)]
        random.shuffle(apks)
        split_idx = int(len(apks) * train_ratio)
        train_apks = apks[:split_idx]
        test_apks = apks[split_idx:]

        # 创建目标目录
        train_cat_dir = os.path.join(train_root, category)
        test_cat_dir = os.path.join(test_root, category)
        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(test_cat_dir, exist_ok=True)

        # 复制文件
        for apk in train_apks:
            shutil.copy(apk, os.path.join(train_cat_dir, os.path.basename(apk)))
        for apk in test_apks:
            shutil.copy(apk, os.path.join(test_cat_dir, os.path.basename(apk)))

        print(f"  {category}: 训练集 {len(train_apks)} 个, 测试集 {len(test_apks)} 个")
    return train_root, test_root


def test_full_pipeline():
    # ==================== 配置区域 ====================
    BASE_WORKING_DIR = "/home/gongjiacheng/LIHAN/"
    ORIGINAL_DATASET_ROOT = "/home/gongjiacheng/code/CICMalDroid2020/"
    
    INTEGRATED_PATH = os.path.join(BASE_WORKING_DIR, "train_apk_system", "integrated_data")
    TRAIN_ROOT = os.path.join(BASE_WORKING_DIR, "train_apk_system")
    TEST_ROOT = os.path.join(BASE_WORKING_DIR, "test_apk_system")
    
    WORKING_PATH = BASE_WORKING_DIR
    TRAIN_VARIANT_PATH = os.path.join(BASE_WORKING_DIR, "train_apk_system","variants")
    TEST_VARIANT_PATH = os.path.join(BASE_WORKING_DIR, "test_apk_system", "variants")
    
    RANDOM_SEED = 42
    FORCE_RETRAIN_CLASSIFIER = False
    # ===================================================

    # 0. 初始化随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    init(autoreset=True)

    print("\n" + "=" * 70)
    print("【安卓恶意软件检测】")
    print("=" * 70)
    """

    # 1. 整合数据集
    integrated_data_path = integrate_cic_maldroid_dataset(
        original_dataset_root=ORIGINAL_DATASET_ROOT,
        target_integrated_path=INTEGRATED_PATH,
        benign_count=None,          # 使用全部良性样本
        malicious_per_category=None, # 使用全部恶意样本（每个类别全部）
        force_resample=False
    )

    # 2. 检查是否已经划分好训练集和测试集
    if (os.path.exists(os.path.join(TRAIN_ROOT, "benign")) and os.listdir(os.path.join(TRAIN_ROOT, "benign")) and
        os.path.exists(os.path.join(TRAIN_ROOT, "malicious")) and os.listdir(os.path.join(TRAIN_ROOT, "malicious")) and
        os.path.exists(os.path.join(TEST_ROOT, "benign")) and os.listdir(os.path.join(TEST_ROOT, "benign")) and
        os.path.exists(os.path.join(TEST_ROOT, "malicious")) and os.listdir(os.path.join(TEST_ROOT, "malicious"))):
        print(Fore.GREEN + "训练集和测试集已按8:2划分，跳过划分步骤。" + Style.RESET_ALL)
        
    else:
        print("\n" + "=" * 60)
        print("【Step 1: 划分训练集和测试集 (80% / 20%)】")
        print("=" * 60)
        train_root, test_root = split_dataset(integrated_data_path, TRAIN_ROOT, TEST_ROOT, train_ratio=0.8, random_seed=RANDOM_SEED)
    """
    # 3. 对训练集和测试集分别进行 Hash 重命名
    print("\n" + "=" * 60)
    print("【Step 2: 对训练集和测试集分别进行 Hash 重命名】")
    print("=" * 60)
    train_renamed_path, train_hash_mapping = hash_rename_apks(
        original_path=TRAIN_ROOT,
        working_path=os.path.join(WORKING_PATH, "train_apk_system"),  # 专门用于训练集的工作目录
        random_seed=RANDOM_SEED
    )
    test_renamed_path, test_hash_mapping = hash_rename_apks(
        original_path=TEST_ROOT,
        working_path=os.path.join(WORKING_PATH, "test_apk_system"),   # 专门用于测试集的工作目录
        random_seed=RANDOM_SEED
    )
    """
    # 4. 对训练集生成变种
    print("\n" + "=" * 60)
    print("【Step 3: 对训练集生成混淆变种】")
    print("=" * 60)
    train_variant_generator = APKVariantGenerator(
        original_path=train_renamed_path,
        variant_path=TRAIN_VARIANT_PATH,
        random_seed=RANDOM_SEED,
        use_fallback=False
    )
    train_variant_generator.generate_variants(skip_existing=True)
    print("失败统计:", train_variant_generator.failed_apks)

    # 5. 对测试集生成变种
    print("\n" + "=" * 60)
    print("【Step 4: 对测试集生成混淆变种】")
    print("=" * 60)
    test_variant_generator = APKVariantGenerator(
        original_path=test_renamed_path,
        variant_path=TEST_VARIANT_PATH,
        random_seed=RANDOM_SEED,
        use_fallback=False
    )
    test_variant_generator.generate_variants(skip_existing=True)
    print("失败统计:", test_variant_generator.failed_apks)

    # 6. 准备 AST 语料库（使用训练集的所有 APK）
    train_benign_apks = glob.glob(os.path.join(train_renamed_path, "benign", "*.apk"))
    train_malicious_apks = glob.glob(os.path.join(train_renamed_path, "malicious", "*.apk"))
    all_train_apks = train_benign_apks + train_malicious_apks
    print(f"\n[语料库准备] 训练集 APK 总数: {len(all_train_apks)}")
    """

    # 7. 初始化主类并训练 Doc2Vec
    ea = ExpAndro(
        dataset="cic_2020",
        run_id="exp_v2",
        data_path=train_renamed_path,   # 训练集路径
        working_path=WORKING_PATH
    )
    print(ea.fname)
    if ea._astmodel is None:
        ea.ASTModel_generate(APKlist=all_train_apks, recollect=False)

    # 8. 解析训练集图数据 (原始 + 变种) 返回路径映射
    
    original_pkl = "/home/gongjiacheng/LIHAN/train_apk_system/variants/graph_data/original/original_paths.pkl"
    variant_pkl = "/home/gongjiacheng/LIHAN/train_apk_system/variants/graph_data/variants/variant_paths.pkl"

    if os.path.exists(original_pkl) and os.path.exists(variant_pkl):
        with open(original_pkl, 'rb') as f:
            train_original_paths = pickle.load(f)
        with open(variant_pkl, 'rb') as f:
            train_variant_paths = pickle.load(f)
        print("从分开的 pickle 文件加载图路径映射成功")
    else:
        train_original_paths, train_variant_paths = load_and_process_graph_data(
            ea_instance=ea,
            renamed_apk_path=train_renamed_path,
            variant_path=TRAIN_VARIANT_PATH,
            hash_mapping=train_hash_mapping,
            force_reparse=False
        )
    
    original_test_pkl = "/home/gongjiacheng/LIHAN/test_apk_system/variants/graph_data/original/original_paths.pkl"
    variant_test_pkl = "/home/gongjiacheng/LIHAN/test_apk_system/variants/graph_data/variants/variant_paths.pkl"
    if os.path.exists(original_test_pkl) and os.path.exists(variant_test_pkl):
        with open(original_test_pkl, 'rb') as f:
            test_original_paths = pickle.load(f)
        with open(variant_test_pkl, 'rb') as f:
            test_variant_paths = pickle.load(f)
        print("pickle 文件加载图路径映射成功")
    """
    # 9. 解析测试集图数据 (原始 + 变种) 返回路径映射
    test_original_paths, test_variant_paths = load_and_process_graph_data(
        ea_instance=ea,
        renamed_apk_path=test_renamed_path,
        variant_path=TEST_VARIANT_PATH,
        hash_mapping=test_hash_mapping,
        force_reparse=False
    )
    """

    # 10. 构建训练集的多变种数据集（用于对比学习预训练）
    mv_dataset = MultiVariantDataset(train_original_paths, train_variant_paths,min_required_pairs=28)

    # 11. 准备分类器训练数据（使用训练集的原始图）
    train_file_paths = list(train_original_paths.values())
    random.shuffle(train_file_paths)
    # 由于我们已经划分了训练集和测试集，不再从训练集中拆分验证集，但为保持兼容，仍取一部分作为验证集
    val_size = int(0.2 * len(train_file_paths))
    val_file_paths = train_file_paths[:val_size]
    train_file_paths = train_file_paths[val_size:]
    train_dataset_cls = GraphFileDataset(train_file_paths)
    val_dataset_cls = GraphFileDataset(val_file_paths)
    print(f"\n[分类器数据集] 训练: {len(train_dataset_cls)}, 验证: {len(val_dataset_cls)}")

    # 12. 对比学习预训练
    print("\n" + "=" * 60)
    print("【Step 5: 对比学习预训练】")
    print("=" * 60)
    seven_strategies = [
        'ConstStringEncryption'
    ]

    allowed_pairs = [(None, s) for s in seven_strategies]   
    filter_func = make_pair_filter('specific', allowed_strategies=allowed_pairs)
    # 从任意一个原始图中获取特征维度

    sample_data = torch.load(train_file_paths[0])
    input_dim = sample_data.x.shape[1]
    encoder = ModelAP(
        input_dim=input_dim,
        hidden_channels=ea.hidden_dim,
        dropout=0.3,
        keep_ratio=0.7,
        proj_dim=128
    ).to(ea.device)
    
    pair_dataset = PairDataset(
        mv_dataset, 
        train_original_paths, 
        num_pairs=7, 
        pair_filter=filter_func   # 您定义的过滤函数
    )

    # 构建 DataLoader
    train_loader = DataLoader(
        pair_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=12,
        collate_fn=collate_pairs,
        pin_memory=True,
        persistent_workers=True
    )
    total_pairs = len(pair_dataset)
    val_size = min(2000, int(0.1 * total_pairs))  # 取 10% 或固定 2000 对作为验证
    indices = random.sample(range(total_pairs), val_size)
    val_pair_dataset = Subset(pair_dataset, indices)

    # 创建 DataLoader（注意 collate_fn 与训练时相同）
    eval_pair_loader = DataLoader(
        val_pair_dataset,
        batch_size=64,          # 可与训练 batch_size 一致
        shuffle=False,
        collate_fn=collate_pairs,
        num_workers=12,
        pin_memory=True
    )
    pretrained_path =  './var_enc.pth' if len(mv_dataset.valid_apks) > 0 else './graph_aug_enc.pth'
    if os.path.exists(pretrained_path):
        print(Fore.GREEN + f"[加载已有预训练编码器] {pretrained_path}" + Style.RESET_ALL)
        encoder.load_state_dict(torch.load(pretrained_path, map_location=ea.device, weights_only=True))
    else:
        # 在 Step 5 附近修改
        if len(mv_dataset.valid_apks) > 0:
            print("[选择] 对比学习")
            encoder = variant_contrast_pretrain(
            encoder=encoder,
            train_loader=train_loader,
            device=ea.device,
            epochs=150,
            lr=0.0001,
            accum_steps=1,
            temperature=0.2,
            eval_loader=eval_pair_loader,
            save_path='./var_enc.pth'
        )
        else:
            print("[选择] 图增强对比学习")
            encoder = graph_augment_pretrain(
                encoder, train_dataset_cls, ea.device, epochs=100, batch_size=32
            )
    ea.pretrained_encoder = encoder
    
    """
    print("\n可视化训练集嵌入...")
    visualize_embeddings(encoder, train_dataset_cls, ea.device, sample_size=2000, method='umap', save_path='train_embeddings_umap.png')

    print("\n可视化验证集嵌入...")
    visualize_embeddings(encoder, val_dataset_cls, ea.device, sample_size=1000, method='umap', save_path='val_embeddings_umap.png')
    """
    # 13. 训练分类器
    print("\n" + "=" * 60)
    print("【Step 6: 训练分类器】")
    print("=" * 60)
    classifier_save_path = os.path.join(ea._relatedfile_path, 'best_model.pt')
    if not FORCE_RETRAIN_CLASSIFIER and os.path.exists(classifier_save_path):
        print(f"[加载已有分类器] {classifier_save_path}")
        ea.load_classifier(classifier_save_path)
    else:
        print("[开始训练分类器]")
        ea.Model_train_test(
            train_dataset=train_dataset_cls,
            test_dataset=val_dataset_cls,   # 使用训练集中的一部分作为验证集
            hidden_dim=ea.hidden_dim,
            lr=0.0001,
            epochs=300,
            use_pretrained=True,
            freeze_encoder=True,
            plot_curves=True
        )

    # 14. 在测试集上评估
    print("\n" + "=" * 60)
    print("【Step 7: 在测试集上评估】")
    print("=" * 60)
    test_file_paths = list(test_original_paths.values())
    print(f"测试集原始图数量: {len(test_file_paths)}")
    if test_file_paths:
        test_dataset = GraphFileDataset(test_file_paths)
        test_loader = GraphDataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        device = ea.device
        model = ea.classifier
        model.to(device)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        # ---------- 增加 precision 和 recall ----------
        acc = metrics.accuracy_score(all_labels, all_preds)
        precision = metrics.precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = metrics.recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = metrics.f1_score(all_labels, all_preds, average='binary', zero_division=0)
        print(f"测试集评估结果 (原始图) - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    else:
        print("测试集无数据，跳过评估")
    if test_variant_paths:
        print("\n[可选] 在测试集变种图上评估")
        for strategy, apk_dict in test_variant_paths.items():
            variant_paths_list = list(apk_dict.values())
            if not variant_paths_list:
                continue
            variant_dataset = GraphFileDataset(variant_paths_list)
            test_loader = GraphDataLoader(variant_dataset, batch_size=16, shuffle=False, num_workers=0)
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    pred = out.argmax(dim=1)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
            # ---------- 增加 precision 和 recall ----------
            acc = metrics.accuracy_score(all_labels, all_preds)
            precision = metrics.precision_score(all_labels, all_preds, average='binary', zero_division=0)
            recall = metrics.recall_score(all_labels, all_preds, average='binary', zero_division=0)
            f1 = metrics.f1_score(all_labels, all_preds, average='binary', zero_division=0)
            print(f"  策略 {strategy:20s} 变种图 - Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
    print("\n" + "=" * 70)
    print("【All Done】")
    print("=" * 70)


if __name__ == "__main__":
    test_full_pipeline()
