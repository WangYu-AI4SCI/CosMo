from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import dgl
from dgl.data.utils import load_graphs

# ========== 编码函数（只定义一次）==========

# 将SMILES符号映射到整数
CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

# 将氨基酸序列映射到整数
CHARPROTSET = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
               "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12,
               "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18,
               "W": 19, "Y": 20, "X": 21}


def label_sequence(line, MAX_SEQ_LEN):
    """编码蛋白质序列"""
    X = np.zeros(MAX_SEQ_LEN, dtype=int)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET.get(ch, 21)  # 使用21表示未知氨基酸
    return X


def label_smiles(line, max_smi_len):
    """编码SMILES序列"""
    X = np.zeros(max_smi_len, dtype=int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET.get(ch, 0)  # 使用0表示未知字符
    return X


# ========== 数据集类 ==========

class ContrastiveDataset(Dataset):
    """
    对比学习数据集，包含四种数据：
    1. SMILES序列
    2. 蛋白质序列
    3. 配体图
    4. 蛋白质图
    """

    def __init__(self, data_dir, max_seq_len=1000, max_smi_len=120):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len

        print("=" * 60)
        print("加载对比学习数据集")
        print("=" * 60)

        # 1. 加载ID文件（FULL_ID）
        ids_file = self.data_dir / "filtered_out_id_pre_train.npy"
        print(f"加载ID文件: {ids_file}")
        self.pdbids = np.load(ids_file, allow_pickle=True)
        self.pdbids = [str(pid).strip() for pid in self.pdbids]
        print(f"加载 {len(self.pdbids)} 个FULL_ID")

        # 2. 加载SMILES数据
        smiles_file = self.data_dir / "filtered_ccd_smiles.tsv"
        print(f"加载SMILES文件: {smiles_file}")
        smi_df = pd.read_csv(smiles_file, sep='\t')
        # 使用FULL_ID作为键
        self.smiles_dict = {}
        for _, row in smi_df.iterrows():
            full_id = str(row['FULL_ID']).strip()
            self.smiles_dict[full_id] = row['SMILES']
        print(f"加载 {len(self.smiles_dict)} 条SMILES记录")

        # 3. 加载蛋白质序列数据
        seq_file = self.data_dir / "ordered_pdb_ccd_sequence.tsv"
        print(f"加载序列文件: {seq_file}")
        seq_df = pd.read_csv(seq_file, sep='\t')
        # 使用FULL_ID作为键
        self.seq_dict = {}
        for _, row in seq_df.iterrows():
            full_id = str(row['FULL_ID']).strip()
            self.seq_dict[full_id] = row['SEQUENCE']
        print(f"加载 {len(self.seq_dict)} 条序列记录")

        # 4. 加载图数据
        print("加载图数据...")

        # 配体图
        lig_file = self.data_dir / "filtered_out_ligand_pre_train.bin"
        print(f"加载配体图: {lig_file}")
        self.lig_graphs, _ = load_graphs(str(lig_file))
        self.lig_graphs = list(self.lig_graphs)
        print(f"加载 {len(self.lig_graphs)} 个配体图")

        # 蛋白质图
        prot_file = self.data_dir / "filtered_out_protein_pre_train.bin"
        print(f"加载蛋白质图: {prot_file}")
        self.prot_graphs, _ = load_graphs(str(prot_file))
        self.prot_graphs = list(self.prot_graphs)
        print(f"加载 {len(self.prot_graphs)} 个蛋白质图")

        # 5. 构建有效样本索引
        self.valid_indices = []
        for idx, full_id in enumerate(self.pdbids):
            if (full_id in self.smiles_dict and
                    full_id in self.seq_dict and
                    idx < len(self.lig_graphs) and
                    idx < len(self.prot_graphs)):
                self.valid_indices.append(idx)

        print(f"\n有效样本数: {len(self.valid_indices)}/{len(self.pdbids)} "
              f"({len(self.valid_indices) / len(self.pdbids) * 100:.1f}%)")

        # 显示前几个样本的信息
        print("\n前5个样本信息:")
        for i in range(min(5, len(self.valid_indices))):
            idx = self.valid_indices[i]
            full_id = self.pdbids[idx]
            print(f"  样本 {i}: {full_id}")
            print(f"    SMILES: {self.smiles_dict[full_id][:50]}...")
            print(f"    序列: {self.seq_dict[full_id][:50]}...")

    def __len__(self):
        """返回数据集大小"""
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取有效索引对应的原始索引
        real_idx = self.valid_indices[idx]
        full_id = self.pdbids[real_idx]

        # 获取SMILES和序列
        smile = self.smiles_dict[full_id]
        seq = self.seq_dict[full_id]

        # 编码序列（使用当前文件中定义的函数）
        smile_encoded = label_smiles(smile, self.max_smi_len)
        seq_encoded = label_sequence(seq, self.max_seq_len)

        # 获取图数据
        lig_graph = self.lig_graphs[real_idx]
        prot_graph = self.prot_graphs[real_idx]

        # 确保返回顺序正确
        return (
            torch.tensor(smile_encoded, dtype=torch.long),  # [max_smi_len]
            torch.tensor(seq_encoded, dtype=torch.long),  # [max_seq_len]
            lig_graph,  # DGLGraph
            prot_graph,  # DGLGraph
            full_id  # FULL_ID
        )

    def train_and_test_split(self, valnum=20000, seed=1234):
        """划分训练集和验证集"""
        np.random.seed(seed)

        total_size = len(self.valid_indices)

        print(f"\n数据集划分:")
        print(f"  总样本数: {total_size}")
        print(f"  验证集大小: {valnum}")

        # 如果验证集大小超过总样本数，调整为20%
        if valnum >= total_size:
            valnum = int(total_size * 0.2)
            print(f"  调整验证集大小为: {valnum} (20%)")

        # 随机选择验证集索引
        all_indices = np.arange(total_size)
        val_inds = np.random.choice(all_indices, valnum, replace=False)
        train_inds = np.setdiff1d(all_indices, val_inds)

        print(f"  训练集: {len(train_inds)} 个样本")
        print(f"  验证集: {len(val_inds)} 个样本")

        return train_inds, val_inds
