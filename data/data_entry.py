from torch.utils.data import DataLoader, Subset
from data.list_dataset import Dataset

from data.list_dataset import ContrastiveDataset
#from list_dataset import PDBbindDataset
from torch.utils.data import DataLoader
import dgl
import numpy as np
import torch
from options import prepare_train_args


import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path


def collate_contrastive(data):
    smiles, seqs, lig_graphs, prot_graphs, pdbids = map(list, zip(*data))
    smiles_batch = torch.stack(smiles, dim=0)
    seqs_batch = torch.stack(seqs, dim=0)
    bgl = dgl.batch(lig_graphs)
    bgp = dgl.batch(prot_graphs)
    # 为所有节点和边设置零初始化器
    for nty in bgl.ntypes:
        bgl.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bgl.canonical_etypes:
        bgl.set_e_initializer(dgl.init.zero_initializer, etype=ety)

    for nty in bgp.ntypes:
        bgp.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bgp.canonical_etypes:
        bgp.set_e_initializer(dgl.init.zero_initializer, etype=ety)

    return pdbids, smiles_batch, seqs_batch, bgl, bgp

def select_loader(args):
    """选择训练数据加载器"""
    # 直接加载所有图数据
    data_dir = Path(args.data_dir)
#----------------------------------------------------------
    print("=" * 60)
    print("加载数据集...")

    # 获取设备信息
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
#------------------------------------------------------------
    # 加载数据
    dataset = ContrastiveDataset(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
        max_smi_len=args.max_smi_len
    )

    # 划分训练集和验证集
    train_inds, val_inds = dataset.train_and_test_split(
        valnum=args.val_num if hasattr(args, 'val_num') else 20000,
        seed=args.seed
    )

    print(f"训练集: {len(train_inds)} 个样本")
    print(f"验证集: {len(val_inds)} 个样本")

    # 创建子集
    train_dataset = Subset(dataset, train_inds)
    val_dataset = Subset(dataset, val_inds)


    # # 创建数据加载器
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=getattr(args, 'num_workers', 8),
    #     collate_fn=collate_contrastive,
    #     drop_last=True
    # )
    #
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=getattr(args, 'num_workers', 8),
    #     collate_fn=collate_contrastive,
    #     drop_last=True
    # )
    #
    # return train_loader, val_loader, train_inds, val_inds

    # 创建数据加载器 - 使用partial固定device参数
    from functools import partial
    # train_collate = partial(collate_contrastive, device=device)
    # val_collate = partial(collate_contrastive, device=device)
    train_collate = partial(collate_contrastive)
    val_collate = partial(collate_contrastive)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=getattr(args, 'num_workers', 8),
        num_workers=getattr(args, 'num_workers', 2),
        collate_fn=train_collate,
        drop_last=True,
        pin_memory=False  # 启用锁页内存加速传输
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=getattr(args, 'num_workers', 8),
        num_workers=getattr(args, 'num_workers', 2),
        collate_fn=val_collate,
        drop_last=True,
        pin_memory=False
    )




#     def select_loader(args):
#         """低内存版本 DataLoader"""
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#         dataset = ContrastiveDataset(
#             data_dir=args.data_dir,
#             max_seq_len=args.max_seq_len,
#             max_smi_len=args.max_smi_len
#         )
#
#         train_inds, val_inds = dataset.train_and_test_split(
#             valnum=args.val_num if hasattr(args, 'val_num') else 20000,
#             seed=getattr(args, 'seed', 42)
#         )
#
#         train_dataset = Subset(dataset, train_inds)
#         val_dataset = Subset(dataset, val_inds)
#         from functools import partial
#
#         collate_fn_train = partial(collate_contrastive, device=device)
#         collate_fn_val = partial(collate_contrastive, device=device)
#
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=getattr(args, 'batch_size', 2),
#             shuffle=True,
#             num_workers=getattr(args, 'num_workers', 2),
#             collate_fn=collate_fn_train,
#             drop_last=True,
#             pin_memory=False
#         )
#
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=getattr(args, 'batch_size', 2),
#             shuffle=False,
#             num_workers=getattr(args, 'num_workers', 2),
#             collate_fn=collate_fn_val,
#             drop_last=True,
#             pin_memory=False
#         )
    print("=" * 60)
    return train_loader, val_loader, train_inds, val_inds

