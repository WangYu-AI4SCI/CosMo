import torch as th
import torch.nn.functional as F
import dgl
import numpy as np
import random
import dgl.function as fn
from torch import nn
import pandas as pd
import torch

from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.model_zoo.attentivefp_predictor import AttentiveFPPredictor
from dgllife.model.model_zoo.mgcn_predictor import MGCNPredictor
from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor
from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
from dgllife.model.gnn.wln import WLN
from dgllife.model.model_zoo.gat_predictor import GATPredictor


# class ECIF_GNN(nn.Module):
# 	def __init__(self,dropout):#,in_feats,hidden_size,dropout):
# 		super(ECIF_GNN, self).__init__()
# 		self.dropout = dropout
# 		self.lig_model = AttentiveFPPredictor(node_feat_size=41,edge_feat_size=10,n_tasks=64,num_layers=3,graph_feat_size=200,dropout=0,num_timesteps=4)
# 		self.prot_model = AttentiveFPPredictor(node_feat_size=41, edge_feat_size=5,n_tasks=64, num_layers=3,graph_feat_size=200,dropout=0,num_timesteps=4)
# 		#self.lig_model =GATPredictor(in_feats=41, hidden_feats=None, num_heads=None, feat_drops=None, attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None, biases=None, classifier_hidden_feats=128, classifier_dropout=0.0, n_tasks=64, predictor_hidden_feats=128, predictor_dropout=0.0)
# 		#self.prot_model =GATPredictor(in_feats=41, hidden_feats=None, num_heads=None, feat_drops=None, attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None, biases=None, classifier_hidden_feats=128, classifier_dropout=0.0, n_tasks=64, predictor_hidden_feats=128, predictor_dropout=0.0)
# 		#self.lig_model = DGLGraphTransformer(in_channels=41, edge_features=6, num_hidden_channels=64,activ_fn=th.nn.SiLU(),transformer_residual=True,num_attention_heads=4,norm_to_apply='batch',dropout_rate=0.15,num_layers=6)
# 		#self.prot_model = DGLGraphTransformer(in_channels=41, edge_features=5, num_hidden_channels=64,activ_fn=th.nn.SiLU(),transformer_residual=True,num_attention_heads=4,norm_to_apply='batch',dropout_rate=0.15,num_layers=6)
# 		self.MLP= nn.Sequential(#nn.Dropout(self.dropout),
#                                     #nn.BatchNorm1d(256),
#                                      #nn.LeakyReLU(),
#                                      #nn.Dropout(0.1),
#                                      #nn.Linear(in_features=256, out_features=128, bias=True),
#                                      #nn.BatchNorm1d(128),
#                                      nn.Linear(in_features=128, out_features=64, bias=True),nn.ReLU())
# 		self.MLP_2 = nn.Sequential(nn.Linear(in_features=128, out_features=64),#, bias=True),
#                                      nn.BatchNorm1d(64),
#                                      nn.ReLU(),
#                                      nn.Linear(in_features=64, out_features=32),#, bias=True),
#                                      #nn.BatchNorm1d(32),
#                                      nn.ReLU(),
#                                     # nn.Linear(in_features=32, out_features=16),#, bias=True),
#                                      #nn.BatchNorm1d(32),
#                                      #nn.ReLU(),
#                                      #nn.Dropout(self.dropout),
#                                      nn.Linear(in_features=32, out_features=1))#,nn.ReLU())#, bias=True))
#
#
# 	def forward(self,bgl_0 ,bgp_0,bgl_1 ,bgp_1):
# 		#print('atom_ndata:',bgl_0.ndata['atom'].float().size())
# 		#print('edata_bond:',bgl_0.edata['bond'].float().size())
# 		#print('ndata_feats:',bgp_0.ndata['feats'].float().size())
# 		#print('edata_feats:',bgp_0.edata['feats'].float().size())
# 		h_l_0 = self.lig_model(bgl_0,bgl_0.ndata['atom'].float(), bgl_0.edata['bond'].float())
# 		h_p_0 = self.prot_model(bgp_0,bgp_0.ndata['feats'].float(), bgp_0.edata['feats'].float())
# 		# 只返回一个2
#
#
#
# 		#print('atom_ndata:',bgl.ndata['atom'].float().size())
# 		#print('edata_bond:',bgl.edata['bond'].float().size())
# 		#print('ndata_feats:',bgp.ndata['feats'].float().size())
# 		#print('edata_feats:',bgp.edata['feats'].float().size())
# 		#print('h_l:',h_l.size())
# 		#print('h_p:',h_p.size())
# 		h_l_1 = self.lig_model(bgl_1,bgl_1.ndata['atom'].float(), bgl_1.edata['bond'].float())
# 		h_p_1 = self.prot_model(bgp_1,bgp_1.ndata['feats'].float(), bgp_1.edata['feats'].float())
# 		output_0 = torch.cat([h_l_0,h_p_0],axis=1)
# 		output_1 = torch.cat([h_l_1,h_p_1],axis=1)
# 		#print("output_1:",output.size())
# 		output_2 = self.MLP(output_0)
# 		output_3 = self.MLP(output_1)
# 		#output = self.MLP(PLEC)
# 		#print("output_2:",output.size())
# 		regression = self.MLP_2(output_0)
# 		return F.normalize(output_2, dim=1),F.normalize(output_3, dim=1),output_0 ,output_1,regression
# 		#return output_2,output_3,output_0 ,output_1,regression


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange, Reduce
from dgllife.model.model_zoo.attentivefp_predictor import AttentiveFPPredictor


class ResDilaCNNBlock(nn.Module):
	def __init__(self, dilaSize, filterSize=256, dropout=0.15, name='ResDilaCNNBlock'):
		super(ResDilaCNNBlock, self).__init__()
		self.layers = nn.Sequential(
			nn.ReLU(),
			nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
			nn.ReLU(),
			nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
		)

	def forward(self, x):
		# x: batchSize × filterSize × seqLen
		return x + self.layers(x)


class ResDilaCNNBlocks(nn.Module):
	def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.5,
				 name='ResDilaCNNBlocks'):
		super(ResDilaCNNBlocks, self).__init__()
		self.blockLayers = nn.Sequential()
		self.linear = nn.Linear(feaSize, filterSize)
		for i in range(blockNum):
			self.blockLayers.add_module(f"ResDilaCNNBlock{i}",
										ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize,
														dropout=dropout))
		self.name = name
		self.act = nn.ReLU()

	def forward(self, x):
		# x: batchSize × seqLen × feaSize
		x = self.linear(x)  # => batchSize × seqLen × filterSize
		x = self.blockLayers(x.transpose(1, 2))  # => batchSize × filterSize × seqLen
		x = self.act(x)  # => batchSize × filterSize × seqLen
		x = Reduce('b c t -> b c', 'max')(x)  # => batchSize × filterSize
		return x


class MultiViewNet(nn.Module):
	"""
    简化的对比学习模型：
    - 输入：smi, seq, lig_graph, prot_graph
    - 输出：两个pair用于对比学习：
      1. 序列视图：smi特征 + seq特征
      2. 图视图：lig_graph特征 + prot_graph特征
    """

	def __init__(self, embed_dim=256):
		super(MultiViewNet, self).__init__()

		# 序列特征编码器
		self.embed_smile = nn.Embedding(65, embed_dim)  # SMILES编码
		self.embed_prot = nn.Embedding(26, embed_dim)  # 蛋白质序列编码

		# 序列特征提取网络
		self.onehot_smi_net= ResDilaCNNBlocks(embed_dim, embed_dim, name='smi_encoder')
		self.onehot_prot_net = ResDilaCNNBlocks(embed_dim, embed_dim, name='seq_encoder')

		# 图特征编码器
		self.lig_model = AttentiveFPPredictor(
			node_feat_size=41, edge_feat_size=10,
			n_tasks=64, num_layers=3, graph_feat_size=200,
			dropout=0, num_timesteps=4
		)

		self.prot_model = AttentiveFPPredictor(
			node_feat_size=41, edge_feat_size=5,
			n_tasks=64, num_layers=3, graph_feat_size=200,
			dropout=0, num_timesteps=4
		)

		# 投影层：将不同维度特征映射到统一维度
		self.projection_lig_gnn = nn.Sequential(
			nn.LayerNorm(64),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, embed_dim)
		)

		self.projection_prot_gnn = nn.Sequential(
			nn.LayerNorm(64),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Linear(128, embed_dim)
		)

		# 序列视图融合层（拼接smi和seq特征）
		self.seq_view_fusion = nn.Sequential(
			nn.Linear(embed_dim * 2, embed_dim * 2),
			nn.ReLU(),
			nn.Linear(embed_dim * 2, embed_dim)
		)

		# 图视图融合层（拼接lig和prot特征）
		self.graph_view_fusion = nn.Sequential(
			nn.Linear(embed_dim * 2, embed_dim * 2),
			nn.ReLU(),
			nn.Linear(embed_dim * 2, embed_dim)
		)

		# 对比学习投影头
		# self.seq_contrastive_proj = nn.Sequential(
		# 	nn.Linear(embed_dim, embed_dim),
		# 	nn.ReLU(),
		# 	nn.Linear(embed_dim, embed_dim)
		# )
		#
		# self.graph_contrastive_proj = nn.Sequential(
		# 	nn.Linear(embed_dim, embed_dim),
		# 	nn.ReLU(),
		# 	nn.Linear(embed_dim, embed_dim)
		# )
		self.seq_contrastive_proj = nn.Sequential(
			#nn.Linear(embed_dim, 128),
			nn.Linear(embed_dim*2, 128),
			nn.ReLU(),
			nn.Linear(128, 64)   #clgnn里面最后维度降到64
		)
		self.graph_contrastive_proj = nn.Sequential(
			# nn.Linear(embed_dim, 128),
			nn.Linear(embed_dim * 2, 128),
			nn.ReLU(),
			nn.Linear(128, 64)
		)



	def forward(self, smi, seq, lig_graph, prot_graph):
		"""
        参数:
            smi: [batch_size, max_smi_len] - SMILES序列
            seq: [batch_size, max_seq_len] - 蛋白质序列
            lig_graph: DGLGraph - 配体图
            prot_graph: DGLGraph - 蛋白质图

        返回:
            seq_view: [batch_size, embed_dim] - 序列视图特征
            graph_view: [batch_size, embed_dim] - 图视图特征
        """
		batch_size = smi.size(0)

		# ====== 1. 处理序列特征 ======
		# SMILES编码
		smile_vectors_onehot = self.embed_smile(smi)  # [batch, max_smi_len, embed_dim]
		compound_seq_feat = self.onehot_smi_net(smile_vectors_onehot) # [batch, embed_dim]
	    # 蛋白质序列编码
		proteinFeature_onehot = self.embed_prot(seq)  # [batch, max_seq_len, embed_dim]
		protein_seq_feat = self.onehot_prot_net(proteinFeature_onehot)  # [batch, embed_dim]

		# 拼接序列特征
		# seq_combined = torch.cat([compound_seq_feat, protein_seq_feat], dim=1)  # [batch, embed_dim*2]
		# seq_view_raw = self.seq_view_fusion(seq_combined)  # [batch, embed_dim]
		seq_view_raw = torch.cat([compound_seq_feat, protein_seq_feat], dim=1)  # [batch, embed_dim*2]
		# ====== 2. 处理图特征 ======
		# 检查图数据是否有需要的特征
		if 'atom' not in lig_graph.ndata:
			raise ValueError("lig_graph缺少'atom'节点特征")
		if 'bond' not in lig_graph.edata:
			raise ValueError("lig_graph缺少'bond'边特征")
		if 'feats' not in prot_graph.ndata:
			raise ValueError("prot_graph缺少'feats'节点特征")
		if 'feats' not in prot_graph.edata:
			raise ValueError("prot_graph缺少'feats'边特征")
		# 配体图特征
		lig_graph_feat = self.lig_model(
			lig_graph,
			lig_graph.ndata['atom'].float(),
			lig_graph.edata['bond'].float()
		)  # [batch, 64]
		# 投影GNN特征到256维
		lig_graph_feat_proj = self.projection_lig_gnn(lig_graph_feat)  # [batch, embed_dim]

		# 蛋白质图特征
		prot_graph_feat = self.prot_model(
			prot_graph,
			prot_graph.ndata['feats'].float(),
			prot_graph.edata['feats'].float()
		)  # [batch, 64]
		#投影GNN特征到256维
		prot_graph_feat_proj = self.projection_prot_gnn(prot_graph_feat)  # [batch, embed_dim]

		# 拼接图特征
		# graph_combined = torch.cat([lig_graph_feat_proj, prot_graph_feat_proj], dim=1)  # [batch, embed_dim*2]
		# graph_view_raw = self.graph_view_fusion(graph_combined)  # [batch, embed_dim]
		graph_view_raw = torch.cat([lig_graph_feat_proj, prot_graph_feat_proj], dim=1)  # [batch, embed_dim*2]

		# ====== 3. 对比学习投影 ======
		seq_view = self.seq_contrastive_proj(seq_view_raw)  # [batch, embed_dim]->[batch, 64]
		graph_view = self.graph_contrastive_proj(graph_view_raw)  # [batch, embed_dim]->[batch, 64]

		# ====== 4. 归一化（对比学习常用） ======
		seq_view = F.normalize(seq_view, dim=1)
		graph_view = F.normalize(graph_view, dim=1)

		# 返回两个视图的特征
		return seq_view, graph_view


