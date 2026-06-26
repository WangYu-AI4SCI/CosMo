# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import csv
import time
import pandas as pd
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable

import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
# from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import argparse

from data.data_entry import select_loader
from model.model_entry import select_model
from options import prepare_train_args
from utils.logger import Logger
from utils.torch_utils import load_match_dict

def nt_xent_loss(out_1, out_2, temperature, eps=1e-5):
    # out = torch.cat([out_1, out_2], dim=0)
    print('out_1', out_1.size())
    print('out_2', out_2.size())
    cov = torch.mm(out_1, out_2.t().contiguous())
    sim = torch.exp(cov / temperature)
    neg = sim.sum(dim=-1)

    row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
    neg = torch.clamp(neg - row_sub, min=eps)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # pos = torch.cat([pos, pos], dim=0)

    return -torch.log(pos / (neg + eps)).mean()


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class ContrastiveTrainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)
        self.logger = Logger(args)

        print("=" * 60)
        print("训练参数配置")
        print("=" * 60)
        print(f"model_type: {args.model_type}")
        print(f"data_dir: {args.data_dir}")
        print(f"batch_size: {args.batch_size}")
        print(f"embed_dim: {args.embed_dim}")
        print(f"temperature: {args.temperature}")
        print(f"val_num: {args.val_num}")
        print(f"max_seq_len: {args.max_seq_len}")
        print(f"max_smi_len: {args.max_smi_len}")
        print("=" * 60)

#------------------设置设备
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 验证CUDA
        if torch.cuda.is_available():
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前设备: {torch.cuda.current_device()}")
            print(f"设备名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("警告: CUDA不可用，将在CPU上训练")
#--------------------------
        # 获取数据加载器 - 现在返回4个值
        print("加载数据集...")
        self.train_loader, self.val_loader, self.train_inds, self.val_inds = select_loader(args)

        print(f"训练集样本数: {len(self.train_inds)}")
        print(f"验证集样本数: {len(self.val_inds)}")

        # 选择模型
        print("初始化模型...")
        self.model = select_model(args)
        print(f"模型类型: {args.model_type}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters())}")

        # 设置对比学习损失
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f"使用设备: {self.device}")

        self.nt_xent_criterion = NTXentLoss(
            self.device,
            batch_size=args.batch_size,
            temperature=args.temperature,
            use_cosine_similarity=True
        )

        # 加载预训练模型（如果有）
        if args.load_model_path != '' and args.load_model_path is not None:
            print(f"加载预训练模型: {args.load_model_path}")
            if args.load_not_strict:
                load_match_dict(self.model, args.load_model_path)
            else:
                self.model.load_state_dict(torch.load(args.load_model_path).state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            args.lr,
            betas=(args.momentum, args.beta),
            weight_decay=args.weight_decay
        )

        print(f"优化器: Adam, lr={args.lr}, weight_decay={args.weight_decay}")

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs
        )
###断点恢复###############################
        # # 在优化器设置之后添加恢复训练逻辑
        # if resume_training and os.path.exists(checkpoint_path):
        #     print(f"🔄 从检查点恢复训练: {checkpoint_path}")
        #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
        #
        #     # 加载模型状态
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #
        #     # 加载优化器状态
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #
        #     # 设置起始epoch
        #     self.start_epoch = checkpoint['epoch'] + 1
        #     print(f"恢复训练: 从 epoch {self.start_epoch} 开始")
        # else:
        #     self.start_epoch = 0
        #     print(f"从头开始训练")
############################################
        # 将模型移动到设备
        self.model.to(self.device)
        print(f"模型已移动到 {self.device}")

        # 验证模型参数设备
        model_device = next(self.model.parameters()).device
        print(f"模型参数设备: {model_device}")

        if model_device != self.device:
            print(f"警告: 模型不在目标设备上！重新移动...")
            self.model.to(self.device)

    def train_per_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        print(f"\n开始第 {epoch} 个epoch的训练...")

        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # 解包数据
            pdbids, smiles, seqs, bgl, bgp = data

            # 移动到设备
            smiles = smiles.to(self.device)
            seqs = seqs.to(self.device)
            bgl = bgl.to(self.device)
            bgp = bgp.to(self.device)

            # # 检查输入数据
            # if i == 0 and epoch == 0:
            #     print(f"第一批数据形状:")
            #     print(f"  smiles: {smiles.shape}")
            #     print(f"  seqs: {seqs.shape}")
            #     print(f"  bgl nodes: {bgl.number_of_nodes()}, edges: {bgl.number_of_edges()}")
            #     print(f"  bgp nodes: {bgp.number_of_nodes()}, edges: {bgp.number_of_edges()}")
#----------------------------------------
            # ====== 验证数据设备 ======
            if i == 0 and epoch == 0:
                print(f"\n第一批数据设备验证:")
                print(f"  smiles设备: {smiles.device}")
                print(f"  seqs设备: {seqs.device}")
                print(f"  bgl设备: {bgl.device if hasattr(bgl, 'device') else '无device属性'}")
                print(f"  bgp设备: {bgp.device if hasattr(bgp, 'device') else '无device属性'}")
                print(f"  模型设备: {next(self.model.parameters()).device}")

                # 检查所有数据是否在同一设备
                data_devices = [smiles.device, seqs.device]
                if hasattr(bgl, 'device'):
                    data_devices.append(bgl.device)
                if hasattr(bgp, 'device'):
                    data_devices.append(bgp.device)

                all_same = all(d == self.device for d in data_devices)
                if all_same:
                    print(f"  ✅ 所有数据都在 {self.device} 上")
                else:
                    wrong_items = [(name, d) for name, d in zip(['smiles', 'seqs', 'bgl', 'bgp'], data_devices)
                                   if d != self.device]
                    print(f"  ❌ 以下数据不在 {self.device} 上: {wrong_items}")

            # 检查数据设备（简化版）
            if i == 0:
                print(f"Batch {i}: 数据设备 - smiles:{smiles.device}, "
                      f"seqs:{seqs.device}, bgl:{bgl.device if hasattr(bgl, 'device') else 'CPU'}")
#--------------------------------
            # 前向传播：获取两个视图的特征
            seq_view, graph_view = self.model(smiles, seqs, bgl, bgp)
            if i == 0 and epoch == 0:
                print(f"输出特征形状:")
                print(f"  seq_view: {seq_view.shape} (设备: {seq_view.device})")
                print(f"  graph_view: {graph_view.shape} (设备: {graph_view.device})")
            # if i == 0 and epoch == 0:
            #     print(f"输出特征形状:")
            #     print(f"  seq_view: {seq_view.shape}")
            #     print(f"  graph_view: {graph_view.shape}")

            # 计算对比损失
            loss = self.nt_xent_criterion(seq_view, graph_view)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 记录损失
            total_loss += loss.item()
            batch_count += 1
            last_loss = loss.item()  # 每个 batch 都更新，最后就是最后一个 batch 的 loss
            # 打印训练进度
            if i % self.args.print_freq == 0:
                print(f'Train: Epoch {epoch} batch {i} Loss {loss.item():.4f}')

        # 更新学习率
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} 学习率: {current_lr}")
#------------------
        # 打印GPU显存使用
        if torch.cuda.is_available():
            print(f"[Epoch {epoch}结束] GPU显存使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
#------------------
        # 返回平均损失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch} 最后一个 batch 训练损失: {last_loss:.4f} | 平均训练损失: {avg_loss:.4f}")
        return avg_loss



    def val_per_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        print(f"\n开始第 {epoch} 个epoch的验证...")

        # 打印验证开始时的GPU状态
        if torch.cuda.is_available():
            print(f"[验证开始] GPU显存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                # 解包数据（数据已在collate中移动到GPU，无需再移动）
                pdbids, smiles, seqs, bgl, bgp = data
                # ✅ 手动移动到 GPU
                smiles = smiles.to(self.device)
                seqs = seqs.to(self.device)
                bgl = bgl.to(self.device)
                bgp = bgp.to(self.device)

                # ====== 验证数据设备（仅第一个batch） ======
                # if i == 0:
                #     print(f"验证Batch {i}设备检查:")
                #     print(f"  smiles设备: {smiles.device}")
                #     print(f"  seqs设备: {seqs.device}")
                #     print(f"  bgl设备: {bgl.device if hasattr(bgl, 'device') else '无device属性'}")
                #     print(f"  bgp设备: {bgp.device if hasattr(bgp, 'device') else '无device属性'}")
                #
                #     # 确认所有数据都在正确设备上
                #     if (smiles.device == self.device and
                #             seqs.device == self.device):
                #         print(f"  ✅ 验证数据已在正确设备上")
                #     else:
                #         print(f"  ⚠️ 验证数据设备不匹配，可能需要手动移动")

                # 前向传播
                seq_view, graph_view = self.model(smiles, seqs, bgl, bgp)

                # 计算对比损失
                loss = self.nt_xent_criterion(seq_view, graph_view)

                # 记录损失
                total_loss += loss.item()
                batch_count += 1

                # 打印验证进度
                # if i % self.args.print_freq == 0:
                #     print(f'Val: Epoch {epoch} batch {i} Loss {loss.item():.4f}')

                last_loss = loss.item()  # 每个 batch 都更新，最后就是最后一个 batch 的 loss

            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch} 最后一个 batch 验证损失: {last_loss:.4f} | 平均验证损失: {avg_loss:.4f}")

        # 打印验证结束时的GPU状态
        if torch.cuda.is_available():
            print(f"[验证结束] GPU显存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        # 返回平均损失
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss


    def train(self):
        """主训练循环"""
        loss_list = []

        print("=" * 60)
        print("开始训练")
        print("=" * 60)
###断点恢复#####################
        # if hasattr(self, 'start_epoch'):
        #     print(f"从 epoch {self.start_epoch} 开始训练")
###############################
        # 训练开始前的GPU信息
        if torch.cuda.is_available():
            print(f"初始GPU显存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"可用GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # 记录最佳模型
        best_val_loss = float('inf')
        best_epoch = -1
        patience_counter = 0
        patience = 40  # 早停耐心值
################################
        # # 如果存在最佳模型，加载其验证损失
        # if os.path.exists('best_model.pt'):
        #     try:
        #         best_checkpoint = torch.load('best_model.pt', map_location=self.device)
        #         best_val_loss = best_checkpoint.get('val_loss', float('inf'))
        #         best_epoch = best_checkpoint.get('epoch', -1)
        #         print(f"加载历史最佳模型: epoch {best_epoch}, val_loss {best_val_loss:.4f}")
        #     except:
        #         print("⚠️ 无法加载历史最佳模型，从头开始记录")
        #
        # patience_counter = 0
        # patience = 60  # 将早停耐心值改为60
        # # 从指定的起始epoch开始训练
        # for epoch in range(self.start_epoch, self.args.epochs):
##################################
        for epoch in range(self.args.epochs):
            train_loss = self.train_per_epoch(epoch)
            val_loss = self.val_per_epoch(epoch)

            # 保存模型和日志
            self.logger.save_curves(epoch)

            # 保存检查点
            # self.logger.save_check_point(self.model, epoch)
            # ✅ 覆盖式保存 latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, 'checkpoint_latest.pt')
            '''
            恢复训练时（可选）
            ckpt = torch.load('checkpoint_latest.pt', map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            '''

            # 保存最佳模型
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_epoch = epoch
            #     patience_counter = 0
            #
            #     # 保存最佳模型
            #     best_model_path = f'best_model_epoch{epoch}_loss{val_loss:.4f}.pt'
            #     torch.save({
            #         'epoch': epoch,
            #         'model_state_dict': self.model.state_dict(),
            #         'optimizer_state_dict': self.optimizer.state_dict(),
            #         'train_loss': train_loss,
            #         'val_loss': val_loss,
            #     }, best_model_path)
            #     print(f"✅ 保存最佳模型: {best_model_path}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_model.pt')
                print(f"✅ 更新并覆盖最优模型 (epoch={epoch}, val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⚠️ 早停触发: {patience}个epoch验证损失未改善")
                    break

            print(f'epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | '
                  f'best val loss {best_val_loss:.4f} (epoch {best_epoch})')

            loss_list.append([float(train_loss), float(val_loss)])

            # 每10个epoch保存一次损失
            if epoch % 10 == 0 or epoch == self.args.epochs - 1:
                train_loss_df = pd.DataFrame(data=loss_list, columns=['train_loss', 'val_loss'])
                train_loss_df.to_csv(f'loss_list_epoch_{epoch}.csv', index=False)

                # 每10个epoch打印一次GPU统计
                if torch.cuda.is_available():
                    print(f"[Epoch {epoch}] GPU显存统计:")
                    print(f"  当前使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                    print(f"  峰值使用: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
                    print(f"  缓存: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
                    torch.cuda.reset_peak_memory_stats()  # 重置峰值统计

        # 保存最终损失记录
        train_loss_df = pd.DataFrame(data=loss_list, columns=['train_loss', 'val_loss'])
        train_loss_df.to_csv('loss_list_final.csv', index=False)

        # 训练结束统计
        print("\n" + "=" * 60)
        print("训练完成")
        print("=" * 60)
        print(f"最佳验证损失: {best_val_loss:.4f} (epoch {best_epoch})")
        print(f"总训练epoch数: {len(loss_list)}")

        if torch.cuda.is_available():
            print(f"最终GPU显存使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"最大GPU显存使用: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

        print("损失记录已保存到 loss_list_final.csv")
        print("=" * 60)


def main():
    try:
        # 设置GPU设备可见性（如果之前没有设置）
        if torch.cuda.is_available():
            print(f"发现 {torch.cuda.device_count()} 个GPU设备")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        trainer = ContrastiveTrainer()
        trainer.train()

    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU显存不足错误: {e}")
        print("建议:")
        print("  1. 减小 batch_size")
        print("  2. 使用梯度累积")
        print("  3. 启用混合精度训练")
        print("  4. 检查模型和数据大小")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

