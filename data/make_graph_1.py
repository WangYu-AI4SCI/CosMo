# import numpy as np
# import pandas as pd
# from mol2graph import pdbbind_handle,drop_nodes,permute_edges,subgraph
# import os
# from copy import deepcopy
# import multiprocessing
#
# '''
# 当前问题：
# 注释掉的代码: 配体图处理部分被注释掉了
# 缺少保存逻辑: 处理结果没有保存到文件
# 错误处理简单: 只打印索引，没有详细错误信息
# 单线程处理: 没有利用多核优势
# '''
## 把原始的 PDB 文件（配体和蛋白质结构文件）转换成图（DGLGraph），
## 再对这些图做数据增强，最终保存成 .bin 或 .npy 文件供模型训练时使用。
## 从PDBbind数据集中提取蛋白质-配体复合物，并将其转换为图结构数据，同时应用数据增强
# filelist = os.listdir('/data')
# protein_path='/data/'
# ligand_path='/data/'
# results = []
# error=[]
# for index,i in enumerate(filelist):
# #def handle(index,i):
#     #output=pdbbind_handle(i,protein_path,ligand_path,5.0)
#     #results.append(output)
#     a=i.split("_")
#     fingerprints =[]
#     ids=[]
#     graphs_p=[]
#     graphs_l=[]
#     # 构建文件路径
#     protein_path_pdb='%s/%s/'%(protein_path,i)+a[0]+'_handle_pocket_5.0.pdb'
#     ligand_path_pdb='%s/%s/'%(protein_path,i)+a[0]+'_'+a[2]+'_'+a[3]+ "_"+a[4]+'.pdb'
#     try:
#         # 生成蛋白质和配体的图
#         # 读取pdb文件，返回蛋白质ID、蛋白质图、配体图
#         pdbid, gp, gl=pdbbind_handle(i,protein_path_pdb,ligand_path_pdb,10.0)
#         ids.append(pdbid)
#
#         # 对图进行数据增强
#         gp_drop_nodes=drop_nodes(deepcopy(gp),0.2)
#         gp_permute_edges=permute_edges(deepcopy(gp),0.2)
#         gp_subgraph=subgraph(deepcopy(gp),0.2)
#         gl_drop_nodes=drop_nodes(deepcopy(gl),0.2)
#         gl_permute_edges=permute_edges(deepcopy(gl),0.2)
#         gl_subgraph=subgraph(deepcopy(gl),0.2)
#
#         del gl_subgraph.nodes['_N'].data['_ID']
#         del gl_subgraph.edata['_ID']
#         # # 删除子图中的ID信息
#         del gp_subgraph.nodes['_N'].data['_ID']
#         del gp_subgraph.edata['_ID']
#         #print()
#
#         print('gl:',gl,gl_drop_nodes,gl_permute_edges,gl_subgraph)
#         print('gp:',gp,gp_drop_nodes,gp_permute_edges,gp_subgraph)
#
#         # 收集所有增强后的图
#         graphs_p.append(gp)  # 原始图
#         graphs_p.append(gp_drop_nodes)  # 节点丢弃增强
#         graphs_p.append(gp_permute_edges)  # 边扰动增强
#         graphs_p.append(gp_subgraph)  # 子图增强
#
#
#         graphs_l.append(gl)
#         graphs_l.append(gl_drop_nodes)
#         graphs_l.append(gl_permute_edges)
#         graphs_l.append(gl_subgraph)
#         # results.append(output)
#         # data=list(PLEC(ligand, protein=receptor, size=1024,  depth_protein=5,depth_ligand=1,distance_cutoff=5, sparse=False))
#  #保存
#         # fingerprints.append(data)
#         np.save("%s/%s/out_id.npy"%(protein_path,i), ids)
#         np.save("%s/%s/out_protein.npy"%(protein_path,i), graphs_p)
#         np.save("%s/%s/out_ligand.npy"%(protein_path,i), graphs_l)
#         # save_graphs("out_PLEC.bin", PLECs)
#         # np.save("%s/%s/out_PLEC.npy"%(protein_path,i),fingerprints)
#     except:
#         print(index) # 记录处理失败的索引
# '''
# 将处理结果保存为numpy文件：
# ID信息
# 蛋白质图（原始+3种增强）
# 配体图（原始+3种增强）
# /data/1abc_protein_ligand_1/
# ├── out_id.npy              # PDB ID
# ├── out_protein.npy         # 蛋白质图数据 (4个增强图)
# ├── out_ligand.npy          # 配体图数据 (4个增强图)
# └── out_PLEC.npy           # PLEC指纹特征
# '''
# #多进程
# #from multiprocessing import Pool
# #import os, time, random
# #import tempfile
# #if __name__=='__main__':
# #    #print('Parent process %s.' % os.getpid())
# #    p = Pool(20)
# #    #f = open('/data/pdb/pdb_id.txt')
#     #lines = f.readlines()
# #    for index,i in enumerate(filelist):
# #        #print(index)
# #        p.apply_async(handle, args=(index,i,))
# #    print('Waiting for all subprocesses done...')
# #    p.close()
# #    p.join()
# #    print('All subprocesses done.')






#新增从pdb文件中获取id.npy,protein.bin,ligand.bin
# import numpy as np
# import pandas as pd
#
# import os
# from copy import deepcopy
# import multiprocessing
# from dgl.data.utils import save_graphs
# import traceback
# import glob
#
# '''
# 目标：将PDBbind数据转换为CL-GNN模型所需的三个文件：
# 1. out_id_v2020_5A_new.npy - 保存所有复合物的ID
# 2. out_ligand_v2020_5A_new.bin - 保存所有配体图
# 3. out_protein_v2020_5A_new.bin - 保存所有蛋白质图
# '''
#
# # 配置路径 - 修正为你的实际路径
# base_data_path = '/data/ai4sci/lyx/CL-GNN-main/data/Pdbbind/v2020-other-PL'
# output_dir = '/data/ai4sci/lyx/CL-GNN-main/data/Pdbbind'
#
# # 确保输出目录存在
# os.makedirs(output_dir, exist_ok=True)
#
#
# # 获取PDBbind复合物目录列表（只获取4字符的PDB ID目录）
# def get_pdbbind_complexes(base_path):
#     """获取有效的PDBbind复合物目录"""
#     if not os.path.exists(base_path):
#         print(f"错误: 基础路径不存在: {base_path}")
#         return []
#
#     all_items = os.listdir(base_path)
#     complexes = []
#
#     for item in all_items:
#         item_path = os.path.join(base_path, item)
#         # 只处理4字符的目录（标准的PDB ID格式）
#         if os.path.isdir(item_path) and len(item) == 4:
#             complexes.append(item)
#
#     print(f"找到 {len(complexes)} 个PDBbind复合物目录")
#     return sorted(complexes)
#
#
# # 获取复合物列表
# complex_list = get_pdbbind_complexes(base_data_path)
#
# if not complex_list:
#     print("错误: 没有找到有效的PDBbind复合物目录")
#     print(f"请检查路径: {base_data_path}")
#     print("该目录应该包含像 '11gs', '1a2b' 这样的4字符子目录")
#     exit(1)
#
# # 初始化存储列表
# all_ids = []
# all_protein_graphs = []
# all_ligand_graphs = []
# error_indices = []
#
#
# def process_single_complex(index, pdbid):
#     """处理单个蛋白质-配体复合物"""
#     try:
#         print(f"处理第 {index} 个复合物: {pdbid}")
#
#         # 构建正确的文件路径 - 根据你的实际文件命名
#         complex_dir = os.path.join(base_data_path, pdbid)
#
#         # 蛋白质文件 - 使用 pocket.pdb
#         protein_path_pdb = os.path.join(complex_dir, f"{pdbid}_pocket.pdb")
#
#         # 配体文件 - 尝试多种可能的命名
#         ligand_path_sdf = os.path.join(complex_dir, f"{pdbid}_ligand.sdf")
#         ligand_path_mol2 = os.path.join(complex_dir, f"{pdbid}_ligand.mol2")
#         ligand_path_pdb = os.path.join(complex_dir, f"{pdbid}_ligand.pdb")
#
#         # 检查文件是否存在
#         if not os.path.exists(protein_path_pdb):
#             raise FileNotFoundError(f"蛋白质文件不存在: {protein_path_pdb}")
#
#         # 确定可用的配体文件
#         ligand_file = None
#         for lig_path in [ligand_path_sdf, ligand_path_mol2, ligand_path_pdb]:
#             if os.path.exists(lig_path):
#                 ligand_file = lig_path
#                 print(f"使用配体文件: {os.path.basename(lig_path)}")
#                 break
#
#         if ligand_file is None:
#             raise FileNotFoundError(f"没有找到任何配体文件 for {pdbid}")
#
#         # 生成蛋白质和配体的图
#         print(f"调用 pdbbind_handle: 蛋白质={protein_path_pdb}, 配体={ligand_file}")
#         pdbid_result, gp, gl = pdbbind_handle(pdbid, base_data_path, base_data_path, 5.0)
#
#         if gp is None or gl is None:
#             raise Exception(f"图生成失败 for {pdbid}")
#
#         print(f"成功生成图: {pdbid} - 蛋白图节点数: {gp.number_of_nodes()}, 配体图节点数: {gl.number_of_nodes()}")
#
#         return {
#             'success': True,
#             'pdbid': pdbid,
#             'protein_graph': gp,
#             'ligand_graph': gl,
#             'index': index
#         }
#
#     except Exception as e:
#         print(f"Error processing {pdbid} (index {index}): {str(e)}")
#         traceback.print_exc()
#         return {
#             'success': False,
#             'index': index,
#             'pdbid': pdbid,
#             'error': str(e)
#         }
#
#
# print("开始处理PDBbind数据集...")
# print(f"总共需要处理: {len(complex_list)} 个复合物")
#
# # 单线程处理（稳定）
# success_count = 0
# for index, pdbid in enumerate(complex_list):
#     result = process_single_complex(index, pdbid)
#
#     if result['success']:
#         all_ids.append(result['pdbid'])
#         all_protein_graphs.append(result['protein_graph'])
#         all_ligand_graphs.append(result['ligand_graph'])
#         success_count += 1
#         print(f"✓ 成功处理: {result['pdbid']} ({index + 1}/{len(complex_list)})")
#     else:
#         error_indices.append(result['index'])
#         print(f"✗ 处理失败: {result['pdbid']} (index {result['index']})")
#
#     # 每处理10个复合物输出一次进度
#     if (index + 1) % 10 == 0:
#         print(f"进度: {index + 1}/{len(complex_list)}, 成功: {success_count}, 失败: {len(error_indices)}")
#
# print(f"\n处理完成统计:")
# print(f"成功处理: {len(all_ids)} 个复合物")
# print(f"处理失败: {len(error_indices)} 个")
# print(f"成功率: {len(all_ids) / len(complex_list) * 100:.1f}%")
#
# if error_indices:
#     print(f"失败的复合物: {[complex_list[i] for i in error_indices]}")
#
# # 保存为CL-GNN所需的三个文件
# if all_ids:
#     print("\n开始保存数据...")
#
#     # 1. 保存ID文件 (.npy)
#     ids_path = f"{output_dir}/out_id_v2020_5A_new.npy"
#     np.save(ids_path, np.array(all_ids))
#     print(f"保存ID文件: {ids_path} - 包含 {len(all_ids)} 个ID")
#
#     # 2. 保存蛋白质图文件 (.bin)
#     protein_path_bin = f"{output_dir}/out_protein_v2020_5A_new.bin"
#     save_graphs(protein_path_bin, all_protein_graphs)
#     print(f"保存蛋白质图: {protein_path_bin} - 包含 {len(all_protein_graphs)} 个图")
#
#     # 3. 保存配体图文件 (.bin)
#     ligand_path_bin = f"{output_dir}/out_ligand_v2020_5A_new.bin"
#     save_graphs(ligand_path_bin, all_ligand_graphs)
#     print(f"保存配体图: {ligand_path_bin} - 包含 {len(all_ligand_graphs)} 个图")
#
#     # 4. 保存处理统计信息
#     stats = {
#         'total_complexes': len(complex_list),
#         'successful': len(all_ids),
#         'failed': len(error_indices),
#         'success_rate': len(all_ids) / len(complex_list),
#         'failed_indices': error_indices,
#         'failed_pdbids': [complex_list[i] for i in error_indices] if error_indices else []
#     }
#     stats_path = f"{output_dir}/processing_stats.npy"
#     np.save(stats_path, stats)
#     print(f"保存统计信息: {stats_path}")
#
#     print("\n数据保存完成！")
#
#     # 显示成功处理的示例
#     print(f"\n成功处理的复合物示例: {all_ids[:5] if len(all_ids) > 5 else all_ids}")
#
# else:
#     print("没有成功处理任何数据，请检查错误信息。")
#
# # 验证保存的文件
# print("\n验证保存的文件:")
# for file in ['out_id_v2020_5A_new.npy', 'out_protein_v2020_5A_new.bin', 'out_ligand_v2020_5A_new.bin']:
#     file_path = f"{output_dir}/{file}"
#     if os.path.exists(file_path):
#         file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
#         print(f"✓ {file}: {file_size:.2f} MB")
#     else:
#         print(f"✗ {file}: 文件不存在")


import numpy as np
import pandas as pd
from mol2graph import mol_to_graph2  # 直接导入底层函数
import os
from copy import deepcopy
import multiprocessing
from dgl.data.utils import save_graphs
import traceback
import glob

'''
目标：将PDBbind数据转换为CL-GNN模型所需的三个文件：
1. out_id_v2020_5A_new.npy - 保存所有复合物的ID
2. out_ligand_v2020_5A_new.bin - 保存所有配体图
3. out_protein_v2020_5A_new.bin - 保存所有蛋白质图
'''

# 配置路径
# base_data_path = '/data/ai4sci/lyx/CL-GNN-main/data/Pdbbind/v2020-other-PL'
# output_dir = '/data/ai4sci/lyx/CL-GNN-main/data/Pdbbind'

# base_data_path = '/data/ai4sci/lyx/CL-GNN-main/data/CASF-2016/coreset'
# output_dir = '/data/ai4sci/lyx/CL-GNN-main/data/CASF-2016'

base_data_path = '/data/ai4sci/lyx/CL-GNN-main/data/CASF_2013_lyx/coreset'
output_dir = '/data/ai4sci/lyx/CL-GNN-main/data/CASF_2013_lyx'

# base_data_path = '/data/ai4sci/lyx/CL-GNN-main/data/Pdbbind_v2016'
# output_dir = '/data/ai4sci/lyx/CL-GNN-main/data/Pdbbind'


# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)


def get_pdbbind_complexes(base_path):
    """获取有效的PDBbind复合物目录"""
    if not os.path.exists(base_path):
        print(f"错误: 基础路径不存在: {base_path}")
        return []

    all_items = os.listdir(base_path)
    complexes = []

    for item in all_items:
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and len(item) == 4:
            # 检查是否包含必要的文件
            pocket_file = os.path.join(item_path, f"{item}_pocket.pdb")
            protein_file = os.path.join(item_path, f"{item}_protein.pdb")
            ligand_sdf = os.path.join(item_path, f"{item}_ligand.sdf")
            ligand_mol2 = os.path.join(item_path, f"{item}_ligand.mol2")

            # 检查至少有一个蛋白质文件和一个配体文件存在
            has_protein = os.path.exists(pocket_file) or os.path.exists(protein_file)
            has_ligand = os.path.exists(ligand_sdf) or os.path.exists(ligand_mol2)

            if has_protein and has_ligand:
                complexes.append(item)
            else:
                print(f"跳过 {item}: 缺少必要文件 (protein: {has_protein}, ligand: {has_ligand})")

    print(f"找到 {len(complexes)} 个有效的PDBbind复合物目录")
    return sorted(complexes)


def pdbbind_handle(pdbid, protein_file, ligand_file, cutoff=5.0):
    """
    自定义PDBbind处理函数，使用实际的文件路径
    """
    try:
        print(f"处理 {pdbid}: 蛋白质={os.path.basename(protein_file)}, 配体={os.path.basename(ligand_file)}")

        # 直接使用提供的文件路径调用mol_to_graph2
        gp, gl = mol_to_graph2(
            protein_file,
            ligand_file,
            cutoff=cutoff,
            explicit_H=False,
            # use_chirality=False
            use_chirality=True
        )

        return pdbid, gp, gl

    except Exception as e:
        print(f"处理 {pdbid} 失败: {str(e)}")
        traceback.print_exc()
        return pdbid, None, None


def process_single_complex(index, pdbid):
    """处理单个蛋白质-配体复合物"""
    try:
        print(f"处理第 {index} 个复合物: {pdbid}")

        # 构建正确的文件路径
        complex_dir = os.path.join(base_data_path, pdbid)

        # 蛋白质文件 - 优先使用pocket.pdb，如果没有则使用protein.pdb
        protein_path_pocket = os.path.join(complex_dir, f"{pdbid}_pocket.pdb")
        protein_path_protein = os.path.join(complex_dir, f"{pdbid}_protein.pdb")

        protein_file = None
        if os.path.exists(protein_path_pocket):
            protein_file = protein_path_pocket
            print(f"使用蛋白质文件: {pdbid}_pocket.pdb")
        elif os.path.exists(protein_path_protein):
            protein_file = protein_path_protein
            print(f"使用蛋白质文件: {pdbid}_protein.pdb")
        else:
            raise FileNotFoundError(f"没有找到蛋白质文件 for {pdbid}")

        # 配体文件 - 尝试多种格式
        ligand_path_mol2 = os.path.join(complex_dir, f"{pdbid}_ligand.mol2")
        ligand_path_sdf = os.path.join(complex_dir, f"{pdbid}_ligand.sdf")

        ligand_file = None
        for lig_path in [ligand_path_mol2, ligand_path_sdf]:
            if os.path.exists(lig_path):
                ligand_file = lig_path
                print(f"使用配体文件: {os.path.basename(lig_path)}")
                break

        if ligand_file is None:
            raise FileNotFoundError(f"没有找到任何配体文件 for {pdbid}")

        # 使用自定义处理函数
        pdbid_result, gp, gl = pdbbind_handle(
            pdbid,
            protein_file,
            ligand_file,
            cutoff=5.0
        )

        if gp is None or gl is None:
            raise Exception(f"图生成失败 for {pdbid}")

        print(f"成功生成图: {pdbid} - 蛋白图节点数: {gp.number_of_nodes()}, 配体图节点数: {gl.number_of_nodes()}")

        return {
            'success': True,
            'pdbid': pdbid,
            'protein_graph': gp,
            'ligand_graph': gl,
            'index': index
        }

    except Exception as e:
        print(f"Error processing {pdbid} (index {index}): {str(e)}")
        return {
            'success': False,
            'index': index,
            'pdbid': pdbid,
            'error': str(e)
        }


# 获取复合物列表
complex_list = get_pdbbind_complexes(base_data_path)

if not complex_list:
    print("错误: 没有找到有效的PDBbind复合物目录")
    exit(1)

# 初始化存储列表
all_ids = []
all_protein_graphs = []
all_ligand_graphs = []
error_indices = []

print("开始处理PDBbind数据集...")
print(f"总共需要处理: {len(complex_list)} 个复合物")

# 单线程处理
success_count = 0
for index, pdbid in enumerate(complex_list):
    result = process_single_complex(index, pdbid)

    if result['success']:
        all_ids.append(result['pdbid'])
        all_protein_graphs.append(result['protein_graph'])
        all_ligand_graphs.append(result['ligand_graph'])
        success_count += 1
        print(f"✓ 成功处理: {result['pdbid']} ({index + 1}/{len(complex_list)})")
    else:
        error_indices.append(result['index'])
        print(f"✗ 处理失败: {result['pdbid']} (index {result['index']})")

    # 每处理10个复合物输出一次进度
    if (index + 1) % 10 == 0:
        print(f"进度: {index + 1}/{len(complex_list)}, 成功: {success_count}, 失败: {len(error_indices)}")

print(f"\n处理完成统计:")
print(f"成功处理: {len(all_ids)} 个复合物")
print(f"处理失败: {len(error_indices)} 个")
print(f"成功率: {len(all_ids) / len(complex_list) * 100:.1f}%")

if error_indices:
    print(f"失败的复合物索引: {error_indices}")
    print(f"失败的复合物ID: {[complex_list[i] for i in error_indices]}")

# 保存为CL-GNN所需的三个文件
if all_ids:
    print("\n开始保存数据...")

    # 1. 保存ID文件 (.npy)
    # ids_path = f"{output_dir}/out_id_v2016.npy"
    ids_path = f"{output_dir}/out_id_casf2013.npy"
    np.save(ids_path, np.array(all_ids))
    print(f"保存ID文件: {ids_path} - 包含 {len(all_ids)} 个ID")

    # 2. 保存蛋白质图文件 (.bin)
    # protein_path_bin = f"{output_dir}/out_protein_v2016.bin"
    protein_path_bin = f"{output_dir}/out_protein_casf2013.bin"
    save_graphs(protein_path_bin, all_protein_graphs)
    print(f"保存蛋白质图: {protein_path_bin} - 包含 {len(all_protein_graphs)} 个图")

    # 3. 保存配体图文件 (.bin)
    # ligand_path_bin = f"{output_dir}/out_ligand_v2016.bin"
    ligand_path_bin = f"{output_dir}/out_ligand_casf2013.bin"
    save_graphs(ligand_path_bin, all_ligand_graphs)
    print(f"保存配体图: {ligand_path_bin} - 包含 {len(all_ligand_graphs)} 个图")

    # 4. 保存处理统计信息
    stats = {
        'total_complexes': len(complex_list),
        'successful': len(all_ids),
        'failed': len(error_indices),
        'success_rate': len(all_ids) / len(complex_list),
        'failed_indices': error_indices,
        'failed_pdbids': [complex_list[i] for i in error_indices] if error_indices else []
    }
    stats_path = f"{output_dir}/processing_stats.npy"
    np.save(stats_path, stats)
    print(f"保存统计信息: {stats_path}")

    print("\n数据保存完成！")

    # 显示成功处理的示例
    print(f"\n成功处理的复合物示例: {all_ids[:5] if len(all_ids) > 5 else all_ids}")

else:
    print("没有成功处理任何数据，请检查错误信息。")

# 验证保存的文件
print("\n验证保存的文件:")
# for file in ['out_id_CASF_2016_5A.npy', 'out_ligand_CASF_2016_5A.bin', 'out_protein_CASF_2016_5A.bin']:
# for file in ['out_id_v2016.npy', 'out_ligand_v2016.bin', 'out_protein_v2016.bin']:
for file in ['out_id_casf2013.npy', 'out_ligand_casf2013.bin', 'out_protein_casf2013.bin']:
    file_path = f"{output_dir}/{file}"
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"✓ {file}: {file_size:.2f} MB")
    else:
        print(f"✗ {file}: 文件不存在")




