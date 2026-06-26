import os
import numpy as np
import torch as th
import dgl
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from itertools import permutations
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================

# 数据路径
RECEPTOR_DIR = "/data/ai4sci/lyx/receptor_01"
OUTPUT_DIR = "/data/ai4sci/lyx/receptor_01/protein_graphs"

# 参数
CUTOFF = 10.0  # 距离截断值（Å）

# 金属原子列表
METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT",
         "ZN", "CO", "PD", "AG", "CR", "FE", "V", "MN", "HG", "GA", "CD", "YB",
         "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y",
         "LA", "CE", "PR", "ND", "GD", "TB", "DY", "ER", "TM", "LU", "HF", "ZR",
         "CE", "U", "PU", "TH"]

# 水分子和溶剂
SOLVENT = ['HOH', 'WAT', 'SOL']


# ============================================================================
# 辅助函数
# ============================================================================

def one_of_k_encoding_unk(x, allowable_set):
    """将输入编码为one-hot，不在集合中的映射到最后一个元素"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [float(x == s) for s in allowable_set]  # 转换为float


def obtain_resname(res):
    """获取残基名称，将金属离子统一标记为'M'"""
    resname = res.resname.strip()

    # 检查是否在金属列表中
    if resname in METAL:
        return "M"

    return resname


def obtain_self_dist(res):
    """计算残基内部距离特征"""
    try:
        xx = res.atoms
        if len(xx) == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        dists = distances.self_distance_array(xx.positions)

        # 获取CA、C、N、O原子
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")

        # 计算距离
        ca_o_dist = distances.dist(ca, o)[-1][0] if len(ca) > 0 and len(o) > 0 else 0.0
        o_n_dist = distances.dist(o, n)[-1][0] if len(o) > 0 and len(n) > 0 else 0.0
        n_c_dist = distances.dist(n, c)[-1][0] if len(n) > 0 and len(c) > 0 else 0.0

        max_dist = dists.max() * 0.1 if len(dists) > 0 else 0.0
        min_dist = dists.min() * 0.1 if len(dists) > 0 else 0.0

        return [float(max_dist), float(min_dist),
                float(ca_o_dist * 0.1), float(o_n_dist * 0.1), float(n_c_dist * 0.1)]
    except Exception as e:
        return [0.0, 0.0, 0.0, 0.0, 0.0]


def obtain_dihedral_angles(res):
    """计算二面角：φ, ψ, ω, χ1"""
    try:
        phi = res.phi_selection().dihedral.value() if res.phi_selection() is not None else 0.0
        psi = res.psi_selection().dihedral.value() if res.psi_selection() is not None else 0.0
        omega = res.omega_selection().dihedral.value() if res.omega_selection() is not None else 0.0
        chi1 = res.chi1_selection().dihedral.value() if res.chi1_selection() is not None else 0.0

        return [float(phi * 0.01), float(psi * 0.01), float(omega * 0.01), float(chi1 * 0.01)]
    except:
        return [0.0, 0.0, 0.0, 0.0]


def calc_res_features(res):
    """计算残基特征（41维）"""
    # 残基类型列表（32种）
    residue_types = [
        'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
        'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS',
        'MSE', 'CSO', 'PTR', 'TPO', 'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP',
        'M', 'X'
    ]

    resname = obtain_resname(res)

    # 残基类型 one-hot (32维)
    type_features = one_of_k_encoding_unk(resname, residue_types)

    # 内部距离特征 (5维)
    dist_features = obtain_self_dist(res)

    # 二面角特征 (4维)
    dihedral_features = obtain_dihedral_angles(res)

    # 总特征: 32 + 5 + 4 = 41维
    features = type_features + dist_features + dihedral_features

    # 确保所有值都是float
    return np.array(features, dtype=np.float32)


def obtain_ca_pos(res):
    """获取CA原子位置或质心"""
    resname = obtain_resname(res)

    try:
        if resname == "M":
            # 金属离子使用第一个原子的位置
            if len(res.atoms) > 0:
                pos = res.atoms.positions[0]
                return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
        else:
            # 尝试获取CA原子
            ca = res.atoms.select_atoms("name CA")
            if len(ca) > 0:
                pos = ca.positions[0]
                return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)

            # 尝试其他CA命名
            ca = res.atoms.select_atoms("name CA")
            if len(ca) > 0:
                pos = ca.positions[0]
                return np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)

            # 使用质心
            if len(res.atoms) > 0:
                center = res.atoms.center_of_mass()
                return np.array([float(center[0]), float(center[1]), float(center[2])], dtype=np.float32)
    except Exception as e:
        pass

    return np.array([0.0, 0.0, 0.0], dtype=np.float32)


def calc_dist(res1, res2):
    """计算两个残基之间的最小原子距离"""
    try:
        if len(res1.atoms) == 0 or len(res2.atoms) == 0:
            return np.array([[999.0]])
        return distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    except:
        return np.array([[999.0]])


def obtain_edge(u, cutoff=10.0):
    """构建边"""
    edgeids = []
    dismin = []
    dismax = []

    # 只保留非溶剂残基
    residues = [res for res in u.residues if res.resname not in SOLVENT]

    for res1, res2 in permutations(residues, 2):
        try:
            dist = calc_dist(res1, res2)
            min_dist = dist.min()
            if min_dist <= cutoff:
                edgeids.append([res1.ix, res2.ix])
                dismin.append(float(min_dist * 0.1))
                dismax.append(float(dist.max() * 0.1))
        except:
            continue

    if edgeids:
        dist_features = np.array([dismin, dismax], dtype=np.float32).T
        return edgeids, dist_features
    else:
        return [], np.array([], dtype=np.float32)


# ============================================================================
# 核心函数：蛋白质转图
# ============================================================================

def prot_to_graph(pdb_file, cutoff=10.0):
    """
    将蛋白质PDB文件转换为DGL图
    """

    # 检查文件
    if not os.path.exists(pdb_file):
        print(f"文件不存在: {pdb_file}")
        return None

    try:
        # 加载蛋白质结构
        u = mda.Universe(pdb_file)

        # 过滤掉水分子和溶剂
        residues = [res for res in u.residues if res.resname not in SOLVENT]

        if len(residues) == 0:
            print(f"警告: {pdb_file} 没有有效残基")
            return None

        # 创建图
        g = dgl.DGLGraph()
        g.add_nodes(len(residues))

        # 计算节点特征和位置
        res_feats = []
        positions = []

        for res in residues:
            # 计算残基特征
            features = calc_res_features(res)
            res_feats.append(features)

            # 计算残基位置
            pos = obtain_ca_pos(res)
            positions.append(pos)

        # 转换为numpy数组并确保数据类型正确
        feats_array = np.stack(res_feats, axis=0)  # (N, 41)
        pos_array = np.stack(positions, axis=0)  # (N, 3)

        # 添加节点特征 - 修复dtype问题
        g.ndata["feats"] = th.tensor(feats_array, dtype=th.float32)
        g.ndata["pos"] = th.tensor(pos_array, dtype=th.float32)

        # 构建边
        edgeids, dist_features = obtain_edge(u, cutoff)

        if edgeids and len(dist_features) > 0:
            src_list, dst_list = zip(*edgeids)
            g.add_edges(src_list, dst_list)

            # 确保边特征数据类型正确
            dist_tensor = th.tensor(dist_features, dtype=th.float32)
            g.edata["dist"] = dist_tensor

        return g

    except Exception as e:
        print(f"处理失败 {pdb_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# 获取PDB ID列表
# ============================================================================

def get_pdbid_list_from_dir(data_dir):
    """
    从数据目录获取所有PDB ID

    Parameters:
    -----------
    data_dir : str
        数据目录路径

    Returns:
    --------
    pdbid_list : list
        PDB ID列表
    """
    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在 {data_dir}")
        return []

    # 获取所有子文件夹
    pdbid_list = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            pdbid_list.append(item)

    return sorted(pdbid_list)


# ============================================================================
# 批量处理函数
# ============================================================================

def process_single_pdbid(pdbid, data_dir, cutoff=10.0):
    """
    处理单个PDB ID
    """

    # 构建蛋白质文件路径
    protein_file = os.path.join(data_dir, pdbid, f"{pdbid}_protein.pdb")

    # 如果protein文件不存在，尝试其他命名
    if not os.path.exists(protein_file):
        # 尝试pocket文件
        pocket_file = os.path.join(data_dir, pdbid, f"{pdbid}_pocket.pdb")
        if os.path.exists(pocket_file):
            protein_file = pocket_file
        else:
            # 尝试查找任何.pdb文件
            pdb_dir = os.path.join(data_dir, pdbid)
            if os.path.exists(pdb_dir):
                pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
                if pdb_files:
                    protein_file = os.path.join(pdb_dir, pdb_files[0])
                else:
                    print(f"警告: 找不到 {pdbid} 的PDB文件")
                    return pdbid, None
            else:
                print(f"警告: 目录不存在 {pdb_dir}")
                return pdbid, None

    # 生成图
    graph = prot_to_graph(protein_file, cutoff)

    return pdbid, graph


def batch_process_proteins(data_dir, output_dir=None, cutoff=10.0,
                           pdbid_list=None):
    """
    批量处理蛋白质
    """

    # 获取PDB ID列表
    if pdbid_list is None:
        pdbid_list = get_pdbid_list_from_dir(data_dir)

    if not pdbid_list:
        print("错误: 没有找到任何PDB ID")
        return {}, []

    print(f"开始处理 {len(pdbid_list)} 个蛋白质...")
    print(f"数据目录: {data_dir}")
    print(f"距离截断值: {cutoff} Å")
    print("-" * 50)

    graphs = {}
    failed = []

    # 串行处理
    for pdbid in tqdm(pdbid_list, desc="处理蛋白质"):
        pdbid, graph = process_single_pdbid(pdbid, data_dir, cutoff)

        if graph is not None:
            graphs[pdbid] = graph
        else:
            failed.append(pdbid)

    print("-" * 50)
    print(f"处理完成:")
    print(f"  成功: {len(graphs)}")
    print(f"  失败: {len(failed)}")

    # 保存图到文件
    if output_dir and graphs:
        os.makedirs(output_dir, exist_ok=True)

        for pdbid, graph in graphs.items():
            save_path = os.path.join(output_dir, f"{pdbid}_protein_graph.bin")
            dgl.save_graphs(save_path, [graph])
            print(f"保存: {save_path}")

    return graphs, failed


# ============================================================================
# 统计和分析函数
# ============================================================================

def analyze_graph(graph, pdbid):
    """分析图统计信息"""
    if graph is None:
        return None

    stats = {
        'pdbid': pdbid,
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'avg_degree': 2 * graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        'feature_dim': graph.ndata['feats'].shape[1] if 'feats' in graph.ndata else 0,
    }

    return stats


def batch_analyze(graphs, output_dir=None):
    """批量分析图"""
    import pandas as pd

    stats_list = []

    for pdbid, graph in graphs.items():
        stats = analyze_graph(graph, pdbid)
        if stats:
            stats_list.append(stats)

    if not stats_list:
        return None

    df = pd.DataFrame(stats_list)

    print("\n=== 图统计信息 ===")
    print(f"总图数: {len(df)}")
    print(f"平均节点数: {df['num_nodes'].mean():.1f} ± {df['num_nodes'].std():.1f}")
    print(f"节点数范围: {df['num_nodes'].min()} - {df['num_nodes'].max()}")
    print(f"平均边数: {df['num_edges'].mean():.1f} ± {df['num_edges'].std():.1f}")
    print(f"平均度数: {df['avg_degree'].mean():.2f}")

    # 保存统计信息
    if output_dir:
        stats_path = os.path.join(output_dir, "graph_statistics.csv")
        df.to_csv(stats_path, index=False)
        print(f"\n统计信息已保存: {stats_path}")

    return df


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""

    print("=" * 60)
    print("蛋白质图生成工具")
    print("=" * 60)

    # 1. 检查数据目录
    print(f"\n1. 检查数据目录: {RECEPTOR_DIR}")

    if not os.path.exists(RECEPTOR_DIR):
        print(f"错误: 数据目录不存在 {RECEPTOR_DIR}")
        return

    # 2. 获取PDB ID列表
    print(f"\n2. 获取PDB ID列表...")
    pdbid_list = get_pdbid_list_from_dir(RECEPTOR_DIR)
    print(f"  找到 {len(pdbid_list)} 个PDB ID")

    if not pdbid_list:
        print("错误: 没有找到任何PDB ID")
        return

    # 显示前10个
    print(f"  前10个: {pdbid_list[:10]}")

    # 3. 批量处理
    print(f"\n3. 开始生成蛋白质图...")
    graphs, failed = batch_process_proteins(
        RECEPTOR_DIR,
        OUTPUT_DIR,
        CUTOFF
    )

    # 4. 分析统计
    if graphs:
        print(f"\n4. 分析统计信息...")
        stats_df = batch_analyze(graphs, OUTPUT_DIR)

    # 5. 保存失败列表
    if failed:
        failed_path = os.path.join(OUTPUT_DIR, "failed_pdbids.txt")
        with open(failed_path, 'w') as f:
            for pdbid in failed:
                f.write(f"{pdbid}\n")
        print(f"\n失败列表已保存: {failed_path}")

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)


# ============================================================================
# 测试单个文件
# ============================================================================

def test_single_file():
    """测试单个文件"""
    test_pdb = "/data/ai4sci/lyx/receptor_01/101mA/101mA_pocket.pdb"

    print(f"测试文件: {test_pdb}")
    print("-" * 40)

    if os.path.exists(test_pdb):
        graph = prot_to_graph(test_pdb, CUTOFF)

        if graph is not None:
            print(f"\n图信息:")
            print(f"  节点数: {graph.number_of_nodes()}")
            print(f"  边数: {graph.number_of_edges()}")
            print(f"  节点特征形状: {graph.ndata['feats'].shape}")
            if 'dist' in graph.edata:
                print(f"  边特征形状: {graph.edata['dist'].shape}")
        else:
            print("处理失败")
    else:
        print(f"文件不存在: {test_pdb}")


# ============================================================================
# 运行
# ============================================================================

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 可选：测试单个文件
    test_single_file()

    # 运行主程序
    # main()