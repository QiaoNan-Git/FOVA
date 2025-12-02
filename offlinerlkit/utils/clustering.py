import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

import seaborn as sns
import sys
import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"   

def find_L1(local_dataset, obs_mean, obs_std, device="cpu", threshold=0.1):
    import torch

    # # 假设 local_dataset["observations"] 是 NumPy 数组或类似结构
    # s = local_dataset["observations"]
    # # s = local_dataset["next_observations"]

    # # 归一化（使用已知的 obs_mean 和 obs_std）
    # eps = 1e-8  # 防止除以零的小常数
    # s_normalized = (s - obs_mean) / (obs_std + eps)

    # # 转换为 PyTorch 张量并移动到 GPU
    # x_tensor = torch.tensor(s_normalized, dtype=torch.float32, device=device)


    # s = local_dataset["observations"]
    # # next_s = local_dataset["next_observations"]
    # a = local_dataset["actions"]
    # X = np.hstack([s, a])
    X = local_dataset["next_observations"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    # 计算 L1 距离矩阵
    def compute_l1_distance_matrix(x_tensor):
        diff = x_tensor[:, None, :] - x_tensor[None, :, :]  # [n, n, d]
        l1_dist = diff.abs().sum(dim=2)  # [n, n]
        return l1_dist

    l1_distance_matrix = compute_l1_distance_matrix(x_tensor)

    # 屏蔽对角线（设为无穷大，避免选到自己）
    n = l1_distance_matrix.shape[0]
    mask = torch.eye(n, dtype=torch.bool, device=device)  # 对角线为 True 的掩码
    l1_distance_matrix.masked_fill_(mask, float('inf'))  # 将对角线设为 inf

    # 生成行和列索引网格
    rows, cols = torch.meshgrid(torch.arange(n, device=device), 
                            torch.arange(n, device=device), 
                            indexing='ij')

    # 提取上三角部分（不包括对角线）
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
    l1_dist_triu = l1_distance_matrix[triu_mask]  # 上三角部分展平为一维向量
    i_triu = rows[triu_mask]  # 对应的行索引
    j_triu = cols[triu_mask]  # 对应的列索引

    # 计算 L1 距离 < threshold 的比例
    num_pairs = l1_dist_triu.shape[0]  # 总对数 = n*(n-1)/2
    num_close_pairs = (l1_dist_triu < threshold).sum().item()
    close_ratio = num_close_pairs / num_pairs

    # 计算 L1 距离 < threshold 且 |i-j| > 1000 的比例
    index_distance_threshold = 1000
    far_close_mask = (l1_dist_triu < threshold) & ((i_triu - j_triu).abs() > index_distance_threshold)
    num_far_close_pairs = far_close_mask.sum().item()
    far_close_ratio = num_far_close_pairs / num_pairs

    # 找到前5个最小的 L1 距离及其索引
    k = 10
    min_values, min_indices = torch.topk(-l1_dist_triu, k)  # 用 - 号模拟取最小
    min_values = -min_values  # 恢复正确的距离值
    min_i = i_triu[min_indices]
    min_j = j_triu[min_indices]

    # 找到前5个最大的 L1 距离及其索引
    max_values, max_indices = torch.topk(l1_dist_triu, k)
    max_i = i_triu[max_indices]
    max_j = j_triu[max_indices]

    # 获取所有满足 L1 < threshold 的 (i,j) 组合
    close_mask = l1_dist_triu < threshold
    close_pairs = list(zip(i_triu[close_mask].cpu().numpy(), 
                          j_triu[close_mask].cpu().numpy()))

    # 输出结果（保留原有打印功能）
    print("===== 最小的5个 L1 距离 =====")
    for idx in range(k):
        print(f"距离: {min_values[idx].item():.6f}, 位置: ({min_i[idx].item()}, {min_j[idx].item()})")

    print("\n===== 最大的5个 L1 距离 =====")
    for idx in range(1):
        print(f"距离: {max_values[idx].item():.6f}, 位置: ({max_i[idx].item()}, {max_j[idx].item()})")

    print(f"\n===== L1 距离 < {threshold} 的比例 =====")
    print(f"比例: {close_ratio:.4f} ({num_close_pairs}/{num_pairs})")

    print(f"\n===== L1 距离 < {threshold} 且 |i-j| > {index_distance_threshold} 的比例 =====")
    print(f"比例: {far_close_ratio:.4f} ({num_far_close_pairs}/{num_pairs})")

    # 返回所有满足 L1 < threshold 的 (i,j) 组合

    # # 绘制L1距离的PDF分布图
    # plt.figure(figsize=(10, 6))
    # l1_dist_np = l1_dist_triu.cpu().numpy()

    # # 使用seaborn的kdeplot绘制概率密度函数
    # # sns.kdeplot(l1_dist_np, bw_adjust=0.5, fill=True, color='skyblue')

    # # 添加阈值线
    # # plt.figure(figsize=(10, 6))

    # # 随机抽取1%的样本（可根据需要调整比例）
    # sample_size = min(100000, len(l1_dist_np))  # 最多取10万点
    # l1_sample = np.random.choice(l1_dist_np, size=sample_size, replace=False)

    # sns.kdeplot(l1_sample, bw_adjust=0.5, fill=True, color='skyblue')
    # plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    # plt.title('PDF of L1 Distances (Sampled)', fontsize=14)
    # plt.xlabel('L1 Distance', fontsize=12)
    # plt.ylabel('Probability Density', fontsize=12)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("pdf.png", dpi=150, bbox_inches='tight')
    # plt.close()  # 避免内存泄漏
    # sys.exit()
    print(X_scaled.shape)
    return close_pairs
    

def cluster_and_visualize(local_dataset, method='kmeans', k=3, device="cpu", n=5):


    if method == 'ogmm':
        return cluster_and_visualize_ogmm(local_dataset, method, k, device, n)
    elif method == 'none':
        return cluster_and_visualize_none(local_dataset, method, k, device, n)
    else:
        return cluster_and_visualize_model(local_dataset, method, k, device, n)
    
def cluster_and_visualize_none(local_dataset, method='kmeans', k=3, device="cpu", n=5):
    divided_datasets = []
    for _ in range(k):
        divided_datasets.append(local_dataset)
    return k, {}, [], divided_datasets


# def cluster_and_visualize_ogmm(local_dataset, method='kmeans', k=3, device="cpu", n=5):
def cluster_and_visualize_ogmm(local_dataset, method='ogmm', k=3, device="cpu", n=5, variance_reduction_threshold=0.0):
    """
    支持GPU加速的真正重叠GMM实现
    
    参数:
        device: "cuda" 或 "cpu"
        variance_reduction_threshold: 允许重叠的最小方差减少比例
    """
    # 数据准备
    s = local_dataset["observations"]
    a = local_dataset["actions"]
    X = np.hstack([s, a])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 转换为PyTorch张量并移到GPU
    if device == "cuda" and torch.cuda.is_available():
        X_tensor = torch.tensor(X_scaled, device="cuda")
    else:
        X_tensor = torch.tensor(X_scaled, device="cpu")
    
    # 训练GMM模型 (sklearn暂不支持GPU，保持CPU)
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    
    # 获取初始聚类结果
    probs = gmm.predict_proba(X_scaled)
    main_labels = np.argmax(probs, axis=1)
    
    # 初始化簇分配矩阵 (使用GPU)
    cluster_assignments = torch.zeros((len(X), k), dtype=torch.bool, device=device)
    cluster_assignments[torch.arange(len(X)), torch.tensor(main_labels, device=device)] = True
    
    # 计算初始各簇方差 (GPU加速)
    cluster_variances = []
    for i in range(k):
        mask = cluster_assignments[:, i]
        if mask.sum() > 1:
            cluster_var = torch.var(X_tensor[mask], dim=0).mean().item()
            cluster_variances.append(cluster_var)
        else:
            cluster_variances.append(0)
    
    max_epochs = n
    prev_assignments = cluster_assignments.clone()
    no_improvement_count = 0
    has_converged = False
    early_stop_patience = 2
    
    for epoch in range(max_epochs):
        if has_converged:
            print(f"Early stopping at epoch {epoch}")
            break
            
        epoch_changes = 0
        for point_idx in tqdm(range(len(X)), desc=f"Epoch {epoch}/{max_epochs}"):
            point = X_tensor[point_idx]
            original_cluster = main_labels[point_idx]
            
            for target_cluster in range(k):
                if target_cluster == original_cluster:
                    continue
                
                # 检查当前分配状态
                current_assignment = cluster_assignments[point_idx, target_cluster]
                if current_assignment:  # 如果已经分配则跳过
                    continue
                
                # 当前目标簇的点和方差
                target_mask = cluster_assignments[:, target_cluster]
                if target_mask.sum() == 0:
                    continue
                
                # 计算当前方差
                current_var = torch.var(X_tensor[target_mask], dim=0).mean().item()
                
                # 模拟添加点到目标簇
                temp_mask = target_mask.clone()
                temp_mask[point_idx] = True
                new_var = torch.var(X_tensor[temp_mask], dim=0).mean().item()
                
                # 计算方差减少比例
                if current_var > 0:
                    var_reduction = (current_var - new_var) / current_var
                else:
                    var_reduction = 0
                
                # 判断是否分配
                if var_reduction > variance_reduction_threshold:
                    cluster_assignments[point_idx, target_cluster] = True
                    cluster_variances[target_cluster] = new_var
                    epoch_changes += 1
        
        # 收敛检测
        if epoch_changes == 0:
            no_improvement_count += 1
            print(f"Epoch {epoch}: No changes detected ({no_improvement_count}/{early_stop_patience})")
        else:
            no_improvement_count = 0
            print(f"Epoch {epoch}: Made {epoch_changes} changes")
        
        # 检查是否提前停止
        if no_improvement_count >= early_stop_patience:
            has_converged = True
        
        # 检查是否完全收敛
        if torch.equal(prev_assignments, cluster_assignments):
            print(f"Converged at epoch {epoch}")
            break
            
        prev_assignments = cluster_assignments.clone()
    
    # 后续处理和可视化...[保持之前的代码不变]
    
    # 转换回CPU numpy数组
    cluster_assignments = cluster_assignments.cpu().numpy()
    
    # 构建重叠数据集
    divided_datasets = []
    total_points = 0
    for cluster_i in range(k):
        cluster_indices = np.where(cluster_assignments[:, cluster_i])[0]
        cluster_data = {key: value[cluster_indices] for key, value in local_dataset.items()}
        divided_datasets.append(cluster_data)
        total_points += len(cluster_indices)
        
        print(f"Cluster {cluster_i}:")
        print(f"  点数: {len(cluster_indices)}")
        if 'rewards' in local_dataset:
            print(f"  总奖励: {sum(cluster_data['rewards']):.2f}")
        print("-"*30)
    
    print(f"原始数据点数: {len(X)}")
    print(f"所有簇总点数(含重叠): {total_points}")
    print(f"平均每个点属于 {total_points/len(X):.2f} 个簇")
    
    # 可视化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    
    # 绘制非重叠点
    for i in range(k):
        pure_mask = cluster_assignments[:, i] & (np.sum(cluster_assignments, axis=1) == 1)
        plt.scatter(X_pca[pure_mask, 0], X_pca[pure_mask, 1], 
                   color=colors[i], label=f'Cluster {i}', alpha=0.6)
    
    # 绘制重叠点（用黑色标记）
    overlap_mask = np.sum(cluster_assignments, axis=1) > 1
    plt.scatter(X_pca[overlap_mask, 0], X_pca[overlap_mask, 1], 
               color='black', marker='x', s=50, label='Overlapping Points')
    
    plt.title(f'Overlapping GMM (k={k}, threshold={variance_reduction_threshold})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    plt.savefig(f"{method}.png")
    for idx, cluster in enumerate(divided_datasets):
        print(f"Cluster {idx}", {len(cluster["observations"])})

    return k, main_labels, [], divided_datasets




def cluster_and_visualize_model(local_dataset, method='kmeans', k=3, device="cpu", n=5):
    """
    对 local_dataset 进行聚类，并可视化结果。

    参数:
        local_dataset (dict): 包含 "observations" 和 "actions" 的字典。
        method (str): 聚类方法，可选值为 'kmeans', 'gmm', 'spectral', 'agglomerative'。 
        k (int): 聚类数，用于指定聚类的类别数量。

    返回:
        best_k (int): 最佳聚类数。
        best_labels (np.array): 每个样本的聚类标签。
        hulls (list): 每个簇的凸包信息（这里为空列表）。
        divided_datasets (list): 划分后的数据集列表。
    """
    # 提取 s 和 a
    s = local_dataset["observations"]  # s 是 7 维
    a = local_dataset["actions"]  # a 是 3 维
    next_s = local_dataset["next_observations"]  # next_s 是 7 维
    # X = s
    X = np.hstack([s, a])

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("分类：")
    # X_scaled_gpu = torch.tensor(X_scaled, device=device)  # 转为 GPU 张量

    if method == 'kmeans':
        # 使用 KMeans 进行聚类
        model = KMeans(n_clusters=k, random_state=42)
    elif method == 'gmm' or method == 'cgmm':
        # 使用高斯混合模型进行聚类
        model = GaussianMixture(n_components=k, random_state=42)
    elif method == 'bgmm':
        # 使用高斯混合模型进行聚类
        model = BayesianGaussianMixture(n_components=k, random_state=42)
    elif method == 'spectral':
        # 使用谱聚类
        model = SpectralClustering(n_clusters=k, random_state=42)
    elif method == 'agglomerative':
        # 使用层次聚类
        model = AgglomerativeClustering(n_clusters=k)
    elif method == 'ward' : 
        # 使用 Ward 方法进行层次聚类 
        model = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
    else:
        raise ValueError("Invalid clustering method. Choose from 'kmeans', 'gmm', 'spectral', 'agglomerative'.")

    labels = model.fit_predict(X_scaled)
    best_k = k
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # labels = model.fit_predict(X_scaled_gpu)
    # best_k = k
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X_scaled_gpu).cpu().numpy()  # 转回 CPU 可视化

    # 可视化降维后的数据
    print("可视化")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Spectral(np.linspace(0, 1, best_k))
    for i in range(best_k):
        # 获取当前簇的 PCA 降维结果
        cluster_pca = X_pca[labels == i]
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], color=colors[i], label=f'Cluster {i + 1}')
    plt.title(f'Clusters ({method}, k={best_k}) - PCA Projection')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    # plt.show()
    plt.savefig(f"{method}.png")
    print(X_pca[:10])
    print(len(X_pca))

    # sys.exit()

    print(f"Best k: {best_k}")

    divided_datasets = []
    for cluster_idx in range(best_k):
        cluster_mask = labels == cluster_idx
        cluster_dataset = {}
        for key, value in local_dataset.items():
            cluster_dataset[key] = value[cluster_mask]
        divided_datasets.append(cluster_dataset)

    for idx, cluster in enumerate(divided_datasets):
        print(f"Cluster {idx}", {len(cluster["rewards"])})
        print(sum(cluster["rewards"]))

    # print(len(cluster["rewards"])) 
    # print()
    # sys.exit()

    if len(local_dataset["rewards"]) < 2e5:
        # print("why")
        # 调用函数进行后续轨迹数据处理
        divided_datasets = process_datasets_subsequent(local_dataset, divided_datasets, device, n)
        # # 调用函数进行前置轨迹数据处理
        divided_datasets = process_datasets_previous(local_dataset, divided_datasets, device, n)
    else:
        if n > 0:       
            device = "cpu"
            divided_datasets = combine_datasets_trajectory(local_dataset, divided_datasets ,device, n)
        # # 调用函数进行后续轨迹数据处理
        # divided_datasets = process_datasets_subsequent(local_dataset, divided_datasets, device, n)
        # # # 调用函数进行前置轨迹数据处理
        # divided_datasets = process_datasets_previous(local_dataset, divided_datasets, device, n)

    # 打印结果
    for idx, cluster in enumerate(divided_datasets):
        print(f"Cluster {idx}", {len(cluster["observations"])})
        # for key, value in cluster.items():
        #     print(f"  {key}: {value}")

    best_labels = labels
    hulls = []  # 因为不需要凸包，所以这里为空列表
    # sys.exit()



    return best_k, best_labels, hulls, divided_datasets


def combine_datasets_trajectory(local_dataset, divided_datasets, device, extend_traj, batch_size=1000):
    # 将 local_dataset 中的数据转换为 PyTorch 张量并移动到 GPU 上
    local_tensors = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in local_dataset.items()}

    # for i in tqdm(range(extend_traj), desc="Processing trajectory data", unit="iter"):
    #     for cluster_dataset in divided_datasets:
    for cluster_dataset in tqdm(divided_datasets, desc="Merging trajectories", unit="dataset"):
        # 将 cluster_dataset 中的 trajectory 和 step 转换为 PyTorch 张量
        cluster_traj = torch.tensor(cluster_dataset["trajectory"], dtype=torch.int64, device=device)
        
        # 找出 cluster_dataset 中存在的所有轨迹索引（去重）
        unique_trajectories = torch.unique(cluster_traj)
        
        # 在 local_dataset 中找出属于这些轨迹的所有数据点
        mask = torch.isin(local_tensors["trajectory"], unique_trajectories)
        
        # 将匹配的轨迹数据添加到 cluster_dataset 中
        for key in cluster_dataset.keys():
            new_data = local_tensors[key][mask].cpu().numpy()
            if cluster_dataset[key].ndim == 1:
                cluster_dataset[key] = np.hstack((cluster_dataset[key], new_data.flatten()))
            else:
                cluster_dataset[key] = np.vstack((cluster_dataset[key], new_data))

        # # 去重：按 ("trajectory", "step") 组合保留唯一数据点
        # if len(cluster_dataset["trajectory"]) > 0:
        #     # 将 trajectory 和 step 组合为结构化数组用于去重
        #     traj_step = np.core.records.fromarrays(
        #         [cluster_dataset["trajectory"], cluster_dataset["step"]],
        #         names="trajectory,step"
        #     )
        #     # 获取唯一索引
        #     _, unique_indices = np.unique(traj_step, return_index=True)
        #     # 对所有字段按唯一索引筛选
        #     for key in cluster_dataset.keys():
        #         if cluster_dataset[key].ndim == 1:
        #             cluster_dataset[key] = cluster_dataset[key][unique_indices]
        #         else:
        #             cluster_dataset[key] = cluster_dataset[key][unique_indices, :]
        # 去重 + 排序
        if len(cluster_dataset["trajectory"]) > 0:
            # 1. 构造结构化数组用于去重
            traj_step = np.core.records.fromarrays(
                [cluster_dataset["trajectory"], cluster_dataset["step"]],
                names="trajectory,step"
            )
            _, unique_indices = np.unique(traj_step, return_index=True)
            
            # 2. 去重后按 (trajectory, step) 排序
            # 获取去重后的 trajectory 和 step
            traj_after_dedup = cluster_dataset["trajectory"][unique_indices]
            step_after_dedup = cluster_dataset["step"][unique_indices]
            
            # 生成排序索引：先按 trajectory 分组，再按 step 升序
            sort_indices = np.lexsort((step_after_dedup, traj_after_dedup))
            
            # 对所有字段应用去重和排序
            for key in cluster_dataset.keys():
                if cluster_dataset[key].ndim == 1:
                    cluster_dataset[key] = cluster_dataset[key][unique_indices][sort_indices]
                else:
                    cluster_dataset[key] = cluster_dataset[key][unique_indices, :][sort_indices, :]


    return divided_datasets


def process_datasets_subsequent(local_dataset, divided_datasets, device, extend_traj, batch_size=1000):
    # 将 local_dataset 中的数据转换为 PyTorch 张量并移动到 GPU 上
    local_tensors = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in local_dataset.items()}

    for i in tqdm(range(extend_traj), desc="Processing subsequent data", unit="iter"):
        for cluster_dataset in divided_datasets:
            # 将 cluster_dataset 中的 observations 转换为 PyTorch 张量并移动到 GPU 上
            cluster_obs = torch.tensor(cluster_dataset["observations"], dtype=torch.float32, device=device)

            # 检查 local_dataset 中相邻观测值是否相等
            obs_local_dataset = torch.all(local_tensors["observations"][1:] == local_tensors["next_observations"][:-1], dim=1)

            # 分批次检查 local_dataset[i] 是否在 cluster_dataset 中
            obs_i_in_cluster = torch.zeros(len(local_tensors["observations"][:-1]), dtype=torch.bool, device=device)
            for j in range(0, len(cluster_obs), batch_size):
                batch_cluster_obs = cluster_obs[j:j + batch_size]
                batch_result = torch.any(torch.all(batch_cluster_obs.unsqueeze(0) == local_tensors["observations"][:-1].unsqueeze(1), dim=2), dim=1)
                obs_i_in_cluster |= batch_result

            # 分批次检查 local_dataset[i+1] 是否不在 cluster_dataset 中
            obs_i_next_in_cluster = torch.zeros(len(local_tensors["observations"][1:]), dtype=torch.bool, device=device)
            for j in range(0, len(cluster_obs), batch_size):
                batch_cluster_obs = cluster_obs[j:j + batch_size]
                batch_result = torch.any(torch.all(batch_cluster_obs.unsqueeze(0) == local_tensors["observations"][1:].unsqueeze(1), dim=2), dim=1)
                obs_i_next_in_cluster |= batch_result

            # 筛选出满足条件的索引
            valid_indices = (obs_local_dataset & obs_i_in_cluster & ~obs_i_next_in_cluster).nonzero(as_tuple=True)[0]

            # 将满足条件的 local_dataset 数据添加到 cluster_dataset 中
            for key in cluster_dataset.keys():
                new_data = local_tensors[key][valid_indices + 1].cpu().numpy()
                if cluster_dataset[key].ndim == 1:
                    cluster_dataset[key] = np.hstack((cluster_dataset[key], new_data.flatten()))
                else:
                    cluster_dataset[key] = np.vstack((cluster_dataset[key], new_data))

    return divided_datasets

def process_datasets_previous(local_dataset, divided_datasets, device, extend_traj, batch_size=1000):
    # 将 local_dataset 中的数据转换为 PyTorch 张量并移动到 GPU 上
    local_tensors = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in local_dataset.items()}

    for i in tqdm(range(extend_traj), desc="Processing previous data", unit="iter"):
        for cluster_dataset in divided_datasets:
            # 将 cluster_dataset 中的 observations 转换为 PyTorch 张量并移动到 GPU 上
            cluster_obs = torch.tensor(cluster_dataset["observations"], dtype=torch.float32, device=device)

            # 检查 local_dataset 中相邻观测值是否相等
            obs_local_dataset = torch.all(local_tensors["observations"][1:] == local_tensors["next_observations"][:-1], dim=1)

            # 分批次检查 local_dataset[i] 是否在 cluster_dataset 中
            obs_i_in_cluster = torch.zeros(len(local_tensors["observations"][1:]), dtype=torch.bool, device=device)
            for j in range(0, len(cluster_obs), batch_size):
                batch_cluster_obs = cluster_obs[j:j + batch_size]
                batch_result = torch.any(torch.all(batch_cluster_obs.unsqueeze(0) == local_tensors["observations"][1:].unsqueeze(1), dim=2), dim=1)
                obs_i_in_cluster |= batch_result

            # 分批次检查 local_dataset[i-1] 是否不在 cluster_dataset 中
            obs_i_prev_in_cluster = torch.zeros(len(local_tensors["observations"][:-1]), dtype=torch.bool, device=device)
            for j in range(0, len(cluster_obs), batch_size):
                batch_cluster_obs = cluster_obs[j:j + batch_size]
                batch_result = torch.any(torch.all(batch_cluster_obs.unsqueeze(0) == local_tensors["observations"][:-1].unsqueeze(1), dim=2), dim=1)
                obs_i_prev_in_cluster |= batch_result

            # 筛选出满足条件的索引
            valid_indices = (obs_local_dataset & obs_i_in_cluster & ~obs_i_prev_in_cluster).nonzero(as_tuple=True)[0]

            # 将满足条件的 local_dataset 数据添加到 cluster_dataset 中
            for key in cluster_dataset.keys():
                new_data = local_tensors[key][valid_indices].cpu().numpy()
                if cluster_dataset[key].ndim == 1:
                    cluster_dataset[key] = np.hstack((cluster_dataset[key], new_data.flatten()))
                else:
                    cluster_dataset[key] = np.vstack((cluster_dataset[key], new_data))

    return divided_datasets






def clustering(X, method='kmeans', k=3, device="cpu", n=5):

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("分类：")
    # X_scaled_gpu = torch.tensor(X_scaled, device=device)  # 转为 GPU 张量

    if method == 'kmeans':
        # 使用 KMeans 进行聚类
        model = KMeans(n_clusters=k, random_state=42)
    elif method == 'gmm':
        # 使用高斯混合模型进行聚类
        model = GaussianMixture(n_components=k, random_state=42)
    elif method == 'bgmm':
        # 使用高斯混合模型进行聚类
        model = BayesianGaussianMixture(n_components=k, random_state=42)
    elif method == 'spectral':
        # 使用谱聚类
        model = SpectralClustering(n_clusters=k, random_state=42)
    elif method == 'agglomerative':
        # 使用层次聚类
        model = AgglomerativeClustering(n_clusters=k)
    elif method == 'ward' : 
        # 使用 Ward 方法进行层次聚类 
        model = AgglomerativeClustering(n_clusters = k, linkage = 'ward')
    else:
        raise ValueError("Invalid clustering method. Choose from 'kmeans', 'gmm', 'spectral', 'agglomerative'.")

    labels = model.fit_predict(X_scaled)
    best_k = k
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    # 可视化降维后的数据
    print("可视化")
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Spectral(np.linspace(0, 1, best_k))
    for i in range(best_k):
        # 获取当前簇的 PCA 降维结果
        cluster_pca = X_pca[labels == i]
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], color=colors[i], label=f'Cluster {i + 1}')
    plt.title(f'Clusters ({method}, k={best_k}) - PCA Projection')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    # plt.show()
    plt.savefig(f"{method}.png")
    print(X_pca[:10])
    print(len(X_pca))


    divided_datasets = []
    for cluster_idx in range(best_k):
        cluster_mask = labels == cluster_idx
        cluster_dataset = {}
        for key, value in local_dataset.items():
            cluster_dataset[key] = value[cluster_mask]
        divided_datasets.append(cluster_dataset)

    for idx, cluster in enumerate(divided_datasets):
        print(f"Cluster {idx}", {len(cluster["rewards"])})
        print(sum(cluster["rewards"]))


    # 打印结果
    for idx, cluster in enumerate(divided_datasets):
        print(f"Cluster {idx}", {len(cluster["observations"])})



    return divided_datasets