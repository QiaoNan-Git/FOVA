import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def cluster_and_visualize(local_dataset, max_k=10):
    """
    对 local_dataset 进行聚类，并可视化结果。
    
    参数:
        local_dataset (dict): 包含 "observations" 和 "actions" 的字典。
        max_k (int): 最大聚类数，用于动态确定 k 的取值范围。
    
    返回:
        best_k (int): 最佳聚类数。
        best_labels (np.array): 每个样本的聚类标签。
        hulls (list): 每个簇的凸包信息。
    """
    # 提取 s 和 a
    s = local_dataset["observations"]  # s 是 7 维
    a = local_dataset["actions"]       # a 是 3 维
    # X = np.hstack([s, a])  # 合并为 10 维数据
    X = s
    # print(s.shape)
    # print(a.shape)
    # print(X.shape)
    # sys.exit()

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 动态确定 k 的取值范围
    def determine_k_range(X, max_k):
        silhouette_scores = []
        inertia_values = []
        range_k = range(2, max_k + 1)
        
        for k in range_k:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            inertia_values.append(kmeans.inertia_)
        
        # 基于轮廓系数选择 k 的范围
        best_k_silhouette = range_k[np.argmax(silhouette_scores)]
        k_range_silhouette = range(max(2, best_k_silhouette - 2), min(max_k, best_k_silhouette + 3))
        
        # 基于肘部法则选择 k 的范围
        best_k_elbow = range_k[np.argmin(np.diff(inertia_values, 2)) + 1]  # 二阶差分最小值
        k_range_elbow = range(max(2, best_k_elbow - 2), min(max_k, best_k_elbow + 3))
        
        # 综合两种方法
        k_range = sorted(set(k_range_silhouette).union(set(k_range_elbow)))
        return list(k_range)

    range_n_clusters = determine_k_range(X_scaled, max_k)
    print(f"Dynamic k range: {range_n_clusters}")

    # 寻找最佳 k 值（基于凸包间距）
    def compute_average_min_distance(X, labels, k):
        min_distances = []
        for i in range(k):
            for j in range(i + 1, k):
                points_i = X[labels == i]
                points_j = X[labels == j]
                if len(points_i) == 0 or len(points_j) == 0:
                    continue
                dist_matrix = distance.cdist(points_i, points_j, 'euclidean')
                min_dist = np.min(dist_matrix)
                min_distances.append(min_dist)
        return np.mean(min_distances) if min_distances else 0

    best_avg_distance = -1
    best_k = 2
    best_labels = None

    # 给寻找最佳 k 值的循环添加进度条
    for k in tqdm(range_n_clusters, desc="Finding best k"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        avg_distance = compute_average_min_distance(X_scaled, labels, k)
        if avg_distance > best_avg_distance:
            best_avg_distance = avg_distance
            best_k = k
            best_labels = labels

    # 使用最佳 k 值聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # 计算凸包
    hulls = []
    # 给计算凸包的循环添加进度条
    for i in tqdm(range(best_k), desc="Computing convex hulls"):
        cluster_points = X_scaled[labels == i]
        if len(cluster_points) >= cluster_points.shape[1] + 1:
            hulls.append((cluster_points, None))
            # hull = ConvexHull(cluster_points)
            # hulls.append((cluster_points, hull))
        else:
            hulls.append((cluster_points, None))

    # 降维可视化（使用 PCA 降到 2 维）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 可视化降维后的数据
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Spectral(np.linspace(0, 1, best_k))
    for i, (cluster_points, hull) in enumerate(hulls):
        # 获取当前簇的 PCA 降维结果
        cluster_pca = X_pca[labels == i]
        plt.scatter(cluster_pca[:, 0], cluster_pca[:, 1], color=colors[i], label=f'Cluster {i + 1}')
        if hull is not None:
            # 计算凸包在 PCA 空间中的投影
            hull_points = cluster_points[hull.vertices]
            hull_pca = pca.transform(hull_points)
            hull_pca = np.vstack((hull_pca, hull_pca[0]))  # 闭合凸包
            plt.plot(hull_pca[:, 0], hull_pca[:, 1], color=colors[i], linewidth=2, linestyle='--')
    plt.title(f'Clusters with Convex Hulls (k={best_k}) - PCA Projection')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()
    plt.savefig("kmeans.png")

    print(f"Best k: {best_k}, Average Minimum Distance: {best_avg_distance:.4f}")

    divided_datasets = []
    for cluster_idx in range(best_k):
        cluster_mask = best_labels == cluster_idx
        cluster_dataset = {}
        for key, value in local_dataset.items():
            cluster_dataset[key] = value[cluster_mask]
        divided_datasets.append(cluster_dataset)

    return best_k, best_labels, hulls, divided_datasets
    # return best_k, best_labels, hulls