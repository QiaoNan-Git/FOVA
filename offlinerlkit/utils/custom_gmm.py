import torch
import numpy as np

class CustomGMM:
    """
    简化版 GMM（仅支持 full 协方差），完全基于 PyTorch 实现，适配 GPU。
    - 迭代路径无 SciPy/NumPy/sklearn 依赖，避免 CPU 瓶颈与 GPU↔CPU 拷贝。
    - 提供三种初始化：init_labels / kmeans（KMeans++ + 少量 Lloyd 迭代，纯 Torch）/ random。
    """

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",  # 'kmeans' | 'random'
        random_state=None,
        init_labels: torch.Tensor = None,
        device=None,
        dtype=torch.float32,
        kmeans_init_steps=5,   # KMeans 迭代次数（初始化用）
    ):
        if covariance_type != "full":
            raise NotImplementedError("Only 'full' covariance is implemented.")

        self.n_components = int(n_components)
        self.covariance_type = covariance_type
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.init_params = init_params
        self.random_state = 42 if random_state is None else int(random_state)

        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        # params
        self.weights_ = None           # [K]
        self.means_ = None             # [K, D]
        self.covariances_ = None       # [K, D, D]
        self.precisions_cholesky_ = None  # [K, D, D]  (chol(Precision))

        # state
        self.converged_ = False
        self.lower_bound_ = -torch.inf
        self.n_iter_ = 0

        # init aids
        self.init_labels = init_labels  # torch.LongTensor[N]
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.kmeans_init_steps = int(kmeans_init_steps)

    # ------------------------- utils -------------------------

    def _to_device(self, x):
        """将输入转到目标 device/dtype。支持 numpy / torch.Tensor。"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.device)
        if x.dtype.is_floating_point:
            x = x.to(self.dtype)
        else:
            # 对标签等整型保持不变
            pass
        return x

    @staticmethod
    def _stable_softmax(logits, dim=-1):
        return torch.softmax(logits, dim=dim)

    # ------------------------- initialization -------------------------

    def _init_with_labels(self, X, labels):
        """根据给定标签生成 one-hot responsibilities。"""
        N = X.shape[0]
        resp = torch.zeros((N, self.n_components), device=X.device, dtype=self.dtype)
        resp[torch.arange(N, device=X.device), labels.long()] = 1.0
        return resp

    def _kmeans_plus_plus(self, X):
        """
        纯 Torch 的 KMeans++ 初始化 + 少量 Lloyd 迭代。
        返回：labels[N], centers[K, D]
        """
        N, D = X.shape
        K = self.n_components

        # 随机选第一个中心
        idx0 = torch.randint(0, N, (1,), device=X.device)
        centers = [X[idx0].squeeze(0)]

        # KMeans++ 选其余中心
        # 距离采用平方欧氏距离
        dist_sq = torch.full((N,), float("inf"), device=X.device, dtype=self.dtype)
        for _ in range(1, K):
            # 更新到最近中心的最小距离
            c = centers[-1]
            dcur = torch.sum((X - c) ** 2, dim=1)
            dist_sq = torch.minimum(dist_sq, dcur)

            # 按距离的概率分布采样下一个中心
            probs = dist_sq / torch.sum(dist_sq)
            next_idx = torch.multinomial(probs, 1)
            centers.append(X[next_idx].squeeze(0))

        centers = torch.stack(centers, dim=0)  # [K, D]

        # 少量 Lloyd 迭代细化中心
        for _ in range(self.kmeans_init_steps):
            # 分配
            # [N, K] = (x - mu)^2 的和
            # 利用 (x - μ)^2 = x^2 + μ^2 - 2 x·μ
            x2 = torch.sum(X * X, dim=1, keepdim=True)        # [N, 1]
            c2 = torch.sum(centers * centers, dim=1, keepdim=True).T  # [1, K]
            distances = x2 + c2 - 2.0 * (X @ centers.T)       # [N, K]
            labels = torch.argmin(distances, dim=1)           # [N]

            # 重新计算中心
            new_centers = torch.zeros_like(centers)
            for k in range(K):
                mask = labels == k
                if mask.any():
                    new_centers[k] = X[mask].mean(dim=0)
                else:
                    # 如果某个簇空了，随机重放
                    ridx = torch.randint(0, N, (1,), device=X.device)
                    new_centers[k] = X[ridx].squeeze(0)
            centers = new_centers

        return labels, centers

    def _random_resp(self, N):
        """随机初始化 responsibilities（Dirichlet-like）。"""
        logits = torch.randn((N, self.n_components), device=self.device, dtype=self.dtype)
        resp = self._stable_softmax(logits, dim=1)
        return resp

    def _initialize_parameters(self, X):
        N, D = X.shape
        X = self._to_device(X)

        if self.init_labels is not None:
            labels = self._to_device(self.init_labels).long()
            resp = self._init_with_labels(X, labels)
        else:
            if self.init_params in ("kmeans", "kmeans++"):
                labels, centers = self._kmeans_plus_plus(X)
                resp = self._init_with_labels(X, labels)
            else:
                resp = self._random_resp(N)

        self._initialize_from_resp(X, resp)

    def _initialize_from_resp(self, X, resp):
        """由 responsibilities 初始化权重、均值、协方差与精度的 Cholesky。"""
        N, D = X.shape
        eps = torch.finfo(self.dtype).eps

        nk = resp.sum(dim=0) + 10.0 * eps            # [K]
        weights = nk / nk.sum()                      # [K]
        means = (resp.T @ X) / nk[:, None]           # [K, D]

        covariances = torch.empty((self.n_components, D, D), device=X.device, dtype=self.dtype)
        I = torch.eye(D, device=X.device, dtype=self.dtype)
        for k in range(self.n_components):
            diff = X - means[k]                      # [N, D]
            wdiff = resp[:, k][:, None] * diff       # [N, D]
            C = (wdiff.T @ diff) / nk[k]             # [D, D]
            C = C + self.reg_covar * I
            covariances[k] = C

        precisions_cholesky = self._compute_precision_cholesky(covariances)

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = precisions_cholesky

    # ------------------------- EM steps -------------------------

    def _compute_precision_cholesky(self, covariances):
        """
        给定协方差矩阵 C，计算精度矩阵的 Cholesky 分解。
        如果协方差矩阵不是正定的，尝试增加正则化项并进行重试。
        使用更稳健的数值方法确保矩阵正定性。
        """
        K, D, _ = covariances.shape
        device, dtype = covariances.device, covariances.dtype
        I = torch.eye(D, device=device, dtype=dtype)

        precisions_cholesky = torch.empty_like(covariances)
        for k in range(K):
            C = covariances[k]
            
            # 首先检查矩阵是否已经正定，通过尝试 Cholesky 分解
            jitter = 1e-6  # 初始扰动
            max_jitter_iter = 10  # 最大尝试次数
            L = None
            
            for i in range(max_jitter_iter):
                try:
                    # 尝试 Cholesky 分解
                    L = torch.linalg.cholesky(C + jitter * I)
                    break
                except RuntimeError:
                    # 增加扰动项
                    jitter *= 10
            else:
                # 如果所有尝试都失败，使用对角矩阵作为 fallback
                # 计算矩阵的特征值来估计所需的最小扰动
                try:
                    eigvals = torch.linalg.eigvalsh(C)
                    min_eigval = eigvals.min()
                    required_jitter = max(1e-6, -min_eigval + 1e-8)
                    L = torch.linalg.cholesky(C + required_jitter * I)
                except:
                    # 最终 fallback: 使用单位矩阵
                    L = torch.linalg.cholesky(I)
            
            # 计算精度矩阵 Σ^{-1}
            precision = torch.cholesky_solve(I, L)
            
            # 确保精度矩阵也是正定的 - 使用更稳健的方法
            try:
                P_chol = torch.linalg.cholesky(precision)
            except RuntimeError:
                # 如果精度矩阵不是正定的，尝试增加扰动
                precision_jitter = 1e-6
                for _ in range(5):  # 尝试5次增加扰动
                    try:
                        precision = precision + precision_jitter * I
                        P_chol = torch.linalg.cholesky(precision)
                        break
                    except RuntimeError:
                        precision_jitter *= 10  # 增加扰动幅度
                else:
                    # 如果所有尝试都失败，使用单位矩阵作为 fallback
                    P_chol = torch.linalg.cholesky(I)
            
            precisions_cholesky[k] = P_chol
        
        return precisions_cholesky

    def _estimate_log_gaussian_prob(self, X):
        """
        计算每个样本对每个分量的 log N(x|μ, Σ)。
        使用 precisions_cholesky（即 chol(Precision)）。
        公式：
          y = (x - μ) @ P_chol
          quad = ||y||^2
          log N = -0.5*(D*log(2π) + quad) + 0.5*log|Precision|
                = -0.5*(D*log(2π) + quad) + sum(log(diag(P_chol)))
        """
        N, D = X.shape
        K = self.n_components
        log_prob = torch.empty((N, K), device=X.device, dtype=self.dtype)

        half_log_2piD = 0.5 * D * torch.log(torch.tensor(2.0 * torch.pi, device=X.device, dtype=self.dtype))

        for k in range(K):
            mean_k = self.means_[k]                         # [D]
            P_chol_k = self.precisions_cholesky_[k]         # [D, D]
            y = (X - mean_k) @ P_chol_k                     # [N, D]
            quad = (y * y).sum(dim=1)                       # [N]
            half_log_det_prec = torch.log(torch.diagonal(P_chol_k)).sum()  # 标量
            log_prob[:, k] = -0.5 * quad - half_log_2piD + half_log_det_prec
        return log_prob  # [N, K]

    def _estimate_weighted_log_prob(self, X):
        """log p(x, z=k) = log π_k + log N(x|μ_k, Σ_k)"""
        log_prob = self._estimate_log_gaussian_prob(X)           # [N, K]
        log_weights = torch.log(self.weights_.clamp_min(1e-32))  # [K]
        return log_prob + log_weights

    def _e_step(self, X):
        """
        返回：
          log_resp: 对数责任度 [N, K]
          log_prob_norm: logsumexp(weighted_log_prob, dim=1) [N]
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)          # [N, K]
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)  # [N, 1]
        log_resp = weighted_log_prob - log_prob_norm                     # [N, K]
        return log_resp, log_prob_norm.squeeze(1)

    def _m_step(self, X, log_resp):
        resp = torch.exp(log_resp)                         # [N, K]
        N, D = X.shape
        eps = torch.finfo(self.dtype).eps

        nk = resp.sum(dim=0) + 10.0 * eps                 # [K]
        weights = nk / nk.sum()                           # [K]
        means = (resp.T @ X) / nk[:, None]                # [K, D]

        covariances = torch.empty((self.n_components, D, D), device=X.device, dtype=self.dtype)
        I = torch.eye(D, device=X.device, dtype=self.dtype)
        for k in range(self.n_components):
            diff = X - means[k]
            wdiff = resp[:, k][:, None] * diff
            C = (wdiff.T @ diff) / nk[k]
            C = C + self.reg_covar * I
            covariances[k] = C

        precisions_cholesky = self._compute_precision_cholesky(covariances)

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.precisions_cholesky_ = precisions_cholesky

    # ------------------------- public API -------------------------

    def fit(self, X):
        """
        使用 EM 拟合。整条路径在 GPU 上，不进行中途 CPU/NumPy/SciPy 调用。
        """
        X = self._to_device(X)
        with torch.no_grad():
            best_lower_bound = -torch.inf
            best_state = None

            # 可选多次随机重启（简单实现：只保留最优一次）
            restarts = max(1, self.n_init)
            for restart in range(restarts):
                self._initialize_parameters(X)

                prev_lower_bound = -torch.inf
                self.converged_ = False

                for n_iter in range(self.max_iter):
                    log_resp, log_prob_norm = self._e_step(X)   # log_resp [N,K], log_prob_norm [N]
                    self._m_step(X, log_resp)

                    lower_bound = log_prob_norm.sum()           # 标准 GMM EM 的对数似然
                    change = (lower_bound - prev_lower_bound).abs()

                    # Debugging print statements
                    print(f"Restart {restart}, Iteration {n_iter}, lower_bound: {lower_bound}, prev_lower_bound: {prev_lower_bound}, change: {change}")

                    if change < self.tol:
                        self.converged_ = True
                        break
                    prev_lower_bound = lower_bound

                self.n_iter_ = n_iter + 1
                self.lower_bound_ = lower_bound

                # 保留最优 - 只更新有限的下界
                if torch.isfinite(lower_bound) and (best_state is None or lower_bound > best_lower_bound):
                    best_lower_bound = lower_bound
                    best_state = (
                        self.weights_.clone(),
                        self.means_.clone(),
                        self.covariances_.clone(),
                        self.precisions_cholesky_.clone(),
                        self.converged_,
                        self.n_iter_,
                        lower_bound,
                    )
                # Debugging print statement
                print(f"Best lower_bound after restart {restart}: {best_lower_bound}")

            # 恢复最优参数 - 如果没有找到有效状态，使用最后一次迭代的状态作为回退
            if best_state is None:
                print("Warning: No valid state found during training. Using last iteration state as fallback.")
                best_state = (
                    self.weights_.clone(),
                    self.means_.clone(),
                    self.covariances_.clone(),
                    self.precisions_cholesky_.clone(),
                    self.converged_,
                    self.n_iter_,
                    self.lower_bound_,
                )
            
            (
                self.weights_,
                self.means_,
                self.covariances_,
                self.precisions_cholesky_,
                self.converged_,
                self.n_iter_,
                self.lower_bound_,
            ) = best_state

        return self


    def score_samples(self, X):
        """
        返回每个样本的对数密度 log p(x)（形状 [N]，numpy）。
        """
        X = self._to_device(X)
        with torch.no_grad():
            wlp = self._estimate_weighted_log_prob(X)        # [N, K]
            logp = torch.logsumexp(wlp, dim=1)               # [N]
        return logp.detach().cpu().numpy()

    def predict_proba(self, X):
        """
        返回责任度（后验概率）[N, K]（torch.Tensor，位于当前 device）。
        """
        X = self._to_device(X)
        with torch.no_grad():
            wlp = self._estimate_weighted_log_prob(X)        # [N, K]
            log_norm = torch.logsumexp(wlp, dim=1, keepdim=True)
            resp = torch.exp(wlp - log_norm)
        return resp

    def predict(self, X):
        """
        返回簇标签（numpy 数组）。
        """
        resp = self.predict_proba(X)                         # [N, K] on device
        labels = torch.argmax(resp, dim=1)
        return labels.detach().cpu().numpy()

    def fit_predict(self, X):
        """
        训练并返回标签。
        """
        self.fit(X)
        return self.predict(X)
