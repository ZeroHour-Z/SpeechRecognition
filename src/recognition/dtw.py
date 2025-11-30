"""
动态时间规整（DTW）算法模块 - 实验三
实现 DTW 算法用于序列对齐和相似度计算
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import sys

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class DTW:
    """动态时间规整（Dynamic Time Warping）算法"""
    
    def __init__(self, distance_metric: str = 'euclidean'):
        """
        初始化 DTW 算法
        
        Args:
            distance_metric: 距离度量方法 ('euclidean', 'manhattan', 'cosine')
        """
        self.distance_metric = distance_metric
    
    def _compute_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算两个向量之间的距离
        
        Args:
            x: 第一个向量
            y: 第二个向量
            
        Returns:
            float: 距离值
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x - y) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x - y))
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            if norm_x == 0 or norm_y == 0:
                return 1.0
            return 1.0 - (dot_product / (norm_x * norm_y))
        else:
            raise ValueError(f"不支持的距离度量方法: {self.distance_metric}")
    
    def compute_distance_matrix(self, sequence1: np.ndarray, sequence2: np.ndarray) -> np.ndarray:
        """
        计算两个序列之间的距离矩阵
        
        Args:
            sequence1: 第一个序列，形状为 (n_frames, n_features)
            sequence2: 第二个序列，形状为 (m_frames, n_features)
            
        Returns:
            np.ndarray: 距离矩阵，形状为 (n_frames, m_frames)
        """
        n = len(sequence1)
        m = len(sequence2)
        distance_matrix = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                distance_matrix[i, j] = self._compute_distance(sequence1[i], sequence2[j])
        
        return distance_matrix
    
    def dtw(self, sequence1: np.ndarray, sequence2: np.ndarray, 
            window: Optional[int] = None) -> Tuple[float, np.ndarray, List[Tuple[int, int]]]:
        """
        计算两个序列之间的 DTW 距离和最优路径
        
        DTW 算法原理：
        1. 计算距离矩阵 D[i, j] = distance(sequence1[i], sequence2[j])
        2. 使用动态规划计算累积距离矩阵
        3. 回溯找到最优对齐路径
        
        累积距离递推公式：
        D_accum[i, j] = D[i, j] + min(
            D_accum[i-1, j],      # 垂直移动
            D_accum[i, j-1],      # 水平移动
            D_accum[i-1, j-1]     # 对角移动
        )
        
        Args:
            sequence1: 第一个序列，形状为 (n_frames, n_features)
            sequence2: 第二个序列，形状为 (m_frames, n_features)
            window: 限制搜索窗口大小（Sakoe-Chiba band），如果为 None 则不限制
            
        Returns:
            Tuple[float, np.ndarray, List[Tuple[int, int]]]:
                - DTW 距离
                - 累积距离矩阵
                - 最优对齐路径 [(i, j), ...]
        """
        n = len(sequence1)
        m = len(sequence2)
        
        # 计算距离矩阵
        distance_matrix = self.compute_distance_matrix(sequence1, sequence2)
        
        # 初始化累积距离矩阵
        accumulated_cost = np.full((n + 1, m + 1), np.inf)
        accumulated_cost[0, 0] = 0.0
        
        # 动态规划计算累积距离
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # 如果使用窗口限制
                if window is not None:
                    if abs(i - j) > window:
                        continue
                
                # 计算三个可能的前驱位置的最小值
                cost = distance_matrix[i - 1, j - 1]
                accumulated_cost[i, j] = cost + min(
                    accumulated_cost[i - 1, j],      # 垂直移动
                    accumulated_cost[i, j - 1],      # 水平移动
                    accumulated_cost[i - 1, j - 1]   # 对角移动
                )
        
        # DTW 距离
        dtw_distance = accumulated_cost[n, m]
        
        # 回溯找到最优路径
        path = []
        i, j = n, m
        
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            
            # 找到前驱位置
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                min_cost = min(
                    accumulated_cost[i - 1, j],
                    accumulated_cost[i, j - 1],
                    accumulated_cost[i - 1, j - 1]
                )
                
                if accumulated_cost[i - 1, j - 1] == min_cost:
                    i -= 1
                    j -= 1
                elif accumulated_cost[i - 1, j] == min_cost:
                    i -= 1
                else:
                    j -= 1
        
        path.reverse()
        
        return dtw_distance, accumulated_cost[1:, 1:], path
    
    def fast_dtw(self, sequence1: np.ndarray, sequence2: np.ndarray, 
                 radius: int = 1) -> Tuple[float, List[Tuple[int, int]]]:
        """
        快速 DTW 算法（使用 Sakoe-Chiba band 限制搜索空间）
        
        Args:
            sequence1: 第一个序列
            sequence2: 第二个序列
            radius: 搜索半径（窗口大小）
            
        Returns:
            Tuple[float, List[Tuple[int, int]]]: (DTW 距离, 最优路径)
        """
        return self.dtw(sequence1, sequence2, window=radius)
    
    def visualize_dtw_path(self, sequence1: np.ndarray, sequence2: np.ndarray,
                          path: List[Tuple[int, int]], 
                          accumulated_cost: Optional[np.ndarray] = None,
                          title: str = "DTW 对齐路径") -> None:
        """
        可视化 DTW 对齐路径
        
        Args:
            sequence1: 第一个序列
            sequence2: 第二个序列
            path: DTW 对齐路径
            accumulated_cost: 累积距离矩阵（可选）
            title: 图表标题
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 累积距离矩阵和路径
        ax1 = axes[0, 0]
        if accumulated_cost is not None:
            im = ax1.imshow(accumulated_cost, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax1, label='累积距离')
        
        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax1.plot(path_y, path_x, 'r-', linewidth=2, label='DTW 路径')
            ax1.scatter(path_y[0], path_x[0], c='green', s=100, marker='o', 
                      label='起点', zorder=5)
            ax1.scatter(path_y[-1], path_x[-1], c='red', s=100, marker='s', 
                      label='终点', zorder=5)
        
        ax1.set_xlabel('序列2 索引')
        ax1.set_ylabel('序列1 索引')
        ax1.set_title('累积距离矩阵和 DTW 路径')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 序列1的特征（显示第一个特征维度）
        ax2 = axes[0, 1]
        if sequence1.ndim == 1:
            ax2.plot(sequence1, 'b-', linewidth=2, label='序列1')
        else:
            ax2.plot(sequence1[:, 0], 'b-', linewidth=2, label='序列1 (特征0)')
        ax2.set_xlabel('帧索引')
        ax2.set_ylabel('特征值')
        ax2.set_title('序列1')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 序列2的特征（显示第一个特征维度）
        ax3 = axes[1, 0]
        if sequence2.ndim == 1:
            ax3.plot(sequence2, 'g-', linewidth=2, label='序列2')
        else:
            ax3.plot(sequence2[:, 0], 'g-', linewidth=2, label='序列2 (特征0)')
        ax3.set_xlabel('帧索引')
        ax3.set_ylabel('特征值')
        ax3.set_title('序列2')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 对齐后的序列对比
        ax4 = axes[1, 1]
        if path:
            # 根据路径提取对齐后的序列
            aligned_seq1 = [sequence1[p[0]] for p in path]
            aligned_seq2 = [sequence2[p[1]] for p in path]
            
            if sequence1.ndim == 1:
                ax4.plot(aligned_seq1, 'b-', linewidth=2, label='序列1 (对齐后)', alpha=0.7)
                ax4.plot(aligned_seq2, 'g-', linewidth=2, label='序列2 (对齐后)', alpha=0.7)
            else:
                ax4.plot([s[0] for s in aligned_seq1], 'b-', linewidth=2, 
                        label='序列1 (对齐后)', alpha=0.7)
                ax4.plot([s[0] for s in aligned_seq2], 'g-', linewidth=2, 
                        label='序列2 (对齐后)', alpha=0.7)
        
        ax4.set_xlabel('对齐后的帧索引')
        ax4.set_ylabel('特征值')
        ax4.set_title('对齐后的序列对比')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def compare_sequences_with_dtw(sequences: List[np.ndarray], 
                               labels: List[str] = None) -> np.ndarray:
    """
    比较多个序列之间的 DTW 距离
    
    Args:
        sequences: 序列列表
        labels: 序列标签（可选）
        
    Returns:
        np.ndarray: DTW 距离矩阵
    """
    n = len(sequences)
    dtw = DTW()
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                distance, _, _ = dtw.dtw(sequences[i], sequences[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    # 可视化距离矩阵
    plt.figure(figsize=(10, 8))
    im = plt.imshow(distance_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='DTW 距离')
    
    if labels:
        plt.xticks(range(n), labels, rotation=45, ha='right')
        plt.yticks(range(n), labels)
    
    plt.title('序列间 DTW 距离矩阵')
    plt.xlabel('序列索引')
    plt.ylabel('序列索引')
    plt.tight_layout()
    plt.show()
    
    return distance_matrix


if __name__ == "__main__":
    # 测试 DTW 算法
    print("测试 DTW 算法...")
    
    # 创建测试序列
    np.random.seed(42)
    seq1 = np.random.randn(20, 5)  # 20帧，每帧5个特征
    seq2 = np.random.randn(25, 5)   # 25帧，每帧5个特征
    
    # 创建 DTW 对象
    dtw = DTW(distance_metric='euclidean')
    
    # 计算 DTW 距离
    distance, accumulated_cost, path = dtw.dtw(seq1, seq2)
    
    print(f"序列1长度: {len(seq1)}")
    print(f"序列2长度: {len(seq2)}")
    print(f"DTW 距离: {distance:.4f}")
    print(f"路径长度: {len(path)}")
    
    # 可视化
    dtw.visualize_dtw_path(seq1, seq2, path, accumulated_cost, 
                          "DTW 算法测试")

