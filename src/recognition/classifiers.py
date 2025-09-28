"""
多种分类器实现模块
包含Naive Bayesian、Fisher线性判别、决策树、支撑向量机、最近邻分类器等
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 尝试导入sklearn，如果没有安装则使用简化版本
try:
    import sklearn
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
    print(f"sklearn版本: {sklearn.__version__}")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"警告: sklearn未安装或导入失败 ({e})，将使用简化版本的分类器")

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("警告: PyTorch未安装，CNN分类器不可用")


class BaseClassifier(ABC):
    """分类器基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        
    @abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """训练分类器"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测分类结果"""
        pass
    
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测分类概率"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取分类器信息"""
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'type': self.__class__.__name__
        }


class NaiveBayesianClassifier(BaseClassifier):
    """朴素贝叶斯分类器"""
    
    def __init__(self):
        super().__init__("朴素贝叶斯")
        if SKLEARN_AVAILABLE:
            self.classifier = GaussianNB()
        else:
            self.classifier = None
            self.means = {}
            self.stds = {}
            self.priors = {}
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """训练朴素贝叶斯分类器"""
        if SKLEARN_AVAILABLE:
            self.classifier.fit(features, labels)
        else:
            # 简化版本：计算每个类别的均值和标准差
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                class_features = features[mask]
                self.means[label] = np.mean(class_features, axis=0)
                self.stds[label] = np.std(class_features, axis=0) + 1e-8  # 避免除零
                self.priors[label] = np.sum(mask) / len(labels)
        self.is_trained = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测分类结果"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict(features)
        else:
            # 简化版本：使用高斯似然
            predictions = []
            for feature in features:
                best_label = None
                best_score = -np.inf
                for label in self.means.keys():
                    # 计算高斯似然
                    diff = feature - self.means[label]
                    score = -0.5 * np.sum((diff / self.stds[label]) ** 2) + np.log(self.priors[label])
                    if score > best_score:
                        best_score = score
                        best_label = label
                predictions.append(best_label)
            return np.array(predictions)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测分类概率"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict_proba(features)
        else:
            # 简化版本：计算概率
            probabilities = []
            for feature in features:
                probs = {}
                total_score = 0
                for label in self.means.keys():
                    diff = feature - self.means[label]
                    score = np.exp(-0.5 * np.sum((diff / self.stds[label]) ** 2)) * self.priors[label]
                    probs[label] = score
                    total_score += score
                
                # 归一化
                for label in probs:
                    probs[label] /= total_score
                
                # 转换为数组格式
                prob_array = [probs.get(i, 0) for i in range(10)]
                probabilities.append(prob_array)
            return np.array(probabilities)


class FisherLinearDiscriminantClassifier(BaseClassifier):
    """Fisher线性判别分类器"""
    
    def __init__(self):
        super().__init__("Fisher线性判别")
        if SKLEARN_AVAILABLE:
            self.classifier = LinearDiscriminantAnalysis()
        else:
            self.classifier = None
            self.means = {}
            self.cov_matrix = None
            self.priors = {}
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """训练Fisher线性判别分类器"""
        if SKLEARN_AVAILABLE:
            self.classifier.fit(features, labels)
        else:
            # 简化版本：计算类内和类间散度矩阵
            unique_labels = np.unique(labels)
            n_features = features.shape[1]
            
            # 计算每个类别的均值
            for label in unique_labels:
                mask = labels == label
                self.means[label] = np.mean(features[mask], axis=0)
                self.priors[label] = np.sum(mask) / len(labels)
            
            # 计算总体协方差矩阵（简化版本）
            self.cov_matrix = np.cov(features.T)
        self.is_trained = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测分类结果"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict(features)
        else:
            # 简化版本：使用马氏距离
            predictions = []
            for feature in features:
                best_label = None
                best_score = np.inf
                for label in self.means.keys():
                    diff = feature - self.means[label]
                    # 马氏距离
                    score = np.dot(diff, np.linalg.solve(self.cov_matrix, diff))
                    if score < best_score:
                        best_score = score
                        best_label = label
                predictions.append(best_label)
            return np.array(predictions)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测分类概率"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict_proba(features)
        else:
            # 简化版本：基于距离的概率
            probabilities = []
            for feature in features:
                distances = {}
                for label in self.means.keys():
                    diff = feature - self.means[label]
                    distances[label] = np.dot(diff, np.linalg.solve(self.cov_matrix, diff))
                
                # 转换为概率（距离越小，概率越大）
                max_dist = max(distances.values())
                probs = {}
                for label, dist in distances.items():
                    probs[label] = np.exp(-dist / max_dist) * self.priors[label]
                
                # 归一化
                total_prob = sum(probs.values())
                for label in probs:
                    probs[label] /= total_prob
                
                # 转换为数组格式
                prob_array = [probs.get(i, 0) for i in range(10)]
                probabilities.append(prob_array)
            return np.array(probabilities)


class DecisionTreeClassifier(BaseClassifier):
    """决策树分类器"""
    
    def __init__(self, max_depth: int = 10):
        super().__init__("决策树")
        if SKLEARN_AVAILABLE:
            from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier
            self.classifier = SklearnDecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            self.classifier = None
            self.tree = None
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """训练决策树分类器"""
        if SKLEARN_AVAILABLE:
            self.classifier.fit(features, labels)
        else:
            # 简化版本：使用简单的阈值分割
            self.tree = self._build_simple_tree(features, labels)
        self.is_trained = True
    
    def _build_simple_tree(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        """构建简化的决策树"""
        # 这是一个非常简化的决策树实现
        # 实际应用中应该使用更复杂的算法
        tree = {
            'feature_idx': 0,  # 使用第一个特征
            'threshold': np.median(features[:, 0]),
            'left': None,
            'right': None,
            'label': None
        }
        
        # 简单的二分
        left_mask = features[:, 0] <= tree['threshold']
        right_mask = ~left_mask
        
        if np.sum(left_mask) > 0:
            tree['left'] = {'label': self._get_majority_label(labels[left_mask])}
        if np.sum(right_mask) > 0:
            tree['right'] = {'label': self._get_majority_label(labels[right_mask])}
        
        return tree
    
    def _get_majority_label(self, labels: np.ndarray) -> int:
        """获取多数标签"""
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测分类结果"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict(features)
        else:
            # 简化版本：使用简单树
            predictions = []
            for feature in features:
                if feature[0] <= self.tree['threshold']:
                    pred = self.tree['left']['label'] if self.tree['left'] else 0
                else:
                    pred = self.tree['right']['label'] if self.tree['right'] else 0
                predictions.append(pred)
            return np.array(predictions)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测分类概率"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict_proba(features)
        else:
            # 简化版本：返回硬分类的概率
            predictions = self.predict(features)
            probabilities = []
            for pred in predictions:
                prob_array = [0] * 10
                prob_array[pred] = 1.0
                probabilities.append(prob_array)
            return np.array(probabilities)


class SupportVectorMachineClassifier(BaseClassifier):
    """支持向量机分类器"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        super().__init__("支持向量机")
        if SKLEARN_AVAILABLE:
            self.classifier = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        else:
            self.classifier = None
            self.means = {}
            self.support_vectors = {}
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """训练支持向量机分类器"""
        if SKLEARN_AVAILABLE:
            self.classifier.fit(features, labels)
        else:
            # 简化版本：使用最近邻作为近似
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                self.means[label] = np.mean(features[mask], axis=0)
        self.is_trained = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测分类结果"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict(features)
        else:
            # 简化版本：使用最近邻
            predictions = []
            for feature in features:
                best_label = None
                best_distance = np.inf
                for label, mean in self.means.items():
                    distance = np.linalg.norm(feature - mean)
                    if distance < best_distance:
                        best_distance = distance
                        best_label = label
                predictions.append(best_label)
            return np.array(predictions)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测分类概率"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict_proba(features)
        else:
            # 简化版本：基于距离的概率
            probabilities = []
            for feature in features:
                distances = {}
                for label, mean in self.means.items():
                    distances[label] = np.linalg.norm(feature - mean)
                
                # 转换为概率（距离越小，概率越大）
                max_dist = max(distances.values())
                probs = {}
                for label, dist in distances.items():
                    probs[label] = np.exp(-dist / max_dist)
                
                # 归一化
                total_prob = sum(probs.values())
                for label in probs:
                    probs[label] /= total_prob
                
                # 转换为数组格式
                prob_array = [probs.get(i, 0) for i in range(10)]
                probabilities.append(prob_array)
            return np.array(probabilities)


class KNearestNeighborsClassifier(BaseClassifier):
    """K近邻分类器"""
    
    def __init__(self, k: int = 3):
        super().__init__(f"K近邻 (k={k})")
        self.k = k
        if SKLEARN_AVAILABLE:
            self.classifier = KNeighborsClassifier(n_neighbors=k)
        else:
            self.classifier = None
            self.training_features = None
            self.training_labels = None
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """训练K近邻分类器"""
        if SKLEARN_AVAILABLE:
            self.classifier.fit(features, labels)
        else:
            # 简化版本：存储训练数据
            self.training_features = features.copy()
            self.training_labels = labels.copy()
        self.is_trained = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测分类结果"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict(features)
        else:
            # 简化版本：计算距离并找最近邻
            predictions = []
            for feature in features:
                # 计算与所有训练样本的距离
                distances = np.linalg.norm(self.training_features - feature, axis=1)
                # 找到k个最近邻
                nearest_indices = np.argsort(distances)[:self.k]
                nearest_labels = self.training_labels[nearest_indices]
                # 投票决定类别
                unique, counts = np.unique(nearest_labels, return_counts=True)
                prediction = unique[np.argmax(counts)]
                predictions.append(prediction)
            return np.array(predictions)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测分类概率"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if SKLEARN_AVAILABLE:
            return self.classifier.predict_proba(features)
        else:
            # 简化版本：基于投票的概率
            probabilities = []
            for feature in features:
                distances = np.linalg.norm(self.training_features - feature, axis=1)
                nearest_indices = np.argsort(distances)[:self.k]
                nearest_labels = self.training_labels[nearest_indices]
                
                # 计算每个类别的投票数
                prob_array = [0] * 10
                for label in nearest_labels:
                    prob_array[label] += 1
                
                # 归一化
                total_votes = sum(prob_array)
                if total_votes > 0:
                    prob_array = [p / total_votes for p in prob_array]
                
                probabilities.append(prob_array)
            return np.array(probabilities)


class CNNClassifier(BaseClassifier):
    """卷积神经网络分类器 (PyTorch实现)"""
    
    def __init__(self, input_length=11, num_classes=10, learning_rate=0.001):
        super().__init__("CNN (PyTorch)")
        self.input_length = input_length
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        
        if PYTORCH_AVAILABLE:
            self._build_model()
        else:
            print("警告: PyTorch未安装，CNN分类器不可用")
    
    def _build_model(self):
        """构建CNN模型"""
        class SpeechCNN(nn.Module):
            def __init__(self, input_length, num_classes):
                super(SpeechCNN, self).__init__()
                
                # 卷积层
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                
                # 池化层
                self.pool = nn.MaxPool1d(2)
                
                # 批归一化
                self.bn1 = nn.BatchNorm1d(32)
                self.bn2 = nn.BatchNorm1d(64)
                self.bn3 = nn.BatchNorm1d(128)
                
                # Dropout
                self.dropout = nn.Dropout(0.5)
                
                # 计算全连接层输入大小
                # 经过3次池化，每次长度减半
                fc_input_size = 128 * (input_length // 8)
                
                # 全连接层
                self.fc1 = nn.Linear(fc_input_size, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, num_classes)
                
                # 激活函数
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # x shape: (batch_size, 1, input_length)
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.pool(x)
                
                x = self.relu(self.bn2(self.conv2(x)))
                x = self.pool(x)
                
                x = self.relu(self.bn3(self.conv3(x)))
                x = self.pool(x)
                
                # 展平
                x = x.view(x.size(0), -1)
                
                # 全连接层
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.dropout(self.relu(self.fc2(x)))
                x = self.fc3(x)
                
                return x
        
        self.model = SpeechCNN(self.input_length, self.num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """训练CNN分类器"""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用CNN分类器")
        
        # 数据标准化
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # 转换为PyTorch张量
        X = torch.FloatTensor(features_scaled).unsqueeze(1)  # 添加通道维度
        y = torch.LongTensor(labels)
        
        # 创建数据加载器
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 训练模型
        self.model.train()
        epochs = 50
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_trained = True
        print(f"CNN训练完成 - 输入长度: {self.input_length}, 类别数: {self.num_classes}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测分类结果"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if not PYTORCH_AVAILABLE:
            # 简化版本：随机预测
            return np.random.randint(0, self.num_classes, len(features))
        
        self.model.eval()
        with torch.no_grad():
            features_scaled = self.scaler.transform(features)
            X = torch.FloatTensor(features_scaled).unsqueeze(1)
            outputs = self.model(X)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.numpy()
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """预测分类概率"""
        if not self.is_trained:
            raise ValueError("分类器尚未训练")
        
        if not PYTORCH_AVAILABLE:
            # 简化版本：均匀分布
            return np.ones((len(features), self.num_classes)) / self.num_classes
        
        self.model.eval()
        with torch.no_grad():
            features_scaled = self.scaler.transform(features)
            X = torch.FloatTensor(features_scaled).unsqueeze(1)
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.numpy()
    
    def get_info(self) -> Dict[str, Any]:
        """获取分类器信息"""
        info = {
            'name': self.name,
            'type': 'CNN (PyTorch)',
            'input_length': self.input_length,
            'num_classes': self.num_classes,
            'learning_rate': self.learning_rate,
            'is_trained': self.is_trained
        }
        
        if self.is_trained and PYTORCH_AVAILABLE:
            # 计算模型参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_architecture': str(self.model)
            })
        
        return info


class ClassifierComparison:
    """分类器对比分析类"""
    
    def __init__(self):
        self.classifiers = {
            'naive_bayes': NaiveBayesianClassifier(),
            'fisher_lda': FisherLinearDiscriminantClassifier(),
            'decision_tree': DecisionTreeClassifier(),
            'svm': SupportVectorMachineClassifier(),
            'knn': KNearestNeighborsClassifier(k=3),
            'cnn': CNNClassifier(input_length=11, num_classes=10)
        }
        self.results = {}
    
    def train_all(self, features: np.ndarray, labels: np.ndarray) -> None:
        """训练所有分类器"""
        print("训练所有分类器...")
        print("=" * 60)
        
        for name, classifier in self.classifiers.items():
            try:
                print(f"训练 {classifier.name}...")
                classifier.train(features, labels)
                print(f"✓ {classifier.name} 训练完成")
            except Exception as e:
                print(f"✗ {classifier.name} 训练失败: {e}")
        
        print("=" * 60)
    
    def evaluate_all(self, test_features: np.ndarray, test_labels: np.ndarray) -> Dict[str, Dict]:
        """评估所有分类器"""
        print("评估所有分类器...")
        print("=" * 60)
        
        results = {}
        
        for name, classifier in self.classifiers.items():
            if not classifier.is_trained:
                print(f"跳过 {classifier.name} - 未训练")
                continue
            
            try:
                # 预测
                predictions = classifier.predict(test_features)
                probabilities = classifier.predict_proba(test_features)
                
                # 计算准确率
                accuracy = np.mean(predictions == test_labels)
                
                # 计算每个类别的准确率
                class_accuracies = {}
                for digit in range(10):
                    mask = test_labels == digit
                    if np.sum(mask) > 0:
                        class_acc = np.mean(predictions[mask] == test_labels[mask])
                        class_accuracies[f'digit_{digit}'] = class_acc
                
                # 计算平均置信度
                max_probs = np.max(probabilities, axis=1)
                avg_confidence = np.mean(max_probs)
                
                results[name] = {
                    'classifier_name': classifier.name,
                    'accuracy': accuracy,
                    'class_accuracies': class_accuracies,
                    'avg_confidence': avg_confidence,
                    'predictions': predictions,
                    'probabilities': probabilities
                }
                
                print(f"{classifier.name}: 准确率 = {accuracy:.3f}, 平均置信度 = {avg_confidence:.3f}")
                
            except Exception as e:
                print(f"✗ {classifier.name} 评估失败: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        print("=" * 60)
        return results
    
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        if not self.results:
            return "没有评估结果"
        
        report = []
        report.append("分类器性能对比报告")
        report.append("=" * 80)
        report.append(f"{'分类器':<20} {'准确率':<10} {'平均置信度':<12} {'状态'}")
        report.append("-" * 80)
        
        # 按准确率排序
        sorted_results = sorted(
            [(name, result) for name, result in self.results.items() if 'accuracy' in result],
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        for name, result in sorted_results:
            if 'error' in result:
                report.append(f"{result.get('classifier_name', name):<20} {'N/A':<10} {'N/A':<12} {'错误'}")
            else:
                report.append(f"{result['classifier_name']:<20} "
                            f"{result['accuracy']:<10.3f} "
                            f"{result['avg_confidence']:<12.3f} "
                            f"{'正常'}")
        
        report.append("-" * 80)
        
        # 详细分析
        if sorted_results:
            best_classifier = sorted_results[0]
            report.append(f"\n最佳分类器: {best_classifier[1]['classifier_name']}")
            report.append(f"准确率: {best_classifier[1]['accuracy']:.3f}")
            report.append(f"平均置信度: {best_classifier[1]['avg_confidence']:.3f}")
            
            # 分析每个数字的识别情况
            report.append(f"\n各数字识别准确率:")
            for digit in range(10):
                digit_acc = best_classifier[1]['class_accuracies'].get(f'digit_{digit}', 0)
                report.append(f"  数字 {digit}: {digit_acc:.3f}")
        
        return "\n".join(report)
    
    def get_recommendation(self) -> str:
        """获取分类器选择建议"""
        if not self.results:
            return "没有评估结果，无法提供建议"
        
        # 分析结果
        valid_results = {name: result for name, result in self.results.items() 
                        if 'accuracy' in result}
        
        if not valid_results:
            return "没有有效的评估结果"
        
        # 按准确率排序
        sorted_results = sorted(valid_results.items(), 
                              key=lambda x: x[1]['accuracy'], reverse=True)
        
        best_name, best_result = sorted_results[0]
        best_accuracy = best_result['accuracy']
        
        recommendations = []
        recommendations.append("分类器选择建议:")
        recommendations.append("=" * 40)
        
        # 基于准确率的建议
        if best_accuracy > 0.9:
            recommendations.append(f"✓ 推荐使用: {best_result['classifier_name']}")
            recommendations.append(f"  理由: 准确率最高 ({best_accuracy:.3f})")
        elif best_accuracy > 0.8:
            recommendations.append(f"✓ 推荐使用: {best_result['classifier_name']}")
            recommendations.append(f"  理由: 准确率较高 ({best_accuracy:.3f})")
        else:
            recommendations.append("⚠ 所有分类器准确率较低，建议:")
            recommendations.append("  1. 增加训练样本数量")
            recommendations.append("  2. 改进特征提取方法")
            recommendations.append("  3. 检查数据质量")
        
        # 基于置信度的建议
        high_confidence_classifiers = [
            (name, result) for name, result in valid_results.items()
            if result['avg_confidence'] > 0.8
        ]
        
        if high_confidence_classifiers:
            recommendations.append(f"\n高置信度分类器:")
            for name, result in high_confidence_classifiers:
                recommendations.append(f"  - {result['classifier_name']}: {result['avg_confidence']:.3f}")
        
        # 分类器特点分析
        recommendations.append(f"\n分类器特点分析:")
        for name, result in sorted_results[:3]:  # 只分析前3名
            classifier_name = result['classifier_name']
            accuracy = result['accuracy']
            confidence = result['avg_confidence']
            
            if '朴素贝叶斯' in classifier_name:
                recommendations.append(f"  {classifier_name}: 简单快速，适合小样本")
            elif 'Fisher' in classifier_name:
                recommendations.append(f"  {classifier_name}: 线性分类，计算效率高")
            elif '决策树' in classifier_name:
                recommendations.append(f"  {classifier_name}: 可解释性强，容易过拟合")
            elif '支持向量机' in classifier_name:
                recommendations.append(f"  {classifier_name}: 泛化能力强，适合高维数据")
            elif 'K近邻' in classifier_name:
                recommendations.append(f"  {classifier_name}: 非参数方法，适合复杂边界")
        
        return "\n".join(recommendations)
