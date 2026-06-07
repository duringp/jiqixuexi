"""
===============================================================================
多模型比较器 - 糖尿病视网膜病变(DR)分类模型训练与比较系统
包含高级特征提取和特征工程功能（修复版）
===============================================================================
"""

# =============================================================================
# 导入必要的库
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, roc_curve, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import joblib
import json
from scipy import stats
from scipy.fft import fft, ifft
from scipy.signal import find_peaks, welch

# 尝试导入小波变换（可选）
try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("警告: pywt未安装，小波特征提取功能不可用。安装命令: pip install PyWavelets")

# 忽略警告信息
warnings.filterwarnings('ignore')

# =============================================================================
# 设置matplotlib中文字体
# =============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FeatureExtractor:
    """
    高级特征提取器（修复版）

    提供多种特征提取和工程方法：
    1. 统计特征（均值、方差、偏度、峰度等）
    2. 频域特征（FFT、功率谱密度）
    3. 小波特征（小波变换系数）
    4. 非线性特征（熵、复杂度等）
    5. 交互特征（特征乘积、比值等）
    6. 多项式特征（高阶组合）
    7. 降维特征（PCA、LDA、ICA）
    """

    def __init__(self):
        """初始化特征提取器"""
        self.feature_names = []
        self.fitted_transformers = {}

    def extract_statistical_features(self, X, feature_names=None):
        """
        提取统计特征（修复版）

        为每个样本计算统计特征

        参数:
        -----------
        X : array
            原始特征矩阵，形状 (n_samples, n_features)
        feature_names : list
            原始特征名称

        返回:
        -----------
        X_stats : array
            统计特征矩阵，形状 (n_samples, n_stat_features)
        new_feature_names : list
            新特征名称列表
        """
        print("  提取统计特征...")

        X = np.array(X)
        n_samples, n_features = X.shape

        # 存储每个样本的统计特征
        stats_features = []
        base_names = []

        # 对每个样本计算统计特征
        for i in range(n_samples):
            sample = X[i, :]
            sample_stats = []

            # 基本统计量
            sample_stats.append(np.mean(sample))  # 均值
            if i == 0:
                base_names.append('mean_of_features')

            sample_stats.append(np.std(sample))  # 标准差
            if i == 0:
                base_names.append('std_of_features')

            sample_stats.append(np.median(sample))  # 中位数
            if i == 0:
                base_names.append('median_of_features')

            sample_stats.append(np.min(sample))  # 最小值
            if i == 0:
                base_names.append('min_of_features')

            sample_stats.append(np.max(sample))  # 最大值
            if i == 0:
                base_names.append('max_of_features')

            sample_stats.append(np.ptp(sample))  # 极差
            if i == 0:
                base_names.append('range_of_features')

            # 偏度和峰度
            sample_stats.append(stats.skew(sample))  # 偏度
            if i == 0:
                base_names.append('skew_of_features')

            sample_stats.append(stats.kurtosis(sample))  # 峰度
            if i == 0:
                base_names.append('kurtosis_of_features')

            # 百分位数
            for p in [25, 75]:
                sample_stats.append(np.percentile(sample, p))
                if i == 0:
                    base_names.append(f'percentile_{p}_of_features')

            # 四分位距
            q75, q25 = np.percentile(sample, [75, 25])
            sample_stats.append(q75 - q25)
            if i == 0:
                base_names.append('iqr_of_features')

            # 变异系数
            if np.mean(sample) != 0:
                sample_stats.append(np.std(sample) / np.abs(np.mean(sample)))
            else:
                sample_stats.append(0)
            if i == 0:
                base_names.append('cv_of_features')

            stats_features.append(sample_stats)

        X_stats = np.array(stats_features)

        print(f"    生成 {X_stats.shape[1]} 个统计特征")

        return X_stats, base_names

    def extract_frequency_features(self, X, feature_names=None):
        """
        提取频域特征（修复版）

        参数:
        -----------
        X : array
            原始特征矩阵，形状 (n_samples, n_features)
        feature_names : list
            原始特征名称

        返回:
        -----------
        X_freq : array
            频域特征矩阵，形状 (n_samples, n_freq_features)
        new_feature_names : list
            新特征名称列表
        """
        print("  提取频域特征...")

        X = np.array(X)
        n_samples, n_features = X.shape

        freq_features = []
        base_names = []

        for i in range(n_samples):
            sample = X[i, :]
            sample_freq = []

            # FFT变换
            fft_vals = np.fft.fft(sample)
            fft_magnitude = np.abs(fft_vals[:len(fft_vals) // 2])  # 取一半

            # 频域特征
            if len(fft_magnitude) > 0:
                sample_freq.append(np.mean(fft_magnitude))  # 平均幅度
                if i == 0:
                    base_names.append('fft_mean')

                sample_freq.append(np.std(fft_magnitude))  # 幅度标准差
                if i == 0:
                    base_names.append('fft_std')

                sample_freq.append(np.max(fft_magnitude))  # 最大幅度
                if i == 0:
                    base_names.append('fft_max')

                # 主频位置
                dominant_freq = np.argmax(fft_magnitude)
                sample_freq.append(dominant_freq / len(fft_magnitude) if len(fft_magnitude) > 0 else 0)
                if i == 0:
                    base_names.append('dominant_freq_ratio')

                # 频谱质心
                if np.sum(fft_magnitude) > 0:
                    centroid = np.sum(np.arange(len(fft_magnitude)) * fft_magnitude) / np.sum(fft_magnitude)
                    sample_freq.append(centroid / len(fft_magnitude) if len(fft_magnitude) > 0 else 0)
                else:
                    sample_freq.append(0)
                if i == 0:
                    base_names.append('spectral_centroid_ratio')

                # 频谱熵
                psd = fft_magnitude / (np.sum(fft_magnitude) + 1e-8)
                spectral_entropy = -np.sum(psd * np.log(psd + 1e-8))
                sample_freq.append(spectral_entropy)
                if i == 0:
                    base_names.append('spectral_entropy')
            else:
                for _ in range(6):
                    sample_freq.append(0)
                    if i == 0:
                        base_names.append(f'freq_empty_{_}')

            freq_features.append(sample_freq)

        X_freq = np.array(freq_features)

        print(f"    生成 {X_freq.shape[1]} 个频域特征")

        return X_freq, base_names

    def extract_wavelet_features(self, X, feature_names=None, wavelet='db4', level=3):
        """
        提取小波特征（修复版）

        参数:
        -----------
        X : array
            原始特征矩阵，形状 (n_samples, n_features)
        feature_names : list
            原始特征名称
        wavelet : str
            小波基函数
        level : int
            分解层数

        返回:
        -----------
        X_wavelet : array
            小波特征矩阵，形状 (n_samples, n_wavelet_features)
        new_feature_names : list
            新特征名称列表
        """
        if not HAS_PYWT:
            print("  小波特征提取跳过（pywt未安装）")
            return np.array([]).reshape(X.shape[0], 0), []

        print(f"  提取小波特征 (wavelet={wavelet}, level={level})...")

        X = np.array(X)
        n_samples, n_features = X.shape

        wavelet_features = []
        base_names = []

        for i in range(n_samples):
            sample = X[i, :]
            sample_wavelet = []

            try:
                # 小波分解
                coeffs = pywt.wavedec(sample, wavelet, level=min(level, 3))

                # 提取各层系数统计特征
                for j, coeff in enumerate(coeffs):
                    if len(coeff) > 0:
                        sample_wavelet.append(np.mean(coeff))
                        if i == 0:
                            base_names.append(f'wavelet_mean_level{j}')

                        sample_wavelet.append(np.std(coeff))
                        if i == 0:
                            base_names.append(f'wavelet_std_level{j}')

                        sample_wavelet.append(np.max(np.abs(coeff)))
                        if i == 0:
                            base_names.append(f'wavelet_max_level{j}')

                        # 能量
                        energy = np.sum(coeff ** 2) / (len(coeff) + 1e-8)
                        sample_wavelet.append(energy)
                        if i == 0:
                            base_names.append(f'wavelet_energy_level{j}')
                    else:
                        for _ in range(4):
                            sample_wavelet.append(0)
                            if i == 0:
                                base_names.append(f'wavelet_empty_level{j}')
            except Exception as e:
                # 如果失败，填充0
                for _ in range(level * 4):
                    sample_wavelet.append(0)
                    if i == 0:
                        base_names.append('wavelet_error')

            wavelet_features.append(sample_wavelet)

        if wavelet_features:
            X_wavelet = np.array(wavelet_features)
        else:
            X_wavelet = np.array([]).reshape(n_samples, 0)

        print(f"    生成 {X_wavelet.shape[1]} 个小波特征")

        return X_wavelet, base_names

    def extract_interaction_features(self, X, feature_names=None, top_k=20):
        """
        提取交互特征（修复版）

        参数:
        -----------
        X : array
            原始特征矩阵，形状 (n_samples, n_features)
        feature_names : list
            原始特征名称
        top_k : int
            选择最重要的k个特征进行交互

        返回:
        -----------
        X_interaction : array
            交互特征矩阵，形状 (n_samples, n_interaction_features)
        new_feature_names : list
            新特征名称列表
        """
        print("  提取交互特征...")

        X = np.array(X)
        n_samples, n_features = X.shape

        # 选择最重要的特征（基于方差）
        variances = np.var(X, axis=0)
        top_k = min(top_k, n_features)
        top_indices = np.argsort(variances)[-top_k:]

        interaction_features = []
        base_names = []

        # 生成交互特征
        for i in range(len(top_indices)):
            for j in range(i + 1, min(i + 3, len(top_indices))):  # 限制数量
                idx_i = top_indices[i]
                idx_j = top_indices[j]

                # 乘积特征
                product = X[:, idx_i] * X[:, idx_j]
                interaction_features.append(product.reshape(-1, 1))
                base_names.append(f'interaction_product_{idx_i}_{idx_j}')

                # 比值特征
                ratio = X[:, idx_i] / (X[:, idx_j] + 1e-8)
                interaction_features.append(ratio.reshape(-1, 1))
                base_names.append(f'interaction_ratio_{idx_i}_{idx_j}')

                # 差值特征
                diff = X[:, idx_i] - X[:, idx_j]
                interaction_features.append(diff.reshape(-1, 1))
                base_names.append(f'interaction_diff_{idx_i}_{idx_j}')

        if interaction_features:
            X_interaction = np.hstack(interaction_features)
        else:
            X_interaction = np.array([]).reshape(n_samples, 0)

        print(f"    生成 {X_interaction.shape[1]} 个交互特征")

        return X_interaction, base_names

    def extract_nonlinear_features(self, X, feature_names=None):
        """
        提取非线性特征（修复版）

        参数:
        -----------
        X : array
            原始特征矩阵，形状 (n_samples, n_features)
        feature_names : list
            原始特征名称

        返回:
        -----------
        X_nonlinear : array
            非线性特征矩阵，形状 (n_samples, n_nonlinear_features)
        new_feature_names : list
            新特征名称列表
        """
        print("  提取非线性特征...")

        X = np.array(X)
        n_samples, n_features = X.shape

        nonlinear_features = []
        base_names = []

        for i in range(n_samples):
            sample = X[i, :]
            sample_nonlinear = []

            # 复杂度特征
            complexity = np.std(sample) * (1 + np.abs(stats.skew(sample)))
            sample_nonlinear.append(complexity)
            if i == 0:
                base_names.append('signal_complexity')

            # 排列熵（简化版）
            try:
                # 使用自相关作为熵的代理
                sample_centered = sample - np.mean(sample)
                if len(sample) > 1:
                    autocorr = np.correlate(sample_centered, sample_centered, mode='full')
                    autocorr = autocorr[len(autocorr) // 2:len(autocorr) // 2 + min(10, len(sample) // 2)]
                    autocorr_norm = np.abs(autocorr[1:] / (autocorr[0] + 1e-8))
                    autocorr_norm = autocorr_norm[:min(5, len(autocorr_norm))]
                    permutation_entropy = -np.sum(autocorr_norm * np.log(autocorr_norm + 1e-8))
                else:
                    permutation_entropy = 0
                sample_nonlinear.append(permutation_entropy)
                if i == 0:
                    base_names.append('permutation_entropy')
            except:
                sample_nonlinear.append(0)
                if i == 0:
                    base_names.append('permutation_entropy_error')

            # 递归量分析
            try:
                rqa = np.sum(np.abs(np.diff(sample))) / (len(sample) - 1) if len(sample) > 1 else 0
                sample_nonlinear.append(rqa)
                if i == 0:
                    base_names.append('recurrence_quantification')
            except:
                sample_nonlinear.append(0)
                if i == 0:
                    base_names.append('recurrence_error')

            # Hurst指数（长程相关性）- 简化版
            try:
                lags = range(2, min(10, len(sample) // 2))
                if len(lags) > 1:
                    tau = []
                    for lag in lags:
                        if len(sample) > lag:
                            diff = np.diff(sample[::lag])
                            if len(diff) > 0:
                                tau.append(np.std(diff))
                    if len(tau) > 1 and len(lags) == len(tau):
                        hurst = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
                    else:
                        hurst = 0.5
                else:
                    hurst = 0.5
                sample_nonlinear.append(min(max(hurst, 0), 1))  # 限制在[0,1]范围
                if i == 0:
                    base_names.append('hurst_exponent')
            except:
                sample_nonlinear.append(0.5)
                if i == 0:
                    base_names.append('hurst_error')

            nonlinear_features.append(sample_nonlinear)

        X_nonlinear = np.array(nonlinear_features)

        print(f"    生成 {X_nonlinear.shape[1]} 个非线性特征")

        return X_nonlinear, base_names

    def extract_polynomial_features(self, X, degree=2, include_bias=False):
        """
        提取多项式特征

        参数:
        -----------
        X : array
            原始特征矩阵
        degree : int
            多项式次数
        include_bias : bool
            是否包含偏置项

        返回:
        -----------
        X_poly : array
            多项式特征矩阵
        poly : PolynomialFeatures
            多项式转换器
        """
        print(f"  提取多项式特征 (degree={degree})...")

        original_shape = X.shape[1]
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)

        print(f"    原始特征数: {original_shape}, 多项式特征数: {X_poly.shape[1]}")

        return X_poly, poly

    def reduce_dimension(self, X, method='pca', n_components=0.95, random_state=42):
        """
        降维特征提取

        参数:
        -----------
        X : array
            特征矩阵
        method : str
            降维方法: 'pca', 'ica', 'svd'
        n_components : int or float
            降维后的维度数或方差保留比例
        random_state : int
            随机种子

        返回:
        -----------
        X_reduced : array
            降维后的特征矩阵
        transformer : object
            降维转换器
        """
        print(f"  降维处理 (method={method})...")

        if method == 'pca':
            transformer = PCA(n_components=n_components, random_state=random_state)
        elif method == 'ica':
            transformer = FastICA(n_components=n_components, random_state=random_state)
        elif method == 'svd':
            transformer = TruncatedSVD(n_components=n_components, random_state=random_state)
        else:
            print(f"    不支持的降维方法: {method}")
            return X, None

        X_reduced = transformer.fit_transform(X)

        if hasattr(transformer, 'explained_variance_ratio_'):
            explained_var = transformer.explained_variance_ratio_
            cumsum_var = np.cumsum(explained_var)
            print(f"    保留 {n_components} 个主成分，解释方差比例: {cumsum_var[-1]:.2%}")

        print(f"    降维后特征数: {X_reduced.shape[1]}")

        return X_reduced, transformer


class DRFeatureEngineering:
    """
    DR专项特征工程类

    专门为糖尿病视网膜病变检测设计的特征工程方法
    """

    def __init__(self):
        """初始化DR特征工程器"""
        self.feature_extractor = FeatureExtractor()

    def create_clinical_features(self, X, feature_names=None):
        """
        创建临床相关特征

        参数:
        -----------
        X : array
            原始特征矩阵，形状 (n_samples, n_features)
        feature_names : list
            原始特征名称

        返回:
        -----------
        X_clinical : array
            临床特征矩阵，形状 (n_samples, n_clinical_features)
        clinical_names : list
            临床特征名称列表
        """
        print("\n  创建临床相关特征...")

        X = np.array(X)
        n_samples, n_features = X.shape

        clinical_features = []
        clinical_names = []

        # 1. 风险比率特征（基于前几个特征）
        for i in range(min(n_features, 5)):
            for j in range(i + 1, min(i + 3, n_features)):
                ratio = (X[:, i] + 1e-8) / (X[:, j] + 1e-8)
                clinical_features.append(ratio.reshape(-1, 1))
                clinical_names.append(f'clinical_ratio_{i}_{j}')

                diff = np.abs(X[:, i] - X[:, j])
                clinical_features.append(diff.reshape(-1, 1))
                clinical_names.append(f'clinical_diff_{i}_{j}')

        # 2. 综合风险指数
        if n_features >= 3:
            # 假设前几个特征最重要
            weights = np.exp(-np.arange(min(3, n_features)))  # 指数衰减权重
            weights = weights / np.sum(weights)
            risk_index = np.sum(X[:, :min(3, n_features)] * weights[:min(3, n_features)], axis=1)
            clinical_features.append(risk_index.reshape(-1, 1))
            clinical_names.append('comprehensive_risk_index')

        # 3. 异常检测特征（基于Z-score）
        for i in range(min(n_features, 10)):
            col = X[:, i]
            mean_val = np.mean(col)
            std_val = np.std(col)
            z_scores = (col - mean_val) / (std_val + 1e-8)
            anomaly_score = np.tanh(np.abs(z_scores))  # 转换为[0,1]范围
            clinical_features.append(anomaly_score.reshape(-1, 1))
            clinical_names.append(f'anomaly_score_feature_{i}')

        if clinical_features:
            X_clinical = np.hstack(clinical_features)
        else:
            X_clinical = np.array([]).reshape(n_samples, 0)

        print(f"    生成 {X_clinical.shape[1]} 个临床特征")

        return X_clinical, clinical_names

    def create_texture_features(self, X, feature_names=None):
        """
        创建纹理特征（模拟图像纹理分析）

        参数:
        -----------
        X : array
            原始特征矩阵，形状 (n_samples, n_features)
        feature_names : list
            原始特征名称

        返回:
        -----------
        X_texture : array
            纹理特征矩阵，形状 (n_samples, n_texture_features)
        texture_names : list
            纹理特征名称列表
        """
        print("  创建纹理特征...")

        X = np.array(X)
        n_samples, n_features = X.shape

        texture_features = []
        texture_names = []

        # 模拟局部二值模式（LBP）特征
        for i in range(min(n_features - 1, 20)):
            local_diff = X[:, i + 1] - X[:, i]
            binary_pattern = (local_diff > 0).astype(float)
            texture_features.append(binary_pattern.reshape(-1, 1))
            texture_names.append(f'lbp_pattern_{i}')

            # 局部梯度
            gradient = np.abs(local_diff) / (np.abs(X[:, i]) + 1e-8)
            gradient = np.clip(gradient, 0, 10)  # 限制范围
            texture_features.append(gradient.reshape(-1, 1))
            texture_names.append(f'local_gradient_{i}')

        # 模拟灰度共生矩阵（GLCM）特征
        if n_features >= 4:
            for i in range(min(n_features, 10)):
                for j in range(i + 1, min(i + 3, n_features)):
                    # 对比度
                    contrast = (X[:, i] - X[:, j]) ** 2
                    contrast = contrast / (np.std(contrast) + 1e-8)  # 归一化
                    texture_features.append(contrast.reshape(-1, 1))
                    texture_names.append(f'glcm_contrast_{i}_{j}')

        if texture_features:
            X_texture = np.hstack(texture_features)
        else:
            X_texture = np.array([]).reshape(n_samples, 0)

        print(f"    生成 {X_texture.shape[1]} 个纹理特征")

        return X_texture, texture_names


class ModelComparator:
    """
    多模型比较器类（增强版，包含特征提取）
    """

    def __init__(self, data_source='synthetic',
                 # 特征工程参数
                 enable_stat_features=True,
                 enable_freq_features=False,
                 enable_wavelet_features=False,
                 enable_interaction_features=True,
                 enable_nonlinear_features=False,
                 enable_poly_features=False,
                 poly_degree=2,
                 enable_dim_reduction=False,
                 reduction_method='pca',
                 reduction_components=0.95,
                 enable_clinical_features=True,
                 enable_texture_features=False,
                 # DR数据集参数
                 dr_dataset='messidor',
                 dr_feature_file=None,
                 normal_feature_file=None,
                 # 自定义数据参数
                 X=None, y=None, feature_names=None,
                 # 其他参数
                 test_size=0.25, random_state=42,
                 n_iterations=5, models_dir='saved_models_dr_enhanced'):
        """
        初始化比较器
        """
        # 存储配置参数
        self.data_source = data_source
        self.dr_dataset = dr_dataset
        self.test_size = test_size
        self.random_state = random_state
        self.n_iterations = n_iterations
        self.models_dir = models_dir

        # 特征工程配置
        self.enable_stat_features = enable_stat_features
        self.enable_freq_features = enable_freq_features
        self.enable_wavelet_features = enable_wavelet_features
        self.enable_interaction_features = enable_interaction_features
        self.enable_nonlinear_features = enable_nonlinear_features
        self.enable_poly_features = enable_poly_features
        self.poly_degree = poly_degree
        self.enable_dim_reduction = enable_dim_reduction
        self.reduction_method = reduction_method
        self.reduction_components = reduction_components
        self.enable_clinical_features = enable_clinical_features
        self.enable_texture_features = enable_texture_features

        # 文件路径参数
        self.dr_feature_file = dr_feature_file
        self.normal_feature_file = normal_feature_file

        # 直接数据参数
        self.direct_X = X
        self.direct_y = y
        self.direct_feature_names = feature_names

        # 初始化特征提取器
        self.feature_extractor = FeatureExtractor()
        self.dr_feature_engineer = DRFeatureEngineering()

        # 创建模型保存目录
        self._create_models_directory()

        # 数据存储变量
        self.X_raw = None
        self.y_raw = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.X_train_enhanced = None
        self.X_test_enhanced = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.original_feature_names = None
        self.feature_selector = None
        self.poly_transformer = None
        self.dim_reducer = None

        # 存储模型和结果
        self.models = {}
        self.results = {}
        self.cv_scores = {}
        self.model_metadata = {}

        # 特征工程统计
        self.feature_engineering_stats = {}

    def _create_models_directory(self):
        """创建模型保存目录"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"✓ 创建模型保存目录: {self.models_dir}")

        self.model_subdirs = {
            'SVM': os.path.join(self.models_dir, 'SVM'),
            'Logistic Regression': os.path.join(self.models_dir, 'Logistic_Regression'),
            'Decision Tree': os.path.join(self.models_dir, 'Decision_Tree'),
            'Random Forest': os.path.join(self.models_dir, 'Random_Forest'),
            'Gradient Boosting': os.path.join(self.models_dir, 'Gradient_Boosting'),
            'Neural Network': os.path.join(self.models_dir, 'Neural_Network'),
            'Best Model': os.path.join(self.models_dir, 'Best_Model')
        }

        for subdir in self.model_subdirs.values():
            if not os.path.exists(subdir):
                os.makedirs(subdir)

    def load_dr_data(self):
        """加载DR数据"""
        print("=" * 80)
        print(f"加载糖尿病视网膜病变(DR)数据集: {self.dr_dataset}")
        print("=" * 80)

        if self.data_source == 'synthetic':
            # 使用合成数据
            print("生成合成DR数据...")
            from sklearn.datasets import make_classification

            X, y = make_classification(
                n_samples=1500,
                n_features=50,
                n_informative=25,
                n_redundant=10,
                n_clusters_per_class=1,
                flip_y=0.05,
                random_state=self.random_state
            )
            self.original_feature_names = [f"feature_{i + 1}" for i in range(X.shape[1])]

        elif self.data_source == 'local':
            # 使用本地特征文件
            if not self.dr_feature_file or not self.normal_feature_file:
                raise ValueError("使用local数据源时必须提供dr_feature_file和normal_feature_file")

            dr_df = pd.read_csv(self.dr_feature_file)
            normal_df = pd.read_csv(self.normal_feature_file)

            dr_df['label'] = 1
            normal_df['label'] = 0

            common_cols = set(dr_df.columns) & set(normal_df.columns)
            dr_df = dr_df[list(common_cols)]
            normal_df = normal_df[list(common_cols)]

            df = pd.concat([dr_df, normal_df], ignore_index=True)
            self.original_feature_names = [col for col in df.columns if col != 'label']
            X = df[self.original_feature_names].values.astype(float)
            y = df['label'].values

        elif self.data_source == 'direct':
            if self.direct_X is None or self.direct_y is None:
                raise ValueError("使用direct数据源时必须提供X和y")
            X = self.direct_X
            y = self.direct_y
            self.original_feature_names = self.direct_feature_names or [f"feature_{i + 1}" for i in range(X.shape[1])]

        else:
            raise ValueError(f"不支持的数据源: {self.data_source}")

        # 处理缺失值
        X = np.nan_to_num(X)

        self.X_raw = X
        self.y_raw = y

        print(f"\n原始数据信息:")
        print(f"  总样本数: {len(y)}")
        print(f"  原始特征数: {X.shape[1]}")
        print(f"  DR样本: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
        print(f"  正常样本: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")

        return X, y

    def apply_feature_engineering(self, X, y=None):
        """
        应用特征工程（修复版）

        参数:
        -----------
        X : array
            原始特征矩阵
        y : array
            标签（可选，用于监督特征选择）

        返回:
        -----------
        X_enhanced : array
            增强后的特征矩阵
        feature_names : list
            特征名称列表
        """
        print("\n" + "=" * 80)
        print("应用特征工程")
        print("=" * 80)

        n_samples = X.shape[0]
        all_features = []
        all_names = []

        # 原始特征
        all_features.append(X)
        all_names.extend(self.original_feature_names)
        feature_counts = {'original': X.shape[1]}

        # 1. 统计特征
        if self.enable_stat_features:
            X_stats, stats_names = self.feature_extractor.extract_statistical_features(
                X, self.original_feature_names
            )
            all_features.append(X_stats)
            all_names.extend(stats_names)
            feature_counts['statistical'] = X_stats.shape[1]

        # 2. 频域特征
        if self.enable_freq_features:
            X_freq, freq_names = self.feature_extractor.extract_frequency_features(
                X, self.original_feature_names
            )
            if X_freq.shape[1] > 0:
                all_features.append(X_freq)
                all_names.extend(freq_names)
                feature_counts['frequency'] = X_freq.shape[1]

        # 3. 小波特征
        if self.enable_wavelet_features:
            X_wavelet, wavelet_names = self.feature_extractor.extract_wavelet_features(
                X, self.original_feature_names
            )
            if X_wavelet.shape[1] > 0:
                all_features.append(X_wavelet)
                all_names.extend(wavelet_names)
                feature_counts['wavelet'] = X_wavelet.shape[1]

        # 4. 交互特征
        if self.enable_interaction_features:
            X_interaction, interaction_names = self.feature_extractor.extract_interaction_features(
                X, self.original_feature_names
            )
            if X_interaction.shape[1] > 0:
                all_features.append(X_interaction)
                all_names.extend(interaction_names)
                feature_counts['interaction'] = X_interaction.shape[1]

        # 5. 非线性特征
        if self.enable_nonlinear_features:
            X_nonlinear, nonlinear_names = self.feature_extractor.extract_nonlinear_features(
                X, self.original_feature_names
            )
            if X_nonlinear.shape[1] > 0:
                all_features.append(X_nonlinear)
                all_names.extend(nonlinear_names)
                feature_counts['nonlinear'] = X_nonlinear.shape[1]

        # 6. 临床特征
        if self.enable_clinical_features:
            X_clinical, clinical_names = self.dr_feature_engineer.create_clinical_features(
                X, self.original_feature_names
            )
            if X_clinical.shape[1] > 0:
                all_features.append(X_clinical)
                all_names.extend(clinical_names)
                feature_counts['clinical'] = X_clinical.shape[1]

        # 7. 纹理特征
        if self.enable_texture_features:
            X_texture, texture_names = self.dr_feature_engineer.create_texture_features(
                X, self.original_feature_names
            )
            if X_texture.shape[1] > 0:
                all_features.append(X_texture)
                all_names.extend(texture_names)
                feature_counts['texture'] = X_texture.shape[1]

        # 合并所有特征
        X_enhanced = np.hstack(all_features)

        print(f"\n特征工程统计:")
        for feat_type, count in feature_counts.items():
            print(f"  {feat_type}: {count} 个特征")
        print(f"  总特征数: {X_enhanced.shape[1]}")

        # 8. 多项式特征
        if self.enable_poly_features:
            X_poly, self.poly_transformer = self.feature_extractor.extract_polynomial_features(
                X_enhanced, degree=self.poly_degree
            )
            feature_counts['polynomial'] = X_poly.shape[1] - X_enhanced.shape[1]
            X_enhanced = X_poly
            print(f"  多项式特征后总特征数: {X_enhanced.shape[1]}")

        # 9. 降维
        if self.enable_dim_reduction:
            X_reduced, self.dim_reducer = self.feature_extractor.reduce_dimension(
                X_enhanced, method=self.reduction_method,
                n_components=self.reduction_components,
                random_state=self.random_state
            )
            feature_counts['reduced'] = X_reduced.shape[1]
            X_enhanced = X_reduced
            print(f"  降维后特征数: {X_enhanced.shape[1]}")

        # 10. 移除低方差特征
        if X_enhanced.shape[1] > 10:
            print("\n  移除低方差特征...")
            selector = VarianceThreshold(threshold=0.01)
            X_enhanced = selector.fit_transform(X_enhanced)
            self.feature_selector = selector
            print(f"    移除后特征数: {X_enhanced.shape[1]}")

        self.feature_names = all_names[:X_enhanced.shape[1]]
        self.feature_engineering_stats = feature_counts

        return X_enhanced, self.feature_names

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        # 加载原始数据
        X, y = self.load_dr_data()

        # 数据分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # 应用特征工程
        print("\n对训练集应用特征工程...")
        self.X_train_enhanced, self.feature_names = self.apply_feature_engineering(
            self.X_train, self.y_train
        )

        print("\n对测试集应用相同的特征工程...")
        # 对测试集应用相同的特征转换
        test_features = [self.X_test]

        if self.enable_stat_features:
            X_stats, _ = self.feature_extractor.extract_statistical_features(
                self.X_test, self.original_feature_names
            )
            test_features.append(X_stats)

        if self.enable_freq_features:
            X_freq, _ = self.feature_extractor.extract_frequency_features(
                self.X_test, self.original_feature_names
            )
            if X_freq.shape[1] > 0:
                test_features.append(X_freq)

        if self.enable_wavelet_features:
            X_wavelet, _ = self.feature_extractor.extract_wavelet_features(
                self.X_test, self.original_feature_names
            )
            if X_wavelet.shape[1] > 0:
                test_features.append(X_wavelet)

        if self.enable_interaction_features:
            X_interaction, _ = self.feature_extractor.extract_interaction_features(
                self.X_test, self.original_feature_names
            )
            if X_interaction.shape[1] > 0:
                test_features.append(X_interaction)

        if self.enable_nonlinear_features:
            X_nonlinear, _ = self.feature_extractor.extract_nonlinear_features(
                self.X_test, self.original_feature_names
            )
            if X_nonlinear.shape[1] > 0:
                test_features.append(X_nonlinear)

        if self.enable_clinical_features:
            X_clinical, _ = self.dr_feature_engineer.create_clinical_features(
                self.X_test, self.original_feature_names
            )
            if X_clinical.shape[1] > 0:
                test_features.append(X_clinical)

        if self.enable_texture_features:
            X_texture, _ = self.dr_feature_engineer.create_texture_features(
                self.X_test, self.original_feature_names
            )
            if X_texture.shape[1] > 0:
                test_features.append(X_texture)

        self.X_test_enhanced = np.hstack(test_features)

        # 应用多项式特征（如果训练时使用了）
        if self.enable_poly_features and self.poly_transformer:
            self.X_test_enhanced = self.poly_transformer.transform(self.X_test_enhanced)

        # 应用降维（如果训练时使用了）
        if self.enable_dim_reduction and self.dim_reducer:
            self.X_test_enhanced = self.dim_reducer.transform(self.X_test_enhanced)

        # 应用特征选择（如果训练时使用了）
        if self.feature_selector:
            self.X_test_enhanced = self.feature_selector.transform(self.X_test_enhanced)

        # 确保特征维度一致
        if self.X_test_enhanced.shape[1] > self.X_train_enhanced.shape[1]:
            self.X_test_enhanced = self.X_test_enhanced[:, :self.X_train_enhanced.shape[1]]
        elif self.X_test_enhanced.shape[1] < self.X_train_enhanced.shape[1]:
            # 补零
            pad_width = self.X_train_enhanced.shape[1] - self.X_test_enhanced.shape[1]
            self.X_test_enhanced = np.pad(self.X_test_enhanced, ((0, 0), (0, pad_width)), 'constant')

        # 数据标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_enhanced)
        self.X_test_scaled = self.scaler.transform(self.X_test_enhanced)

        print(f"\n最终数据统计:")
        print(f"  训练集: {len(self.X_train)} 样本, {self.X_train_scaled.shape[1]} 特征")
        print(f"  测试集: {len(self.X_test)} 样本, {self.X_test_scaled.shape[1]} 特征")

        return self.X_train_scaled, self.y_train

    def cross_validate_model(self, model, model_name, X, y, use_scaled=True):
        """交叉验证"""
        print(f"\n开始对 {model_name} 进行 {self.n_iterations} 折交叉验证...")

        X_cv = X if use_scaled else X
        skf = StratifiedKFold(n_splits=self.n_iterations, shuffle=True, random_state=self.random_state)

        fold_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y), 1):
            print(f"  第 {fold}/{self.n_iterations} 次迭代训练中...", end=' ')

            X_train_fold = X_cv[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_cv[val_idx]
            y_val_fold = y[val_idx]

            from copy import deepcopy
            fold_model = deepcopy(model)
            fold_model.fit(X_train_fold, y_train_fold)
            fold_models.append(fold_model)

            y_val_pred = fold_model.predict(X_val_fold)

            if hasattr(fold_model, 'predict_proba'):
                y_val_proba = fold_model.predict_proba(X_val_fold)[:, 1]
            elif hasattr(fold_model, 'decision_function'):
                y_val_proba = 1 / (1 + np.exp(-fold_model.decision_function(X_val_fold)))
            else:
                y_val_proba = None

            fold_scores['accuracy'].append(accuracy_score(y_val_fold, y_val_pred))
            fold_scores['precision'].append(precision_score(y_val_fold, y_val_pred, zero_division=0))
            fold_scores['recall'].append(recall_score(y_val_fold, y_val_pred, zero_division=0))
            fold_scores['f1'].append(f1_score(y_val_fold, y_val_pred, zero_division=0))

            if y_val_proba is not None:
                fold_scores['auc'].append(roc_auc_score(y_val_fold, y_val_proba))
            else:
                fold_scores['auc'].append(0)

            print(f"完成 F1={fold_scores['f1'][-1]:.4f}")

        results = {}
        for metric in fold_scores:
            results[metric] = {
                'mean': np.mean(fold_scores[metric]),
                'std': np.std(fold_scores[metric]),
                'scores': fold_scores[metric]
            }

        print(f"\n  {self.n_iterations}折交叉验证结果:")
        print(f"    准确率: {results['accuracy']['mean']:.4f} (±{results['accuracy']['std']:.4f})")
        print(f"    F1分数: {results['f1']['mean']:.4f} (±{results['f1']['std']:.4f})")
        print(f"    AUC: {results['auc']['mean']:.4f} (±{results['auc']['std']:.4f})")

        return results, fold_models

    def save_model_with_metadata(self, model, model_name, fold_models=None, is_best=False):
        """保存模型及其元数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.model_subdirs['Best Model'] if is_best else self.model_subdirs[model_name]

        # 保存模型
        final_model_path = os.path.join(save_dir, f'{model_name.replace(" ", "_")}_final_model_{timestamp}.pkl')
        joblib.dump(model, final_model_path)
        print(f"  ✓ 保存模型: {final_model_path}")

        if fold_models:
            fold_models_path = os.path.join(save_dir, f'{model_name.replace(" ", "_")}_cv_folds_{timestamp}.pkl')
            joblib.dump(fold_models, fold_models_path)

        # 保存元数据
        metadata = {
            'model_name': model_name,
            'dr_dataset': self.dr_dataset,
            'data_source': self.data_source,
            'n_iterations': self.n_iterations,
            'test_size': self.test_size,
            'feature_engineering_stats': self.feature_engineering_stats,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'timestamp': timestamp
        }

        if model_name in self.cv_scores:
            metadata['cv_scores'] = {
                'accuracy_mean': float(self.cv_scores[model_name]['accuracy']['mean']),
                'f1_mean': float(self.cv_scores[model_name]['f1']['mean']),
                'auc_mean': float(self.cv_scores[model_name]['auc']['mean'])
            }

        metadata_path = os.path.join(save_dir, f'{model_name.replace(" ", "_")}_metadata_{timestamp}.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def save_scaler_and_preprocessors(self):
        """保存预处理器"""
        preprocessors_dir = os.path.join(self.models_dir, 'Preprocessors')
        os.makedirs(preprocessors_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存标准化器
        scaler_path = os.path.join(preprocessors_dir, f'scaler_{timestamp}.pkl')
        joblib.dump(self.scaler, scaler_path)

        # 保存特征选择器
        if self.feature_selector:
            selector_path = os.path.join(preprocessors_dir, f'feature_selector_{timestamp}.pkl')
            joblib.dump(self.feature_selector, selector_path)

        # 保存多项式转换器
        if self.poly_transformer:
            poly_path = os.path.join(preprocessors_dir, f'poly_transformer_{timestamp}.pkl')
            joblib.dump(self.poly_transformer, poly_path)

        # 保存降维器
        if self.dim_reducer:
            reducer_path = os.path.join(preprocessors_dir, f'dim_reducer_{timestamp}.pkl')
            joblib.dump(self.dim_reducer, reducer_path)

        print(f"✓ 保存预处理器到: {preprocessors_dir}")

    def _use_scaled_features(self, model_name):
        """判断模型是否使用标准化后的特征。"""
        return model_name in ['SVM', 'Logistic Regression', 'Neural Network']

    def train_svm(self, C=1.0):
        """训练SVM模型"""
        print("=" * 80)
        print("训练 SVM 模型 (支持向量机)")
        print("=" * 80)

        model = SVC(C=C, kernel='rbf', gamma='scale',
                    class_weight='balanced', probability=True,
                    random_state=self.random_state)

        cv_results, fold_models = self.cross_validate_model(
            model, 'SVM', self.X_train_scaled, self.y_train, use_scaled=True
        )
        self.cv_scores['SVM'] = cv_results

        model.fit(self.X_train_scaled, self.y_train)
        self.models['SVM'] = model
        self.save_model_with_metadata(model, 'SVM', fold_models)
        return model

    def train_logistic_regression(self, C=1.0):
        """训练逻辑回归模型"""
        print("=" * 80)
        print("训练逻辑回归模型")
        print("=" * 80)

        model = LogisticRegression(C=C, max_iter=2000, class_weight='balanced',
                                   random_state=self.random_state, solver='lbfgs')

        cv_results, fold_models = self.cross_validate_model(
            model, 'Logistic Regression', self.X_train_scaled, self.y_train, use_scaled=True
        )
        self.cv_scores['Logistic Regression'] = cv_results

        model.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = model
        self.save_model_with_metadata(model, 'Logistic Regression', fold_models)
        return model

    def train_decision_tree(self, max_depth=10, min_samples_split=20, min_samples_leaf=10):
        """训练决策树模型"""
        print("=" * 80)
        print("训练决策树模型")
        print("=" * 80)

        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, class_weight='balanced',
                                       random_state=self.random_state)

        cv_results, fold_models = self.cross_validate_model(
            model, 'Decision Tree', self.X_train_enhanced, self.y_train, use_scaled=False
        )
        self.cv_scores['Decision Tree'] = cv_results

        model.fit(self.X_train_enhanced, self.y_train)
        self.models['Decision Tree'] = model
        self.save_model_with_metadata(model, 'Decision Tree', fold_models)
        return model

    def train_random_forest(self, n_estimators=100, max_depth=10,
                            min_samples_split=20, min_samples_leaf=10):
        """训练随机森林模型"""
        print("=" * 80)
        print("训练随机森林模型 (集成学习)")
        print("=" * 80)

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       class_weight='balanced', random_state=self.random_state, n_jobs=-1)

        cv_results, fold_models = self.cross_validate_model(
            model, 'Random Forest', self.X_train_enhanced, self.y_train, use_scaled=False
        )
        self.cv_scores['Random Forest'] = cv_results

        model.fit(self.X_train_enhanced, self.y_train)
        self.models['Random Forest'] = model
        self.save_model_with_metadata(model, 'Random Forest', fold_models)
        return model

    def train_gradient_boosting(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        """训练梯度提升模型"""
        print("=" * 80)
        print("训练梯度提升模型 (Gradient Boosting)")
        print("=" * 80)

        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           max_depth=max_depth, random_state=self.random_state)

        cv_results, fold_models = self.cross_validate_model(
            model, 'Gradient Boosting', self.X_train_enhanced, self.y_train, use_scaled=False
        )
        self.cv_scores['Gradient Boosting'] = cv_results

        model.fit(self.X_train_enhanced, self.y_train)
        self.models['Gradient Boosting'] = model
        self.save_model_with_metadata(model, 'Gradient Boosting', fold_models)
        return model

    def train_neural_network(self, hidden_layer_sizes=(128, 64), alpha=0.0001,
                             learning_rate_init=0.001, max_iter=500):
        """训练神经网络模型（MLP多层感知机）"""
        print("=" * 80)
        print("训练神经网络模型 (MLP多层感知机)")
        print("=" * 80)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=self.random_state
        )

        cv_results, fold_models = self.cross_validate_model(
            model, 'Neural Network', self.X_train_scaled, self.y_train, use_scaled=True
        )
        self.cv_scores['Neural Network'] = cv_results

        model.fit(self.X_train_scaled, self.y_train)
        self.models['Neural Network'] = model
        self.save_model_with_metadata(model, 'Neural Network', fold_models)
        return model

    def compare_all_models(self):
        """比较所有模型的性能"""
        print("\n" + "=" * 80)
        print("DR检测模型性能对比分析")
        print("=" * 80)

        results_summary = []

        for model_name, model in self.models.items():
            print(f"\n{'=' * 40}")
            print(f"模型: {model_name}")
            print('=' * 40)

            use_scaled = self._use_scaled_features(model_name)
            X_test_eval = self.X_test_scaled if use_scaled else self.X_test_enhanced

            # 预测
            y_test_pred = model.predict(X_test_eval)

            # 获取概率
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test_eval)[:, 1]
                test_auc = roc_auc_score(self.y_test, y_test_proba)
            else:
                test_auc = 0

            # 计算指标
            test_acc = accuracy_score(self.y_test, y_test_pred)
            test_precision = precision_score(self.y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(self.y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)

            cv_f1_mean = self.cv_scores[model_name]['f1']['mean']
            cv_f1_std = self.cv_scores[model_name]['f1']['std']

            print(f"测试集性能:")
            print(f"  准确率: {test_acc:.4f}")
            print(f"  精确率: {test_precision:.4f}")
            print(f"  召回率: {test_recall:.4f}")
            print(f"  F1分数: {test_f1:.4f}")
            print(f"  AUC: {test_auc:.4f}")
            print(f"交叉验证F1: {cv_f1_mean:.4f} (±{cv_f1_std:.4f})")

            results_summary.append({
                'Model': model_name,
                'Test_Acc': test_acc,
                'Test_Precision': test_precision,
                'Test_Recall': test_recall,
                'Test_F1': test_f1,
                'Test_AUC': test_auc,
                'CV_F1_Mean': cv_f1_mean,
                'CV_F1_Std': cv_f1_std
            })

        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Test_F1', ascending=False)

        print("\n" + "=" * 80)
        print("DR检测模型性能排名（按测试集F1分数）")
        print("=" * 80)
        print(results_df[['Model', 'Test_Acc', 'Test_F1', 'Test_AUC', 'CV_F1_Mean']].to_string(index=False))

        return results_df

    def plot_cv_results(self):
        """绘制交叉验证结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        models = list(self.cv_scores.keys())
        cv_means = [self.cv_scores[m]['f1']['mean'] for m in models]
        cv_stds = [self.cv_scores[m]['f1']['std'] for m in models]

        # 柱状图
        bars = axes[0].bar(models, cv_means, yerr=cv_stds, capsize=5,
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7B731'],
                           alpha=0.8)
        axes[0].set_xlabel('模型', fontsize=12)
        axes[0].set_ylabel('交叉验证F1分数', fontsize=12)
        axes[0].set_title(f'{self.n_iterations}折交叉验证结果 (增强特征)', fontsize=14)
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0.85, color='green', linestyle='--', label='优秀阈值(0.85)')
        axes[0].legend()

        # 在柱状图上添加数值
        for bar, mean in zip(bars, cv_means):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{mean:.3f}', ha='center', fontsize=10)

        # 折线图
        for model_name in models:
            scores = self.cv_scores[model_name]['f1']['scores']
            axes[1].plot(range(1, len(scores) + 1), scores, 'o-',
                         label=model_name, linewidth=2, markersize=8)

        axes[1].set_xlabel('迭代次数', fontsize=12)
        axes[1].set_ylabel('F1分数', fontsize=12)
        axes[1].set_title('各次迭代F1分数变化 (增强特征)', fontsize=14)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        axes[1].axhline(y=0.85, color='green', linestyle='--', alpha=0.5)

        plt.suptitle('DR检测模型训练过程分析（特征增强版）', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (model_name, model) in enumerate(self.models.items()):
            if idx >= 6:
                break

            use_scaled = self._use_scaled_features(model_name)
            X_eval = self.X_test_scaled if use_scaled else self.X_test_enhanced
            y_pred = model.predict(X_eval)

            cm = confusion_matrix(self.y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['正常', 'DR'],
                        yticklabels=['正常', 'DR'],
                        ax=axes[idx])
            axes[idx].set_title(f'{model_name}', fontsize=12)
            axes[idx].set_xlabel('预测标签', fontsize=10)
            axes[idx].set_ylabel('真实标签', fontsize=10)

            # 计算并显示准确率和F1
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            axes[idx].text(0.5, -0.15, f'准确率: {acc:.4f} | F1: {f1:.4f}',
                           transform=axes[idx].transAxes, ha='center', fontsize=10)

        # 隐藏多余的子图
        for idx in range(len(self.models), 6):
            axes[idx].axis('off')

        plt.suptitle('DR检测模型混淆矩阵对比（特征增强版）', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7B731', '#DDA0DD']

        for idx, (model_name, model) in enumerate(self.models.items()):
            use_scaled = self._use_scaled_features(model_name)
            X_eval = self.X_test_scaled if use_scaled else self.X_test_enhanced

            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_eval)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = 1 / (1 + np.exp(-model.decision_function(X_eval)))
            else:
                continue

            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)

            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                     label=f'{model_name} (AUC = {auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类器 (AUC=0.5)')
        plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        plt.title('DR检测模型ROC曲线对比（特征增强版）', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_best_model(self, model, model_name, model_path='best_dr_model_enhanced.pkl'):
        """保存最佳DR检测模型"""
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': model_name,
            'dr_dataset': self.dr_dataset,
            'feature_engineering_stats': self.feature_engineering_stats,
            'n_iterations': self.n_iterations,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'feature_selector': self.feature_selector,
            'poly_transformer': self.poly_transformer,
            'dim_reducer': self.dim_reducer,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'feature_engineering_config': {
                'stat_features': self.enable_stat_features,
                'freq_features': self.enable_freq_features,
                'wavelet_features': self.enable_wavelet_features,
                'interaction_features': self.enable_interaction_features,
                'nonlinear_features': self.enable_nonlinear_features,
                'poly_features': self.enable_poly_features,
                'dim_reduction': self.enable_dim_reduction,
                'clinical_features': self.enable_clinical_features,
                'texture_features': self.enable_texture_features
            }
        }

        # 保存到根目录
        joblib.dump(model_data, model_path)
        print(f"\n✓ 最佳DR检测模型 ({model_name}) 已保存到 {model_path}")

        # 保存到最佳模型子目录
        best_model_path = os.path.join(self.model_subdirs['Best Model'], 'best_dr_model_enhanced.pkl')
        joblib.dump(model_data, best_model_path)
        print(f"✓ 最佳DR检测模型已保存到 {best_model_path}")

    def run_full_comparison(self):
        """运行完整的DR检测模型比较流程（包含特征工程）"""
        # 加载和预处理数据
        self.load_and_preprocess_data()

        # 保存预处理器
        print("\n" + "=" * 80)
        print("保存预处理器")
        print("=" * 80)
        self.save_scaler_and_preprocessors()

        # 训练所有模型
        print("\n" + "=" * 80)
        print(f"开始训练DR检测模型（{self.n_iterations}折交叉验证）")
        print("=" * 80 + "\n")

        self.train_svm(C=1.0)
        print()
        self.train_logistic_regression(C=1.0)
        print()
        self.train_decision_tree(max_depth=10, min_samples_split=20, min_samples_leaf=10)
        print()
        self.train_random_forest(n_estimators=100, max_depth=10,
                                 min_samples_split=20, min_samples_leaf=10)
        print()
        self.train_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=5)
        print()
        self.train_neural_network(hidden_layer_sizes=(128, 64), alpha=0.0001,
                                  learning_rate_init=0.001, max_iter=500)

        # 绘制结果
        self.plot_cv_results()

        # 比较模型
        results_df = self.compare_all_models()

        # 可视化对比
        self.plot_confusion_matrices()
        self.plot_roc_curves()

        # 保存最佳模型
        best_model_name = results_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        self.save_best_model(best_model, best_model_name)

        return results_df


# =============================================================================
# 使用示例
# =============================================================================

def example_feature_engineering_demo():
    """
    示例：使用完整的特征工程功能进行DR检测
    """
    print("\n" + "=" * 80)
    print("DR检测系统 - 完整特征工程版本")
    print("=" * 80)

    # 配置特征工程参数
    comparator = ModelComparator(
        data_source='synthetic',  # 使用合成数据

        # 启用特征提取方法
        enable_stat_features=True,  # 统计特征
        enable_freq_features=False,  # 频域特征（可选）
        enable_wavelet_features=False,  # 小波特征（需要pywt）
        enable_interaction_features=True,  # 交互特征
        enable_nonlinear_features=True,  # 非线性特征
        enable_poly_features=False,  # 多项式特征（会增加很多特征）
        poly_degree=2,
        enable_dim_reduction=False,  # 降维
        reduction_method='pca',
        reduction_components=0.95,
        enable_clinical_features=True,  # 临床特征
        enable_texture_features=True,  # 纹理特征

        test_size=0.25,
        random_state=42,
        n_iterations=5,
        models_dir='saved_models_dr_enhanced'
    )

    # 运行完整对比
    results = comparator.run_full_comparison()

    # 打印最终结果
    print("\n" + "=" * 80)
    print("DR检测模型训练总结（特征增强版）")
    print("=" * 80)

    best_model = results.iloc[0]
    print(f"\n🏆 最佳DR检测模型: {best_model['Model']}")
    print(f"   - 测试准确率: {best_model['Test_Acc']:.4f}")
    print(f"   - 测试F1分数: {best_model['Test_F1']:.4f}")
    print(f"   - 测试AUC: {best_model['Test_AUC']:.4f}")
    print(f"   - 交叉验证F1: {best_model['CV_F1_Mean']:.4f}")

    return results


def main():
    """主函数"""
    # 运行完整特征工程版本
    results = example_feature_engineering_demo()

    return results


if __name__ == "__main__":
    main()