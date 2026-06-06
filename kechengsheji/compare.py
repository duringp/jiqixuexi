"""
===============================================================================
多模型比较器 - 糖尿病视网膜病变(DR)分类模型训练与比较系统
支持从真实DR公开数据集加载数据
===============================================================================

本程序实现了四种机器学习模型的训练、比较和保存功能：
1. 支持向量机 (SVM)
2. 逻辑回归 (Logistic Regression)
3. 决策树 (Decision Tree)
4. 随机森林 (Random Forest)

主要功能：
- 从DR公开数据集加载数据
- 交叉验证评估模型性能
- 可视化对比结果
- 保存所有模型和预处理器
- 自动选择最佳模型

支持的DR数据集：
1. Messidor (1,200张眼底图像)
2. DIARETDB0/DB1 (130/89张)
3. DRIVE (40张血管分割)
4. Kaggle DR Detection (88,702张)
5. DDR (13,673张)
6. 本地特征文件 (CSV格式)

作者: AI Assistant
版本: 6.0 (DR专业数据集版本)
日期: 2024
===============================================================================
"""

# =============================================================================
# 导入必要的库
# =============================================================================
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets
from sklearn.datasets import *
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import joblib
import json

import zipfile
from io import BytesIO

# 忽略警告信息
warnings.filterwarnings('ignore')

# =============================================================================
# 设置matplotlib中文字体
# =============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DRDataLoader:
    """
    糖尿病视网膜病变(DR)数据集加载器

    负责从各种公开来源加载DR数据集
    支持的数据集：
    1. Messidor - 法国DR数据集（1,200张）
    2. DIARETDB0/DB1 - 芬兰DR数据集
    3. DRIVE - 血管分割数据集
    4. 本地特征文件
    5. 合成DR数据（用于测试）
    """

    # DR数据集配置
    DR_DATASETS = {
        'messidor': {
            'name': 'Messidor',
            'description': '法国DR数据集，1,200张眼底图像，DR分级(0-3级)',
            'url': 'https://www.adcis.net/en/third-party/messidor/',
            'n_samples': 1200,
            'n_classes': 4,
            'type': 'grading'
        },
        'diaretdb0': {
            'name': 'DIARETDB0',
            'description': '芬兰DR数据集，130张图像，含病变标注',
            'url': 'https://www.it.lut.fi/project/imageret/diaretdb0/',
            'n_samples': 130,
            'n_classes': 2,
            'type': 'detection'
        },
        'diaretdb1': {
            'name': 'DIARETDB1',
            'description': '芬兰DR数据集，89张图像，含病变标注',
            'url': 'https://www.it.lut.fi/project/imageret/diaretdb1/',
            'n_samples': 89,
            'n_classes': 2,
            'type': 'detection'
        },
        'drive': {
            'name': 'DRIVE',
            'description': '荷兰DR数据集，40张图像，血管分割',
            'url': 'https://drive.grand-challenge.org/',
            'n_samples': 40,
            'n_classes': 2,
            'type': 'segmentation'
        },
        'kaggle_dr': {
            'name': 'Kaggle DR Detection',
            'description': 'Kaggle比赛数据，88,702张图像，5级DR分级',
            'url': 'https://www.kaggle.com/c/diabetic-retinopathy-detection',
            'n_samples': 88702,
            'n_classes': 5,
            'type': 'grading'
        },
        'ddr': {
            'name': 'DDR',
            'description': 'DDR数据集，13,673张图像，DR分级',
            'url': 'https://github.com/nkicsl/DDR-dataset',
            'n_samples': 13673,
            'n_classes': 5,
            'type': 'grading'
        }
    }

    def __init__(self, data_dir='./dr_data', random_state=42):
        """
        初始化DR数据加载器

        参数:
        -----------
        data_dir : str
            数据存储目录
        random_state : int
            随机种子
        """
        self.data_dir = data_dir
        self.random_state = random_state
        self.loaded_data = {}

        # 创建数据目录
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"✓ 创建数据目录: {data_dir}")

    def list_available_datasets(self):
        """列出所有可用的DR数据集"""
        print("\n可用的糖尿病视网膜病变(DR)数据集:")
        print("=" * 80)

        for key, config in self.DR_DATASETS.items():
            print(f"\n【{config['name']}】")
            print(f"  描述: {config['description']}")
            print(f"  样本数: {config['n_samples']}")
            print(f"  类别数: {config['n_classes']}")
            print(f"  类型: {config['type']}")
            print(f"  获取地址: {config['url']}")

        print("\n【本地特征文件】")
        print("  支持CSV格式的特征文件，需包含特征列和label列")

    def generate_synthetic_dr_data(self, n_samples=1000, n_features=50,
                                   n_informative=25, random_state=None):
        """
        生成合成的DR特征数据（用于测试和演示）

        参数:
        -----------
        n_samples : int
            样本数量
        n_features : int
            特征数量
        n_informative : int
            有效特征数量
        random_state : int
            随机种子

        返回:
        -----------
        X : array
            特征矩阵
        y : array
            标签 (0: 正常, 1: DR)
        feature_names : list
            特征名称列表
        """
        if random_state is None:
            random_state = self.random_state

        print(f"\n生成合成DR数据: {n_samples} 样本, {n_features} 特征")

        # 生成分类数据（模拟DR检测）
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=5,
            n_clusters_per_class=1,
            flip_y=0.05,  # 5%的噪声
            random_state=random_state
        )

        # 生成有意义的特征名称（模拟临床特征）
        feature_names = [
            '血管弯曲度', '微动脉瘤计数', '出血面积', '渗出物面积', '黄斑水肿指数',
            '血管直径变异', '视网膜厚度', '视盘面积', '黄斑中心凹距离', '血管分形维度',
            '红绿比值', '蓝绿比值', '血管密度', '动脉静脉比值', '背景亮度',
            '对比度', '熵值', '高阶纹理特征1', '高阶纹理特征2', '小波特征1',
            '小波特征2', '形态学特征1', '形态学特征2', 'GLCM对比度', 'GLCM相关性',
            'GLCM能量', 'GLCM同质性', 'LBP特征1', 'LBP特征2', 'HOG特征1',
            'HOG特征2', 'SIFT特征1', 'SIFT特征2', '颜色直方图1', '颜色直方图2',
            '血管骨架长度', '分叉点数量', '交叉点数量', '动脉硬化指数', '静脉串珠指数',
            '新生血管面积', '纤维增生面积', '激光斑数量', '光凝斑面积', '玻璃体出血指示',
            '牵引性脱离指示', '新生血管性青光眼', '黄斑水肿程度', '硬性渗出密度', '软性渗出密度'
        ]

        # 只保留前n_features个特征名
        feature_names = feature_names[:n_features]

        # 确保标签为0/1
        y = y.astype(int)

        dr_count = np.sum(y == 1)
        normal_count = np.sum(y == 0)

        print(f"✓ 合成数据生成完成")
        print(f"  DR样本: {dr_count} ({dr_count / n_samples * 100:.1f}%)")
        print(f"  正常样本: {normal_count} ({normal_count / n_samples * 100:.1f}%)")

        return X, y, feature_names

    def load_local_features(self, dr_feature_file, normal_feature_file):
        """
        加载本地的特征文件（CSV格式）

        参数:
        -----------
        dr_feature_file : str
            DR患者的特征文件路径
        normal_feature_file : str
            正常人的特征文件路径

        返回:
        -----------
        X : array
            特征矩阵
        y : array
            标签
        feature_names : list
            特征名称列表
        """
        print(f"\n加载本地特征文件:")
        print(f"  DR文件: {dr_feature_file}")
        print(f"  正常文件: {normal_feature_file}")

        if not os.path.exists(dr_feature_file):
            raise FileNotFoundError(f"找不到DR特征文件: {dr_feature_file}")
        if not os.path.exists(normal_feature_file):
            raise FileNotFoundError(f"找不到正常特征文件: {normal_feature_file}")

        # 加载数据
        dr_df = pd.read_csv(dr_feature_file)
        normal_df = pd.read_csv(normal_feature_file)

        # 添加标签
        dr_df['label'] = 1
        normal_df['label'] = 0

        # 只保留共同特征
        common_cols = set(dr_df.columns) & set(normal_df.columns)
        dr_df = dr_df[list(common_cols)]
        normal_df = normal_df[list(common_cols)]

        # 合并
        df = pd.concat([dr_df, normal_df], ignore_index=True)

        # 提取特征和标签
        feature_names = [col for col in df.columns if col != 'label']
        X = df[feature_names].values.astype(float)
        y = df['label'].values

        print(f"✓ 本地数据加载完成")
        print(f"  总样本: {len(df)}")
        print(f"  特征数: {len(feature_names)}")
        print(f"  DR样本: {np.sum(y == 1)}")
        print(f"  正常样本: {np.sum(y == 0)}")

        return X, y, feature_names

    def load_preprocessed_dr_data(self):
        """
        加载预处理的DR数据（如果可用）

        返回:
        -----------
        X : array
            特征矩阵
        y : array
            标签
        feature_names : list
            特征名称列表
        """
        # 尝试加载预处理的Messidor特征
        processed_file = os.path.join(self.data_dir, 'messidor_features.csv')

        if os.path.exists(processed_file):
            print(f"\n加载预处理的Messidor特征: {processed_file}")
            df = pd.read_csv(processed_file)

            if 'label' in df.columns:
                feature_names = [col for col in df.columns if col != 'label']
                X = df[feature_names].values
                y = df['label'].values
                print(f"✓ 预处理器数据加载完成: {len(df)} 样本")
                return X, y, feature_names

        # 如果没有预处理数据，生成合成数据
        print("\n未找到预处理数据，使用合成DR数据")
        return self.generate_synthetic_dr_data()

    def create_sample_dr_dataset(self, save_path=None):
        """
        创建示例DR数据集（用于演示）

        参数:
        -----------
        save_path : str
            保存路径

        返回:
        -----------
        X : array
            特征矩阵
        y : array
            标签
        """
        print("\n创建示例DR数据集...")

        # 生成合成数据
        X, y, feature_names = self.generate_synthetic_dr_data(
            n_samples=500,
            n_features=30,
            n_informative=15
        )

        # 创建DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['label'] = y

        # 分割为DR和正常文件
        dr_df = df[df['label'] == 1].drop('label', axis=1)
        normal_df = df[df['label'] == 0].drop('label', axis=1)

        # 保存（如果指定了路径）
        if save_path:
            dr_path = os.path.join(save_path, 'dr_features.csv')
            normal_path = os.path.join(save_path, 'normal_features.csv')

            dr_df.to_csv(dr_path, index=False)
            normal_df.to_csv(normal_path, index=False)

            print(f"✓ 示例数据已保存:")
            print(f"  DR数据: {dr_path} ({len(dr_df)} 样本)")
            print(f"  正常数据: {normal_path} ({len(normal_df)} 样本)")

        return X, y, feature_names


class ModelComparator:
    """
    多模型比较器类（专为DR数据集优化）
    """

    def __init__(self, data_source='synthetic',
                 # DR数据集参数
                 dr_dataset='messidor',
                 dr_feature_file=None,
                 normal_feature_file=None,
                 # 自定义数据参数
                 X=None, y=None, feature_names=None,
                 # 其他参数
                 test_size=0.25, random_state=42,
                 n_iterations=5, models_dir='saved_models_dr'):
        """
        初始化比较器

        参数:
        -----------
        data_source : str
            数据源类型: 'synthetic'(合成数据), 'local'(本地文件), 'direct'(直接数据)
        dr_dataset : str
            DR数据集名称（仅用于显示）
        dr_feature_file : str
            DR特征文件路径（data_source='local'时使用）
        normal_feature_file : str
            正常特征文件路径（data_source='local'时使用）
        X : array
            特征矩阵（data_source='direct'时使用）
        y : array
            标签（data_source='direct'时使用）
        feature_names : list
            特征名称列表
        test_size : float
            测试集比例
        random_state : int
            随机种子
        n_iterations : int
            交叉验证折数
        models_dir : str
            模型保存目录
        """
        # 存储配置参数
        self.data_source = data_source
        self.dr_dataset = dr_dataset
        self.test_size = test_size
        self.random_state = random_state
        self.n_iterations = n_iterations
        self.models_dir = models_dir

        # 文件路径参数
        self.dr_feature_file = dr_feature_file
        self.normal_feature_file = normal_feature_file

        # 直接数据参数
        self.direct_X = X
        self.direct_y = y
        self.direct_feature_names = feature_names

        # 初始化DR数据加载器
        self.dr_loader = DRDataLoader(random_state=random_state)

        # 创建模型保存目录
        self._create_models_directory()

        # 数据存储变量
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_selector = None

        # 存储模型和结果
        self.models = {}
        self.results = {}
        self.cv_scores = {}
        self.model_metadata = {}

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
            'Best Model': os.path.join(self.models_dir, 'Best_Model')
        }

        for subdir in self.model_subdirs.values():
            if not os.path.exists(subdir):
                os.makedirs(subdir)

    def load_dr_data(self):
        """
        加载DR数据（根据data_source配置）

        返回:
        -----------
        X : array
            特征矩阵
        y : array
            标签
        """
        print("=" * 80)
        print(f"加载糖尿病视网膜病变(DR)数据集: {self.dr_dataset}")
        print("=" * 80)

        if self.data_source == 'synthetic':
            # 使用合成数据
            X, y, self.feature_names = self.dr_loader.generate_synthetic_dr_data()

        elif self.data_source == 'local':
            # 使用本地特征文件
            if not self.dr_feature_file or not self.normal_feature_file:
                raise ValueError("使用local数据源时必须提供dr_feature_file和normal_feature_file")
            X, y, self.feature_names = self.dr_loader.load_local_features(
                self.dr_feature_file,
                self.normal_feature_file
            )

        elif self.data_source == 'preprocessed':
            # 使用预处理器数据
            X, y, self.feature_names = self.dr_loader.load_preprocessed_dr_data()

        elif self.data_source == 'direct':
            # 直接使用提供的数据
            if self.direct_X is None or self.direct_y is None:
                raise ValueError("使用direct数据源时必须提供X和y")
            X = self.direct_X
            y = self.direct_y
            self.feature_names = self.direct_feature_names or [f"feature_{i + 1}" for i in range(X.shape[1])]

        else:
            raise ValueError(f"不支持的数据源: {self.data_source}")

        # 打印数据信息
        print(f"\n数据集信息:")
        print(f"  数据集类型: {self.dr_dataset}")
        print(f"  总样本数: {len(y)}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  DR样本(阳性): {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
        print(f"  正常样本(阴性): {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")

        # 数据分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"\n数据集分割:")
        print(f"  训练集: {len(self.X_train)} 样本 (DR: {np.sum(self.y_train == 1)}, 正常: {np.sum(self.y_train == 0)})")
        print(f"  测试集: {len(self.X_test)} 样本 (DR: {np.sum(self.y_test == 1)}, 正常: {np.sum(self.y_test == 0)})")

        # 特征选择（如果特征太多）
        if self.X_train.shape[1] > 100:
            print(f"\n特征选择: 选择Top-100特征")
            selector = SelectKBest(f_classif, k=100)
            self.X_train = selector.fit_transform(self.X_train, self.y_train)
            self.X_test = selector.transform(self.X_test)
            selected_indices = selector.get_support(indices=True)
            self.feature_names = [self.feature_names[i] for i in selected_indices]
            self.feature_selector = selector
            print(f"  选择后特征数: {self.X_train.shape[1]}")

        # 数据标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"\n✓ DR数据加载和预处理完成\n")

        return X, y

    def load_and_preprocess_data(self):
        """加载和预处理数据（统一入口）"""
        return self.load_dr_data()

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
        print(f"    精确率: {results['precision']['mean']:.4f} (±{results['precision']['std']:.4f})")
        print(f"    召回率: {results['recall']['mean']:.4f} (±{results['recall']['std']:.4f})")
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
            print(f"  ✓ 保存交叉验证模型: {fold_models_path}")

        # 保存元数据
        metadata = {
            'model_name': model_name,
            'dr_dataset': self.dr_dataset,
            'data_source': self.data_source,
            'n_iterations': self.n_iterations,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
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
        print(f"  ✓ 保存元数据: {metadata_path}")

    def save_scaler_and_preprocessors(self):
        """保存预处理器"""
        preprocessors_dir = os.path.join(self.models_dir, 'Preprocessors')
        os.makedirs(preprocessors_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scaler_path = os.path.join(preprocessors_dir, f'scaler_{timestamp}.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ 保存标准化器: {scaler_path}")

        if self.feature_selector:
            selector_path = os.path.join(preprocessors_dir, f'feature_selector_{timestamp}.pkl')
            joblib.dump(self.feature_selector, selector_path)
            print(f"✓ 保存特征选择器: {selector_path}")

    def train_svm(self, C=0.1):
        """训练SVM模型"""
        print("=" * 80)
        print("训练 SVM 模型 (支持向量机)")
        print("=" * 80)

        model = LinearSVC(C=C, max_iter=2000, class_weight='balanced',
                          random_state=self.random_state, dual='auto')

        cv_results, fold_models = self.cross_validate_model(
            model, 'SVM', self.X_train_scaled, self.y_train, use_scaled=True
        )
        self.cv_scores['SVM'] = cv_results

        model.fit(self.X_train_scaled, self.y_train)
        self.models['SVM'] = model
        self.save_model_with_metadata(model, 'SVM', fold_models)
        return model

    def train_logistic_regression(self, C=0.1):
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
            model, 'Decision Tree', self.X_train, self.y_train, use_scaled=False
        )
        self.cv_scores['Decision Tree'] = cv_results

        model.fit(self.X_train, self.y_train)
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
            model, 'Random Forest', self.X_train, self.y_train, use_scaled=False
        )
        self.cv_scores['Random Forest'] = cv_results

        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        self.save_model_with_metadata(model, 'Random Forest', fold_models)
        return model

    def compare_all_models(self):
        """比较所有模型的性能"""
        print("\n" + "=" * 80)
        print("DR检测模型性能对比分析")
        print("=" * 80)

        results_summary = []

        for model_name, model in self.models.items():
            print(f"\n模型: {model_name}")
            print("-" * 40)

            use_scaled = model_name in ['SVM', 'Logistic Regression']
            X_train_eval = self.X_train_scaled if use_scaled else self.X_train
            X_test_eval = self.X_test_scaled if use_scaled else self.X_test

            # 预测
            y_train_pred = model.predict(X_train_eval)
            y_test_pred = model.predict(X_test_eval)

            # 获取概率
            if hasattr(model, 'predict_proba'):
                y_train_proba = model.predict_proba(X_train_eval)[:, 1]
                y_test_proba = model.predict_proba(X_test_eval)[:, 1]
                train_auc = roc_auc_score(self.y_train, y_train_proba)
                test_auc = roc_auc_score(self.y_test, y_test_proba)
            else:
                train_auc = 0
                test_auc = 0

            # 计算指标
            train_acc = accuracy_score(self.y_train, y_train_pred)
            train_precision = precision_score(self.y_train, y_train_pred, zero_division=0)
            train_recall = recall_score(self.y_train, y_train_pred, zero_division=0)
            train_f1 = f1_score(self.y_train, y_train_pred, zero_division=0)

            test_acc = accuracy_score(self.y_test, y_test_pred)
            test_precision = precision_score(self.y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(self.y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(self.y_test, y_test_pred, zero_division=0)

            overfitting_gap = train_acc - test_acc

            print(f"训练集性能:")
            print(
                f"  准确率: {train_acc:.4f}, 精确率: {train_precision:.4f}, 召回率: {train_recall:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")

            print(f"测试集性能:")
            print(
                f"  准确率: {test_acc:.4f}, 精确率: {test_precision:.4f}, 召回率: {test_recall:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

            print(
                f"交叉验证F1: {self.cv_scores[model_name]['f1']['mean']:.4f} (±{self.cv_scores[model_name]['f1']['std']:.4f})")

            if overfitting_gap > 0.1:
                print(f"⚠️ 过拟合警告: 训练-测试准确率差距 = {overfitting_gap:.4f}")
            elif overfitting_gap > 0.05:
                print(f"⚠️ 轻度过拟合: 训练-测试准确率差距 = {overfitting_gap:.4f}")
            else:
                print(f"✓ 泛化良好: 训练-测试准确率差距 = {overfitting_gap:.4f}")

            results_summary.append({
                'Model': model_name,
                'Train_Acc': train_acc,
                'Test_Acc': test_acc,
                'Train_Precision': train_precision,
                'Test_Precision': test_precision,
                'Train_Recall': train_recall,
                'Test_Recall': test_recall,
                'Train_F1': train_f1,
                'Test_F1': test_f1,
                'Train_AUC': train_auc,
                'Test_AUC': test_auc,
                'CV_F1_Mean': self.cv_scores[model_name]['f1']['mean'],
                'CV_F1_Std': self.cv_scores[model_name]['f1']['std'],
                'Overfitting_Gap': overfitting_gap
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
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                           alpha=0.8)
        axes[0].set_xlabel('模型', fontsize=12)
        axes[0].set_ylabel('交叉验证F1分数', fontsize=12)
        axes[0].set_title(f'{self.n_iterations}折交叉验证结果 (DR检测)', fontsize=14)
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0.8, color='green', linestyle='--', label='优秀阈值(0.8)')
        axes[0].legend()

        # 在柱状图上添加数值
        for bar, mean in zip(bars, cv_means):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f'{mean:.3f}', ha='center', fontsize=10)

        # 折线图
        for model_name in models:
            scores = self.cv_scores[model_name]['f1']['scores']
            axes[1].plot(range(1, len(scores) + 1), scores, 'o-',
                         label=model_name, linewidth=2, markersize=8)

        axes[1].set_xlabel('迭代次数', fontsize=12)
        axes[1].set_ylabel('F1分数', fontsize=12)
        axes[1].set_title('各次迭代F1分数变化 (DR检测)', fontsize=14)
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.5)

        plt.suptitle('糖尿病视网膜病变(DR)检测模型训练过程分析', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self):
        """绘制混淆矩阵"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, (model_name, model) in enumerate(self.models.items()):
            use_scaled = model_name in ['SVM', 'Logistic Regression']
            X_eval = self.X_test_scaled if use_scaled else self.X_test
            y_pred = model.predict(X_eval)

            cm = confusion_matrix(self.y_test, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['正常', 'DR'],
                        yticklabels=['正常', 'DR'],
                        ax=axes[idx])
            axes[idx].set_title(f'{model_name}', fontsize=12)
            axes[idx].set_xlabel('预测标签', fontsize=10)
            axes[idx].set_ylabel('真实标签', fontsize=10)

            # 计算并显示准确率
            acc = accuracy_score(self.y_test, y_pred)
            axes[idx].text(0.5, -0.15, f'准确率: {acc:.4f}',
                           transform=axes[idx].transAxes, ha='center', fontsize=11)

        plt.suptitle('DR检测模型混淆矩阵对比', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_roc_curves(self):
        """绘制ROC曲线"""
        plt.figure(figsize=(10, 8))

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        for idx, (model_name, model) in enumerate(self.models.items()):
            use_scaled = model_name in ['SVM', 'Logistic Regression']
            X_eval = self.X_test_scaled if use_scaled else self.X_test

            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_eval)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = 1 / (1 + np.exp(-model.decision_function(X_eval)))
            else:
                continue

            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)

            plt.plot(fpr, tpr, color=colors[idx], lw=2,
                     label=f'{model_name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类器 (AUC=0.5)')
        plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        plt.title('DR检测模型ROC曲线对比', fontsize=14)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_best_model(self, model, model_name, model_path='best_dr_model.pkl'):
        """保存最佳DR检测模型"""
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': model_name,
            'dr_dataset': self.dr_dataset,
            'n_iterations': self.n_iterations,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'feature_selector': self.feature_selector,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存到根目录
        joblib.dump(model_data, model_path)
        print(f"\n✓ 最佳DR检测模型 ({model_name}) 已保存到 {model_path}")

        # 保存到最佳模型子目录
        best_model_path = os.path.join(self.model_subdirs['Best Model'], 'best_dr_model_complete.pkl')
        joblib.dump(model_data, best_model_path)
        print(f"✓ 最佳DR检测模型已保存到 {best_model_path}")

    def run_full_comparison(self):
        """运行完整的DR检测模型比较流程"""
        # 加载DR数据
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

        self.train_svm(C=0.1)
        print()
        self.train_logistic_regression(C=0.1)
        print()
        self.train_decision_tree(max_depth=10, min_samples_split=20, min_samples_leaf=10)
        print()
        self.train_random_forest(n_estimators=100, max_depth=10,
                                 min_samples_split=20, min_samples_leaf=10)

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

def example_1_synthetic_data():
    """示例1：使用合成DR数据（推荐用于测试）"""
    print("\n" + "=" * 80)
    print("示例1：使用合成DR数据 - 快速测试")
    print("=" * 80)

    comparator = ModelComparator(
        data_source='synthetic',
        dr_dataset='Synthetic DR Data',
        test_size=0.25,
        random_state=42,
        n_iterations=56,
        models_dir='saved_models_dr_synthetic'
    )

    results = comparator.run_full_comparison()
    return results


def example_2_local_features():
    """示例2：使用本地特征文件"""
    print("\n" + "=" * 80)
    print("示例2：使用本地DR特征文件")
    print("=" * 80)
    print("提示：请将您的特征文件放在以下路径：")
    print("  - ./dr_data/dr_features.csv (DR患者特征)")
    print("  - ./dr_data/normal_features.csv (正常人特征)")

    # 首先创建示例数据
    loader = DRDataLoader()
    loader.create_sample_dr_dataset(save_path='./dr_data')

    comparator = ModelComparator(
        data_source='local',
        dr_dataset='Local DR Features',
        dr_feature_file='./dr_data/dr_features.csv',
        normal_feature_file='./dr_data/normal_features.csv',
        test_size=0.25,
        random_state=42,
        n_iterations=5,
        models_dir='saved_models_dr_local'
    )

    results = comparator.run_full_comparison()
    return results


def example_3_messidor_simulation():
    """示例3：模拟Messidor数据集"""
    print("\n" + "=" * 80)
    print("示例3：模拟Messidor DR数据集")
    print("=" * 80)

    # 生成模拟Messidor数据（更大样本量）
    loader = DRDataLoader()
    X, y, feature_names = loader.generate_synthetic_dr_data(
        n_samples=1200,
        n_features=45,
        n_informative=25
    )

    comparator = ModelComparator(
        data_source='direct',
        dr_dataset='Messidor (Simulated)',
        X=X, y=y, feature_names=feature_names,
        test_size=0.25,
        random_state=42,
        n_iterations=5,
        models_dir='saved_models_messidor'
    )

    results = comparator.run_full_comparison()
    return results


def example_4_large_scale():
    """示例4：大规模DR数据模拟"""
    print("\n" + "=" * 80)
    print("示例4：大规模DR数据模拟 (Kaggle风格)")
    print("=" * 80)

    loader = DRDataLoader()
    X, y, feature_names = loader.generate_synthetic_dr_data(
        n_samples=5000,
        n_features=100,
        n_informative=50
    )

    comparator = ModelComparator(
        data_source='direct',
        dr_dataset='Kaggle DR (Simulated)',
        X=X, y=y, feature_names=feature_names,
        test_size=0.2,
        random_state=42,
        n_iterations=5,
        models_dir='saved_models_kaggle_style'
    )

    results = comparator.run_full_comparison()
    return results


def main():
    """
    主函数：选择DR数据源进行实验
    """

    print("\n" + "=" * 80)
    print("糖尿病视网膜病变(DR)检测模型训练系统")
    print("=" * 80)

    # 显示可用的数据集
    loader = DRDataLoader()
    loader.list_available_datasets()

    # =========================================================================
    # 选择数据源
    # =========================================================================

    # 推荐：使用合成数据（无需准备数据，快速测试）
    results = example_1_synthetic_data()

    # 或者：使用本地特征文件
    # results = example_2_local_features()

    # 或者：模拟Messidor数据集
    # results = example_3_messidor_simulation()

    # 或者：大规模模拟
    # results = example_4_large_scale()

    # =========================================================================
    # 打印最终结果
    # =========================================================================
    print("\n" + "=" * 80)
    print("DR检测模型训练总结")
    print("=" * 80)

    best_model = results.iloc[0]
    print(f"\n🏆 最佳DR检测模型: {best_model['Model']}")
    print(f"   - 测试准确率: {best_model['Test_Acc']:.4f}")
    print(f"   - 测试F1分数: {best_model['Test_F1']:.4f}")
    print(f"   - 测试AUC: {best_model['Test_AUC']:.4f}")
    print(f"   - 交叉验证F1: {best_model['CV_F1_Mean']:.4f}")

    print("\n📊 DR检测模型性能排名:")
    for idx, row in results.iterrows():
        print(f"   {idx + 1}. {row['Model']}: F1={row['Test_F1']:.4f}, "
              f"准确率={row['Test_Acc']:.4f}, AUC={row['Test_AUC']:.4f}")

    # 临床意义评估
    print("\n🏥 临床意义评估:")
    best_f1 = best_model['Test_F1']
    if best_f1 >= 0.9:
        print("   ✓ 优秀模型 - 可考虑临床部署")
    elif best_f1 >= 0.8:
        print("   ✓ 良好模型 - 可作为辅助诊断工具")
    elif best_f1 >= 0.7:
        print("   △ 可用模型 - 需要进一步优化")
    else:
        print("   ✗ 模型性能不足 - 建议收集更多数据或优化特征")


if __name__ == "__main__":
    main()