[README.md](https://github.com/user-attachments/files/29468187/README.md)
# -
testing
# 图书类目自动标引系统

本项目为机器学习课程综合实践项目，主要完成图书文本数据的自动类目标引。程序以图书标题、关键词、摘要等字段为输入，通过数据清洗、文本特征提取、模型训练、模型评估和 Streamlit 可视化展示，实现对图书类别的自动预测。

## 项目简介

项目面向中文图书类目分类任务，使用 TF-IDF 将文本内容转换为特征向量，并对比多种机器学习分类算法。实验流程包括原始数据读取、缺失值处理、重复值处理、异常数据检查、特征工程、模型训练、性能评估和结果可视化。

当前实验中使用的主要模型包括：

- Multinomial Naive Bayes
- Rocchio
- Sparse Perceptron

经过特征组合和参数优化后，最优模型为 MultinomialNB，最终测试集准确率达到 73.48%。

## 项目结构

```text
ml_chapter3_project/
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── outputs/
│   ├── full_ml_project.py
│   ├── streamlit_app.py
│   ├── model_comparison_results.csv
│   ├── hyperparameter_tuning_results.csv
│   ├── best_model_summary.json
│   └── visualizations/
├── requirements_streamlit.txt
└── README.md
```

## 主要功能

- 读取训练集、验证集和测试集数据
- 对文本字段进行清洗和格式统一
- 处理缺失值、重复值和异常样本
- 使用 TF-IDF 提取文本特征
- 使用卡方检验进行特征选择
- 训练并对比多种机器学习分类模型
- 输出准确率、宏平均 Precision、Recall、F1 和 AUC
- 生成模型对比图、混淆矩阵、ROC 曲线和特征重要性图
- 使用 Streamlit 展示实验结果

## 环境依赖

建议使用 Python 3.9 及以上版本。

安装依赖：

```bash
pip install -r requirements_streamlit.txt
```

如果依赖文件不完整，可手动安装：

```bash
pip install pandas numpy matplotlib streamlit scikit-learn
```

## 运行方式

进入项目目录后，运行完整实验：

```bash
python outputs/full_ml_project.py
```

启动可视化页面：

```bash
streamlit run outputs/streamlit_app.py
```

运行后可在浏览器中查看模型对比结果、调参结果、混淆矩阵、ROC 曲线和特征重要性分析。

## 实验结果

模型优化前，MultinomialNB 在测试集上的准确率约为 59.44%。经过文本字段组合调整、TF-IDF 参数优化和朴素贝叶斯平滑参数调整后，模型性能得到提升。

最终最优结果如下：

| 模型 | 参数 | 测试集 Accuracy | 测试集 Macro-F1 |
| --- | --- | ---: | ---: |
| MultinomialNB | alpha=0.1 | 73.48% | 64.20% |

该结果说明，标题、关键词和摘要字段对图书类目识别具有较强区分能力，合理的特征筛选和参数设置能够明显提升分类效果。

## 可视化展示

Streamlit 页面主要包含以下内容：

- 数据集基本信息
- 数据清洗结果
- 模型性能对比表
- 超参数调优结果
- 混淆矩阵
- ROC 曲线
- 特征重要性分析

## 项目说明

本项目用于课程实验和学习交流，重点展示传统机器学习方法在中文文本分类任务中的完整建模流程。由于数据集类别数量较多、部分类别样本数量有限，模型仍存在一定误分类情况。后续可通过扩充训练数据、优化分词效果、引入 SVM 或 BERT 等模型进一步提升分类性能。
