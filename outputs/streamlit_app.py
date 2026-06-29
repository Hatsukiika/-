from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
VIS_DIR = APP_DIR / "visualizations"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def show_svg(path: Path) -> None:
    if path.exists():
        svg = path.read_text(encoding="utf-8")
        components.html(svg, height=680, scrolling=True)
    else:
        st.warning(f"未找到图表：{path.name}")


st.set_page_config(page_title="图书类目自动标引实验", layout="wide")
st.title("机器学习实验课程：图书类目自动标引")

cleaning = load_json(APP_DIR / "data_cleaning_report.json")
best = load_json(APP_DIR / "best_model_summary.json")
model_results = pd.read_csv(APP_DIR / "model_comparison_results.csv")
tuning_results = pd.read_csv(APP_DIR / "hyperparameter_tuning_results.csv")
features = pd.read_csv(APP_DIR / "feature_importance_best.csv")

st.header("1. 数据概览与清洗")
col1, col2, col3, col4 = st.columns(4)
col1.metric("原始样本数", cleaning["raw_rows"])
col2.metric("清洗后样本数", cleaning["clean_rows"])
col3.metric("删除重复样本", cleaning["duplicate_rows_removed"])
col4.metric("删除异常样本", cleaning["outlier_rows_removed"])

st.subheader("清洗策略")
st.write(
    "文本字段缺失值填充为空字符串，文件路径缺失填充为 unknown；"
    "标签统一转换为二级类目，例如 F270 -> F2；"
    "文本长度异常值使用 IQR 规则处理。"
)

left, right = st.columns(2)
with left:
    show_svg(VIS_DIR / "eda_label_distribution.svg")
with right:
    show_svg(VIS_DIR / "eda_missing_values.svg")
show_svg(VIS_DIR / "eda_correlation_heatmap.svg")

st.header("2. 特征工程与特征选择")
st.write(
    f"使用标题分词、作者关键词、摘要分词构造 TF-IDF 特征；"
    f"在训练集上用卡方统计量筛选特征，最终特征数为 {best['n_features']}。"
)
st.dataframe(pd.read_csv(APP_DIR / "feature_selection_chi2.csv").head(30), use_container_width=True)

st.header("3. 模型训练与对比")
st.write(
    "候选模型包括 MultinomialNB、Rocchio、SparsePerceptron。"
    "评估指标包括 Accuracy、Macro-Precision、Macro-Recall、Macro-F1 和 Macro-AUC。"
)
st.dataframe(model_results, use_container_width=True)
show_svg(VIS_DIR / "model_comparison.svg")

st.header("4. 最优模型调参")
st.write(f"调参后最优模型：**{best['model']}**，参数：`{best['params']}`")
metric_cols = st.columns(4)
metric_cols[0].metric("Val Accuracy", f"{best['val_metrics']['accuracy']:.4f}")
metric_cols[1].metric("Val Macro-F1", f"{best['val_metrics']['macro_f1']:.4f}")
metric_cols[2].metric("Test Accuracy", f"{best['test_metrics']['accuracy']:.4f}")
metric_cols[3].metric("Test Macro-F1", f"{best['test_metrics']['macro_f1']:.4f}")
st.dataframe(tuning_results, use_container_width=True)

st.header("5. 可视化分析")
tab1, tab2, tab3 = st.tabs(["混淆矩阵", "ROC 曲线", "特征重要性"])
with tab1:
    show_svg(VIS_DIR / "confusion_matrix_best.svg")
with tab2:
    show_svg(VIS_DIR / "roc_curve_best.svg")
with tab3:
    show_svg(VIS_DIR / "feature_importance_best.svg")
    st.dataframe(features.head(30), use_container_width=True)

st.header("6. 结论")
st.write(
    "朴素贝叶斯适合词频特征，训练速度快；Rocchio 适合作为文本分类基线；"
    "SparsePerceptron 能学习线性分类边界，但对类别不均衡和训练轮数更敏感。"
    "本实验以验证集 Macro-F1 选择最优模型，兼顾多类别不均衡场景下的公平评估。"
)
