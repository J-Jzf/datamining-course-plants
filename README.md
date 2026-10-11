# 北美植物分布的关联规则与区域聚类

本项目为我的研究生课程“数据科学与工程”报告实验，研究题目为**《基于关联规则与区域聚类的北美植物分布格局挖掘》**。使用历史 USDA / UCI Plants 分布记录，把每个物种视为一条地区集合事务：一方面发现地区共现规则，另一方面按共享植物组成划分区域。项目包含完整实验、可复用模型、推理脚本、数值结果、论文配图及方法框架图；不包含最终论文正文。

全部参数写在 Python 或命令行选项中，无需 YAML。默认数据目录为本项目下的 `plantsdata`，默认结果目录为 `outputs`。

## 快速开始

需要 Python **3.10 或更新版本**。在终端切换到项目目录：

```powershell
cd "\datamining"
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` 会完成清洗、规则发现、聚类、独立验证、所有扩展实验和全部绘图。运行过程会打印进度。相同数据、参数和随机种子可复现数值结果；算法耗时随电脑和运行负载变化。地图边界随项目提供，正常运行无需联网或重复下载数据。

结果已经保存在 `outputs` 中，想先看效果可以打开 **`outputs/index.html`** 图片浏览页，也可直接打开 `outputs/figures`。浏览页只列出本次运行生成的图片。只调整图片样式时运行：

```powershell
python visualize.py
```

单独训练、推理或绘制两个概念图：

```powershell
python train.py
python predict.py --regions OR WA --top-k 5
python predict.py --regions MT WY --top-k 5 --output outputs/results/my_prediction.csv
python draw_diagrams.py
```

默认推理示例以 Oregon、Washington 为已记录地区，输出尚未输入的候选记录地区。`score` 是匹配规则中最大的**训练集置信度**，同时输出 lift、support 和触发前件。多个规则共享物种和地区，不能作为独立概率相加，因此采用最大置信度排序。

也可按清洗后的拉丁学名查询：

```powershell
python predict.py --species "abies amabilis"
```

这种查询把该物种所有已记录地区作为输入，只显示集合外候选。没有符合规则的候选时，脚本会给出明确提示。地区代码使用大写；`NL` 表示合并后的 Newfoundland and Labrador。可用代码和全称见 `outputs/results/regions.csv`。

## 项目目录

```text
datamining/
├─ plantsdata/                    # 原始数据
│  ├─ plants.data
│  ├─ plants.names
│  └─ stateabbr.txt
├─ plantmine/
│  ├─ data.py                     # 分类层级和地区清洗、逐行溯源
│  ├─ mining.py                   # FP-growth、Apriori、规则指标与验证
│  ├─ clustering.py               # Jaccard/cosine、平均/完全连接聚类
│  ├─ experiments.py              # 核心、扩展及记录补全实验
│  ├─ inference.py                # 冻结规则的候选地区推理
│  ├─ cli.py                      # 公共命令行参数
│  ├─ maps.py                     # 地图边界与地图绘制
│  ├─ plots.py                    # 数值结果到论文配图
│  └─ diagrams.py                 # 方法框架图、研究主题介绍图
├─ assets/geodata/                # 随项目保存的地图边界与来源
├─ tests/                        # 数据、算法、统计及推理测试
├─ run_all.py                    # 一键完成实验和绘图
├─ train.py                      # 运行实验并保存模型
├─ predict.py                    # 根据已记录地区推理
├─ visualize.py                  # 不重训，重新生成所有图片
├─ draw_diagrams.py              # 单独生成 framework / 主题图
├─ requirements.txt
├─ README.md
└─ outputs/
   ├─ index.html                 # 本次运行图片浏览页
   ├─ model/                     # 冻结训练规则、地区聚类模型
   ├─ results/                   # CSV / JSON / NPZ 数值结果
   └─ figures/                   # PNG / SVG / PDF 图片
```

## 数据及预处理

原始文件实际有 **34,781** 条记录，与附带 `plants.names` 中的 22,632 条声明不一致；项目以实际文件为准，并保存审计信息。原始记录混有属汇总、种、亚种、变种等层级，直接使用会重复计算。

处理步骤：

1. 删除仅含属名的汇总记录，保留种及种下记录。
2. 将 `var.`、`ssp.`、`subsp.`、`f.` 等记录按种名合并，对地区取并集；保留具名杂交种。此步骤是名字层级归一化，不做现代分类同物异名解析。
3. 使用独立地区表修正原始元数据中颠倒的 `AL` / `AB`，补充 `PE`；将旧 `NF`、`LB` 合并为 `NL`。
4. 主实验范围明确为美国 50 州、华盛顿特区及加拿大 13 个省／地区，共 **64** 区域。`pr`、`vi`、`gl`、`dengl`、`fraspm` 不在主实验范围内，原始代码和记录数保留在审计中，不擅自认定 `gl` 与 `dengl` 是别名。
5. 同一种合并后没有任何主实验范围内记录的物种被排除。

最终得到 **21,400 × 64** 的物种—地区布尔矩阵，共 **177,950** 个记录为存在的位置。`data_audit.json` 给出计数和处理政策，`taxonomy_lineage.csv` 给出每条原始记录的处理过程，`cleaned_species.csv` 给出最终物种及其地区集合。

**矩阵中的 1 表示“有分布记录”，0 表示“此数据未记录”，不能当作确认不存在。** 数据是历史分布快照，没有气候、土地面积、采样强度、时间或本土／引入状态标签；分析解释为记录共现和组成相似性，不据此推断环境因果或生态适生概率。

## 算法流程与实验

清洗后的物种按固定种子 `2026` 随机划分：80% 训练、20% 留出验证。规则筛选和聚类均只使用训练物种；地图上全数据记录物种数属于描述统计。训练集发现的规则冻结后才进入验证集。

| 实验 | 实现和评估方式 |
|---|---|
| 关联规则主实验 | 真正的加权 FP 树及条件前缀树挖掘；项集长度 ≤3，前件为 1 或 2 个地区，后件为 1 个地区。默认 support ≥0.03、confidence ≥0.75、lift ≥1.15；同时输出 leverage、conviction。 |
| 区域聚类主实验 | 将地区视为物种组成集合，计算 Jaccard 距离，以 average linkage 聚类；在预先设定的 k=4…9 中选训练轮廓系数最高者，另保存 k=2…10 的评估结果。允许指定固定 k。 |
| 留出集规则复现 | 对全部冻结训练规则重新计算 support / confidence / lift；单侧 Fisher 精确检验，以冻结规则集合为检验族做 BH 调整。复现条件为原指标门槛、前件记录数 ≥20、q≤0.05。 |
| 记录补全推理 | 对最多 2,000 个至少有两个已记录地区的留出物种隐藏一个地区，比较规则排序、训练集地区频率排序、随机排序；报告 Recall@1/3/5 和覆盖率，无规则命中也计入总查询数。 |
| Bootstrap 稳定性 | 训练物种有放回抽样 30 次；评估冻结规则再次达到三个指标门槛的比例和指标百分位区间；固定 k 重新聚类，输出地区共分配概率及 ARI。不是规则发现后的生态因果置信区间。 |
| 按属划分验证 | 将整个属分配至训练或验证，两侧没有共享属；独立发现规则并验证，另外保存两种划分共同规则的对照表。 |
| 广布植物敏感性 | 删除训练集中分布最广的恰好 5% 物种，固定规则键和 k，重新评估规则与聚类；报告指标变化、门槛保留比例和 ARI。并列值按确定的训练行顺序处理。 |
| 距离／连接方式敏感性 | 比较 Jaccard/cosine × average/complete 四种组合；固定 k，报告各自距离下的轮廓系数和相对主实验 ARI。Ward 不用于 Jaccard 距离。 |
| 挖掘算法效率与一致性 | 真正的 Apriori join/prune 和 FP-growth，使用相同支持度及长度限制；训练子集 5,000、10,000、全部，支持度 0.03、0.06、0.10，每组合重复两次。逐个比较完整项集及计数，相同才保存结果；交替运行顺序减少顺序偏差。 |
| 支持度敏感性 | 在全训练集比较不同支持度下的项集数、规则数和留出复现比例。 |
| 空间关系扩展 | 地区代表点的大圆距离与植物组成 Jaccard 距离的 Spearman 相关；999 次联合置换地区标签，保留矩阵行列依赖，用双侧置换 p 值。代表点是近似位置，不是精确 GIS 质心。 |

Apriori 使用垂直整数位集合计数，FP-growth 使用前缀树。在仅 64 个项目且限制三项的任务中，位集合 Apriori 可能明显更快；实验不预设 FP-growth 必须获胜。耗时同时受实现方式和输入分布影响，不能据一次课程实验给出一般算法复杂度结论。

留出检验提供独立数据上的统计核验，但不同物种仍可能具有分类学相关性；按属划分是更严格的补充检查。Bootstrap 使用“物种事务”为重采样单位，图中的地区数量不是独立重复样本数。

## 修改参数

```powershell
python run_all.py --clusters 6 --output-dir outputs_k6
python run_all.py --min-support 0.05 --min-confidence 0.8 --min-lift 1.3 --output-dir outputs_strict
python run_all.py --bootstrap-repeats 100 --benchmark-repeats 3 --permutations 4999 --output-dir outputs_more
python run_all.py --data-dir plantsdata --output-dir outputs_new
```

只运行核心实验（含留出验证及记录补全）：

```powershell
python run_all.py --core-only --output-dir outputs_core
```

只运行实验，不绘图：`python run_all.py --no-plots`。仅重画概念图：`python draw_diagrams.py`。

各脚本的完整选项可用 `python xxx.py --help` 查看。修改参数时建议使用新的 `--output-dir`，便于保留既有实验；对同一输出目录重新运行会覆盖对应结果。使用自定义输出目录推理时，需要同时指定 `--model-dir`；用 `--species` 查询时还要指定该目录下的 `--dataset`。

## 结果文件

| 文件（均在 `outputs/results`） | 内容 |
|---|---|
| `run_summary.json` | 参数、软件版本、数据哈希、主要数值结果和运行时间 |
| `data_audit.json`、`taxonomy_lineage.csv`、`cleaned_species.csv` | 清洗审计、原始行溯源、物种事务 |
| `dataset.npz`、`splits.npz`、`grouped_splits.npz` | 清洗矩阵和两类训练／验证行索引 |
| `core_rules.csv`、`rules_ranked.csv` | 训练规则、独立验证指标和 q 值；排序表用于查看，推理模型使用训练规则 |
| `regions.csv`、`region_distance.npy`、`region_linkage.npy`、`cluster_selection.csv` | 地区信息、聚类标签、距离、树和 k 的评估 |
| `bootstrap_rules.csv`、`bootstrap_clusters.csv`、`bootstrap_coassignment.npy` | 规则复现频率、指标区间、聚类 ARI 和共分配概率 |
| `grouped_validation_rules.csv`、`matched_split_rules.csv` | 按属验证，以及两种划分共同规则的比较 |
| `broad_species_rules.csv`、`broad_species_summary.json`、`broad_species_labels.npy` | 删除广布植物后的规则和聚类变化 |
| `metric_sensitivity.csv` | 距离和连接方式对照 |
| `algorithm_benchmark.csv`、`support_sensitivity.csv` | 运行时间、一致性断言与支持度对照 |
| `missing_region_queries.csv`、`missing_region_summary.csv` | 每次隐藏地区查询和补全指标 |
| `spatial_pairs.csv`、`spatial_test.json` | 地理／组成距离及置换检验 |

`outputs/model` 保存 `rules.csv`、`model.json`、`regions.csv`、距离和聚类树。规则的 `antecedent` 用 `|` 分隔，`consequent` 为单个地区。聚类编号从 0 开始，图片显示为 1…k；编号只用于标识组别，没有大小顺序。

## 图像输出

所有图片在 `outputs/figures`，每幅同时生成 **PNG（300 dpi）、SVG、PDF**。图片没有整图标题，保留坐标、图例、面板编号和必要标注。SVG / PDF 便于论文排版和后续修改；地图为实际行政区边界，framework 及主题图为 Python 绘制的矢量图。

| 图片文件名（不含格式后缀） | 内容 |
|---|---|
| `method_framework` | 数据处理、两条方法分支与核心／扩展验证 |
| `plant_topic_overview` | 真实物种示例、分布矩阵、规则与区域分组 |
| `01_region_clusters` | 区域聚类地图，含 AK / HI 及东北部细节 |
| `02_recorded_richness` | 地区记录物种数地图 |
| `03_stable_rule_geography` | 独立验证及 Bootstrap 稳定规则的地理箭头 |
| `04_region_similarity_dendrogram` | 聚类树、相似度热图与 k 的评估 |
| `05_floristic_embedding` | 组成距离的二维展示，位置不是地理坐标 |
| `06_rule_support_confidence` | 规则指标气泡图，分别展示两种前件长度 |
| `07_rule_heldout_validation` | 训练集与留出集指标的配对比较 |
| `08_bootstrap_stability` | 地区共分配概率和重采样聚类稳定性 |
| `09_rule_stability` | 规则指标的重采样区间与重现频率 |
| `10_algorithm_efficiency` | 相同结果下的算法耗时对照 |
| `11_genus_validation` | 两种划分共同规则的留出指标比较 |
| `12_broad_species_sensitivity` | 移除广布物种后的指标变化 |
| `13_metric_support_sensitivity` | 距离、连接方式和支持度对照 |
| `14_spatial_relation` | 地理距离与组成距离的关系 |
| `16_masked_region_completion` | 隐藏地区后的 Recall@k 对比 |

主要地图、框架图和相似度图可以优先用于论文；扩展图用于解释结果稳定性及条件。概念图里的植物图形是程序绘制的示意符号，物种名字和矩阵示例来自真实记录。

## 已验证的默认运行

本机已完整执行 `python run_all.py`，包含所有扩展与绘图，约 **2.9 分钟**。实际生成 **17 幅图 × 3 种格式**，图片浏览页无失效链接；完整测试为 **17 passed**。

| 指标 | 本次默认结果 |
|---|---:|
| 清洗后物种／地区 | 21,400 ／ 64 |
| 随机划分训练／留出物种 | 17,120 ／ 4,280 |
| 训练频繁项集／规则 | 18,854 ／ 27,037 |
| 通过留出复现检查的规则 | 25,038 |
| 所选 k／训练轮廓系数 | 4 ／ 0.221 |
| Bootstrap 平均聚类 ARI | 0.921 |
| 广布物种删除后的规则门槛保留比例 | 16.6% |
| 规则／地区频率／随机的 Recall@5 | 62.25% ／ 31.95% ／ 10.05% |

这些结果对应上面的默认参数；改变参数后，以新输出目录的 CSV 和 `run_summary.json` 为准。推理示例输出可参考 `outputs/results/prediction_example.csv`。

## 测试与来源

```powershell
python -m pytest -q
```

测试覆盖物种层级合并、地区修正、算法与穷举结果一致性、规则指标、验证集零前件、空地区距离、聚类距离一致性、按属划分无交叉、BH 调整、空间矩阵置换及完整前件匹配推理。

- [UCI Plants 数据集](https://archive.ics.uci.edu/dataset/180/plants)：历史 USDA 分布事务数据，UCI DOI 为 [10.24432/C5HS40](https://doi.org/10.24432/C5HS40)，UCI 页面标注 CC BY 4.0。原始来源请引用 USDA, NRCS. 2008. *The PLANTS Database*, National Plant Data Center, Baton Rouge, LA, USA（原始记录日期为 2008-12-31）。
- [USDA PLANTS](https://plants.usda.gov/)：原始数据库。本项目使用用户下载的历史快照，不通过当前网站重新抓取分布。
- [Natural Earth 1:50m 行政区边界](https://www.naturalearthdata.com/downloads/50m-cultural-vectors/)：地图底图，公共领域；具体来源和随项目保存的文件见 `assets/geodata`。
- [SciPy 层次聚类](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)、[scikit-learn silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)：聚类和评估实现。
