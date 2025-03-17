# DroneRF 数据处理与分析系统

## 文件说明与运行顺序

本系统包含5个主要文件，用于处理和分析无人机RF信号数据。以下是各文件的作用和运行顺序：

### 1. Main_1_Data_aggregation.m

**功能**：数据聚合，将原始CSV文件处理并聚合为MAT文件。

**输入**：
- 原始数据集路径：`D:\code\dronerf\DroneRF\`
  - 包含四个子文件夹：Background、Bepop、AR、Phantom
  - 每个子文件夹直接包含CSV文件
  - 文件命名格式如：`10100H_0.csv`和`10100L_0.csv`

**输出**：
- 聚合数据路径：`D:\code\dronerf\output\aggregated_data\`
- 输出文件格式：`[BUI代码].mat`（例如：`00000.mat`、`10000.mat`等）

**数据集结构**：
```
DroneRF/
├── Background/
│   ├── 00000H_0.csv
│   ├── 00000L_0.csv
│   ├── 00000H_1.csv
│   ├── 00000L_1.csv
│   └── ...
├── Bepop/
│   ├── 10000H_0.csv
│   ├── 10000L_0.csv
│   ├── 10001H_0.csv
│   ├── 10001L_0.csv
│   └── ...
├── AR/
│   ├── 10100H_0.csv
│   ├── 10100L_0.csv
│   ├── 10101H_0.csv
│   ├── 10101L_0.csv
│   └── ...
└── Phantom/
    ├── 11000H_0.csv
    ├── 11000L_0.csv
    ├── 11000H_1.csv
    ├── 11000L_1.csv
    └── ...
```

### 2. Main_2_Data_labeling.m

**功能**：数据标记，为聚合后的数据添加标签。

**输入**：
- 聚合数据路径：`D:\code\dronerf\output\aggregated_data\`

**输出**：
- 标记数据路径：`D:\code\dronerf\output\labeled_data\`
- 输出文件：`RF_Data.csv`（包含数据和标签）

### 3. Demo_3_Analysis.m

**功能**：数据分析，生成时域图、频谱图和箱线图。

**输入**：
- 聚合数据路径：`D:\code\dronerf\output\aggregated_data\`

**输出**：
- 分析结果路径：`D:\code\dronerf\output\analysis\`
- 输出文件：
  - `TimeDomain_[类别].png`：各类别的时域图
  - `Spectrum_[类别].png`：各类别的频谱图
  - `Box_All.png`：所有类别的箱线图对比

### 4. Classification.py

**功能**：使用深度学习方法对RF信号进行分类。

**输入**：
- 标记数据路径：`D:\code\dronerf\output\labeled_data\RF_Data.csv`

**输出**：
- 分类结果路径：`D:\code\dronerf\output\classification_results\`
- 输出文件：
  - `confusion_matrix.png`：混淆矩阵
  - `normalized_confusion_matrix.png`：归一化混淆矩阵
  - `roc_curve.png`：ROC曲线
  - `classification_report.txt`：分类报告
  - `training_curves/`：训练曲线图

### 5. plot_rf_signals.py

**功能**：从聚合数据生成详细的时域和频域图。

**输入**：
- 聚合数据路径：`D:\code\dronerf\output\aggregated_data\`

**输出**：
- 信号分析结果路径：`D:\code\dronerf\output\signal_analysis\`
- 输出文件：
  - `TimeDomain_[BUI代码].png`：各信号的时域图
  - `Spectrum_[BUI代码].png`：各信号的频谱图

## 参数配置

### plot_rf_signals.py 主要参数：
- 采样率：40 MHz
- 信号增益：30 dB
- 时域图显示时间：0.25秒
- 频率范围：2400-2480 MHz（2.4GHz频段）

## 数据类别说明

系统处理的数据包括四类无人机RF信号：
1. 背景信号（Background）：BUI代码 00000
2. Bebop无人机：BUI代码 10000, 10001, 10010, 10011
3. AR无人机：BUI代码 10100, 10101, 10110, 10111
4. Phantom无人机：BUI代码 11000

## 运行流程

1. 运行 `Main_1_Data_aggregation.m` 进行数据聚合
2. 运行 `Main_2_Data_labeling.m` 进行数据标记
3. 运行 `Demo_3_Analysis.m` 进行数据分析和可视化
4. 运行 `Classification.py` 进行数据分类
5. 运行 `plot_rf_signals.py` 生成详细的信号图

## 输出目录结构

```
output/
├── aggregated_data/
│   ├── 00000.mat
│   ├── 10000.mat
│   └── ...
├── labeled_data/
│   └── RF_Data.csv
├── analysis/
│   ├── TimeDomain_Background.png
│   ├── Spectrum_Background.png
│   └── ...
├── classification_results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── ...
└── signal_analysis/
    ├── TimeDomain_00000.png
    ├── Spectrum_00000.png
    └── ...
```

## 注意事项

1. 请确保所有输入路径中包含正确的数据文件
2. 各脚本中的路径可能需要根据实际环境进行调整
3. MATLAB脚本需要MATLAB环境运行，Python脚本需要Python环境及相关依赖库
4. 文件命名中的BUI代码（如00000、10000等）表示不同的无人机类型和活动模式
5. 每个CSV文件对应一段RF信号记录，H和L后缀分别表示高频和低频部分