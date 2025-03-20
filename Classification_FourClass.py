#########################################################################
#
# Copyright 2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#########################################################################

############################## 库导入 ################################
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

############################# 训练参数设置 ###############################
# 设置随机种子，确保结果可重现
np.random.seed(1)

# 交叉验证参数
K = 10                    # 交叉验证折数，将数据集分成10份进行训练和测试

# 神经网络结构参数
number_inner_layers  = 3  # 隐藏层数量，设置3层隐藏层
number_inner_neurons = 256  # 每层神经元数量，实际使用的是这个数的一半，即128个神经元
inner_activation_fun = 'relu'  # 隐藏层激活函数，使用ReLU函数提高非线性表达能力和减轻梯度消失问题
outer_activation_fun = 'softmax'  # 输出层激活函数，使用Softmax函数进行多分类

# 优化器参数
optimizer_loss_fun   = 'categorical_crossentropy'  # 损失函数，使用多类别交叉熵
optimizer_algorithm  = 'adam'  # 优化算法，使用Adam优化器，结合了动量和自适应学习率

# 训练参数
number_epoch         = 30  # 训练轮数，每个样本将被训练30次
batch_length         = 10  # 批量大小，每次更新权重使用10个样本
show_inter_results   = 1   # 是否显示中间结果，1表示显示训练过程的简要信息

# 文件路径设置
data_path = r"D:\code\drone\dronerf\output\labeled_data\RF_Data.csv"  # 标记数据的路径
results_path = r"D:\code\drone\dronerf\output\result4"  # 四分类结果的保存路径

############################## 函数 ###############################
def decode(datum):
    """
    将独热编码(one-hot)转换回标签索引
    参数:
        datum: 独热编码的标签数组
    返回:
        y: 标签索引数组
    """
    y = np.zeros((datum.shape[0],1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y

def encode(datum):
    """
    将标签索引转换为独热编码(one-hot)
    参数:
        datum: 标签索引数组
    返回:
        独热编码的标签数组
    """
    return to_categorical(datum)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    """
    绘制并保存混淆矩阵
    参数:
        cm: 混淆矩阵
        classes: 类别名称列表
        normalize: 是否归一化
        title: 图表标题
        cmap: 颜色映射
        save_path: 保存路径，如果为None则显示图表
    """
    plt.figure(figsize=(10, 8))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title + ' (normalized)')
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
    
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_test, y_score, n_classes, save_path=None):
    """
    绘制并保存ROC曲线
    参数:
        y_test: 测试集的真实标签（二值化后）
        y_score: 预测的概率分数
        n_classes: 类别数量
        save_path: 保存路径，如果为None则显示图表
    """
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 绘制所有ROC曲线
    plt.figure(figsize=(10, 8))
    
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
    for i, color in zip(range(n_classes), colors[:n_classes]):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_history(history, fold, save_path=None):
    """
    绘制并保存训练历史曲线（损失和准确率）
    参数:
        history: Keras训练历史对象
        fold: 当前折数
        save_path: 保存路径，如果为None则显示图表
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制损失曲线
    ax1.plot(history.history['loss'])
    ax1.set_title(f'Model Loss (Fold {fold})')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(history.history['accuracy'])
    ax2.set_title(f'Model Accuracy (Fold {fold})')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_average_metrics(all_histories, save_path=None):
    """
    绘制并保存所有折叠的平均训练指标
    参数:
        all_histories: 所有折叠的训练历史列表
        save_path: 保存路径，如果为None则显示图表
    """
    # 提取每个折叠的损失和准确率
    all_loss = []
    all_acc = []
    
    for history in all_histories:
        all_loss.append(history.history['loss'])
        all_acc.append(history.history['accuracy'])
    
    # 计算平均值
    avg_loss = np.mean(all_loss, axis=0)
    avg_acc = np.mean(all_acc, axis=0)
    
    # 计算标准差
    std_loss = np.std(all_loss, axis=0)
    std_acc = np.std(all_acc, axis=0)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # 绘制平均损失曲线
    epochs = range(1, len(avg_loss) + 1)
    ax1.plot(epochs, avg_loss, 'b-', label='Average Loss')
    ax1.fill_between(epochs, avg_loss - std_loss, avg_loss + std_loss, alpha=0.2, color='b')
    ax1.set_title('Average Model Loss Across All Folds')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制平均准确率曲线
    ax2.plot(epochs, avg_acc, 'r-', label='Average Accuracy')
    ax2.fill_between(epochs, avg_acc - std_acc, avg_acc + std_acc, alpha=0.2, color='r')
    ax2.set_title('Average Model Accuracy Across All Folds')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

############################### 主程序 ##################################
# 创建输出目录（如果不存在）
if not os.path.exists(results_path):
    os.makedirs(results_path)

# 创建训练曲线保存目录
curves_path = os.path.join(results_path, "training_curves")
if not os.path.exists(curves_path):
    os.makedirs(curves_path)

# 加载数据
print("Loading data ...")
Data = np.loadtxt(data_path, delimiter=",")

# 准备数据
print("Preparing data ...")
x = np.transpose(Data[0:2047,:])  # 特征数据
Label_2 = np.transpose(Data[2049:2050,:]); Label_2 = Label_2.astype(int)  # 四分类标签（背景/Bebop/AR/Phantom）

# 使用标签2（四分类）
label_to_use = Label_2
y = encode(label_to_use)  # 将标签转换为独热编码
n_classes = y.shape[1]  # 类别数量

# 定义类别名称
class_names = ["Background", "Bebop", "AR", "Phantom"]

# 初始化结果收集变量
cvscores = []  # 存储每折的准确率
cnt = 0  # 折数计数器
all_y_test = []  # 存储所有测试集的真实标签
all_y_pred = []  # 存储所有测试集的预测标签
all_y_pred_prob = []  # 存储所有测试集的预测概率
all_histories = []  # 存储所有折叠的训练历史

# 使用分层K折交叉验证
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
for train, test in kfold.split(x, decode(y)):
    cnt = cnt + 1
    print(f"Fold {cnt}/{K}")
    
    # 构建神经网络模型
    model = Sequential()
    # 添加隐藏层
    for i in range(number_inner_layers):
        model.add(Dense(int(number_inner_neurons/2), input_dim=x.shape[1], activation=inner_activation_fun))
    # 添加输出层
    model.add(Dense(y.shape[1], activation=outer_activation_fun))
    
    # 编译模型
    model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])
    
    # 训练模型并保存历史
    history = model.fit(x[train], y[train], epochs=number_epoch, batch_size=batch_length, verbose=show_inter_results)
    all_histories.append(history)
    
    # 绘制并保存当前折叠的训练曲线
    plot_training_history(history, cnt, save_path=os.path.join(curves_path, f"training_curves_fold_{cnt}.png"))
    
    # 评估模型
    scores = model.evaluate(x[test], y[test], verbose=show_inter_results)
    print(f"Accuracy: {scores[1]*100:.2f}%")
    cvscores.append(scores[1]*100)
    
    # 预测概率
    y_pred_prob = model.predict(x[test])
    # 预测类别
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_test_decoded = np.argmax(y[test], axis=1)
    
    # 保存预测结果
    np.savetxt(os.path.join(results_path, f"Results_fold_{cnt}.csv"), 
               np.column_stack((y[test], y_pred_prob)), delimiter=",", fmt='%s')
    
    # 收集所有折叠的结果用于最终评估
    all_y_test.extend(y_test_decoded)
    all_y_pred.extend(y_pred)
    all_y_pred_prob.append(y_pred_prob)

# 绘制并保存所有折叠的平均训练指标
plot_average_metrics(all_histories, save_path=os.path.join(curves_path, "average_training_metrics.png"))

# 打印平均准确率
print(f"Average Accuracy: {np.mean(cvscores):.2f}% ± {np.std(cvscores):.2f}%")

# 将结果转换为numpy数组
all_y_test = np.array(all_y_test)
all_y_pred = np.array(all_y_pred)
all_y_pred_prob = np.vstack(all_y_pred_prob)

# 生成分类报告
report = classification_report(all_y_test, all_y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

# 保存分类报告到文本文件
with open(os.path.join(results_path, "classification_report.txt"), "w") as f:
    f.write(report)

# 计算混淆矩阵
cm = confusion_matrix(all_y_test, all_y_pred)

# 绘制并保存混淆矩阵
plot_confusion_matrix(cm, classes=class_names, 
                     title='Confusion Matrix', 
                     save_path=os.path.join(results_path, "confusion_matrix.png"))
plot_confusion_matrix(cm, classes=class_names, normalize=True, 
                     title='Normalized Confusion Matrix', 
                     save_path=os.path.join(results_path, "normalized_confusion_matrix.png"))

# 为ROC曲线准备二值化标签
y_test_bin = label_binarize(all_y_test, classes=range(n_classes))

# 绘制并保存ROC曲线
plot_roc_curve(y_test_bin, all_y_pred_prob, n_classes, 
              save_path=os.path.join(results_path, "roc_curve.png"))

print(f"\nAll results have been saved to: {results_path}")
print(f"Training curves have been saved to: {curves_path}")
######################################################################### 