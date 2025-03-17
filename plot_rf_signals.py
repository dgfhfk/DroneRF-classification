import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import glob

# =========================================================
# 配置参数（所有可调整的参数都集中在这里）
# =========================================================

# 文件路径配置
INPUT_PATH = r'D:\code\dronerf\output\aggregated_data'  # 输入数据路径（mat文件所在目录）
OUTPUT_PATH = r'D:\code\dronerf\output\signal_analysis'  # 输出数据路径（图像保存目录）

# 信号处理参数
SAMPLE_RATE = 40e6  # 采样率 (Hz)，用于计算时间和频率 - 已更新为40 MHz
DISPLAY_TIME = 0.25  # 时域图显示时间（秒）- 可以调整此参数来改变时域图显示的时间长度
GAIN_DB = 30        # 信号增益 (dB) - 用于放大信号
NPERSEG = 1024      # STFT分段长度，影响频谱图的频率分辨率
NOVERLAP = 512      # STFT重叠长度，影响频谱图的时间分辨率

# 频率范围参数 (MHz)
FREQ_MIN = 2400     # 最小显示频率 (MHz)，2.4GHz频段下限
FREQ_MAX = 2480     # 最大显示频率 (MHz)，2.4GHz频段上限

# 图像配置
FIG_SIZE = (12, 8)  # 图像大小（宽度，高度），单位为英寸

# BUI代码到无人机类型的映射
BUI_MAPPING = {
    '00000': '背景信号',     # 背景RF活动
    '10000': 'Bebop无人机',  # Bebop无人机RF活动
    '10001': 'Bebop无人机',  # Bebop无人机RF活动
    '10010': 'Bebop无人机',  # Bebop无人机RF活动
    '10011': 'Bebop无人机',  # Bebop无人机RF活动
    '10100': 'AR无人机',     # AR无人机RF活动
    '10101': 'AR无人机',     # AR无人机RF活动
    '10110': 'AR无人机',     # AR无人机RF活动
    '10111': 'AR无人机',     # AR无人机RF活动
    '11000': 'Phantom无人机' # Phantom无人机RF活动
}
# =========================================================

def plot_signal(data, title, save_path, is_time_domain=True):
    """
    绘制时域或频域信号
    
    参数:
        data: 信号数据
        title: 图表标题
        save_path: 图像保存路径
        is_time_domain: 如果为True，绘制时域图；否则绘制频域图
    """
    plt.figure(figsize=FIG_SIZE)
    
    # 应用增益
    data_amplified = data * (10 ** (GAIN_DB / 20))  # 将dB增益转换为线性增益并应用
    
    if is_time_domain:
        # 时域图 - 显示指定时间的数据
        # 注意：可以通过调整DISPLAY_TIME参数来改变显示的时间长度
        
        # 打印数据长度信息，帮助调试
        print(f"数据长度: {len(data_amplified)} 点")
        
        # 计算实际可显示的样本点数
        samples_for_display = min(int(SAMPLE_RATE * DISPLAY_TIME), len(data_amplified))
        
        # 打印实际显示的时间长度
        actual_time = samples_for_display / SAMPLE_RATE
        print(f"实际显示时间: {actual_time:.6f} 秒 ({samples_for_display} 点)")
        
        # 生成时间轴（秒）
        time = np.arange(samples_for_display) / SAMPLE_RATE
        
        # 绘制时域图
        plt.plot(time, data_amplified[:samples_for_display])
        plt.title(f'{title} - Time Domain ({actual_time:.3f}s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
    else:
        # 频域图 - 使用FFT计算频谱
        n = len(data_amplified)
        freq = np.fft.rfftfreq(n, d=1/SAMPLE_RATE) / 1e6  # 频率轴（MHz）
        spectrum = np.abs(np.fft.rfft(data_amplified))
        
        # 归一化频谱
        spectrum_db = 20 * np.log10(spectrum / np.max(spectrum))
        
        # 查找感兴趣频率范围的索引
        freq_indices = np.where((freq >= FREQ_MIN) & (freq <= FREQ_MAX))[0]
        
        if len(freq_indices) > 0:
            # 只绘制感兴趣的频率范围
            plt.plot(freq[freq_indices], spectrum_db[freq_indices])
            plt.title(f'{title} - Frequency Spectrum (Gain: {GAIN_DB} dB)')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Power (dB)')
            plt.grid(True)
            plt.xlim(FREQ_MIN, FREQ_MAX)  # 设置x轴范围为指定的频率范围
            plt.ylim(-80, 5)  # 设置y轴范围，使图像更清晰
        else:
            # 如果没有找到感兴趣的频率范围，则绘制全频谱
            plt.plot(freq, spectrum_db)
            plt.title(f'{title} - Frequency Spectrum (Full Range, Gain: {GAIN_DB} dB)')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Power (dB)')
            plt.grid(True)
            plt.ylim(-80, 5)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """主函数：处理所有.mat文件并生成时域和频域图"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # 获取输入目录中的所有.mat文件
    mat_files = glob.glob(os.path.join(INPUT_PATH, '*.mat'))
    
    if not mat_files:
        print(f"在 {INPUT_PATH} 中没有找到.mat文件")
        return
    
    print(f"找到 {len(mat_files)} 个.mat文件，采样率: {SAMPLE_RATE/1e6} MHz，增益: {GAIN_DB} dB")
    
    # 使用tqdm创建进度条
    with tqdm(total=len(mat_files), desc="处理MAT文件") as pbar:
        # 处理每个.mat文件
        for mat_file in mat_files:
            file_name = os.path.basename(mat_file).split('.')[0]
            pbar.set_description(f"处理 {file_name}")
            
            try:
                # 加载.mat文件
                mat_data = sio.loadmat(mat_file)
                
                # 提取数据（假设'Data'是.mat文件中的变量名）
                if 'Data' in mat_data:
                    data = mat_data['Data']
                    
                    # 打印数据形状，帮助调试
                    print(f"\n文件 {file_name} 的数据形状: {data.shape}")
                    
                    # 从BUI代码获取无人机类型
                    drone_type = BUI_MAPPING.get(file_name, '未知类型')
                    
                    # 创建标题
                    title = f"{drone_type} ({file_name})"
                    
                    # 对于时域图，使用数据的第一列
                    if data.shape[1] > 0:
                        # 时域图
                        time_domain_path = os.path.join(OUTPUT_PATH, f'TimeDomain_{file_name}.png')
                        plot_signal(data[:, 0], title, time_domain_path, is_time_domain=True)
                        
                        # 频域图
                        freq_domain_path = os.path.join(OUTPUT_PATH, f'Spectrum_{file_name}.png')
                        plot_signal(data[:, 0], title, freq_domain_path, is_time_domain=False)
                        
                        print(f"已保存: {time_domain_path}")
                        print(f"已保存: {freq_domain_path}")
                    else:
                        print(f"警告: {file_name} 中没有数据列")
                else:
                    print(f"警告: 在 {file_name} 中未找到'Data'变量")
            except Exception as e:
                print(f"处理文件 {mat_file} 时出错: {str(e)}")
            
            # 更新进度条
            pbar.update(1)

if __name__ == '__main__':
    main()