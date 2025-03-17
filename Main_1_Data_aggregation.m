%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%
%
% Copyright 2019 Mohammad Al-Sa'd
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
%
% Authors: Mohammad F. Al-Sa'd (mohammad.al-sad@tuni.fi)
%          Amr Mohamed         (amrm@qu.edu.qa)
%          Abdulla Al-Ali
%          Tamer Khattab
%
% The following reference should be cited whenever this script is used:
%     M. Al-Sa'd et al. "RF-based drone detection and identification using
%     deep learning approaches: an initiative towards a large open source
%     drone database", 2019.
%
% Last Modification: 12-02-2019
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%

close all; clear; clc

% 设置输入和输出路径
base_path = 'D:\code\dronerf\';
save_filename = [base_path 'output\aggregated_data\'];   % 聚合数据的保存路径

% 创建输出目录（如果不存在）
if ~exist(save_filename, 'dir')
    mkdir(save_filename);
end

%% 参数
% 定义数据类别和对应的BUI代码
BUI{1,1} = {'00000'};                         % BUI of RF background activities
BUI{1,2} = {'10000','10001','10010','10011'}; % BUI of the Bebop drone RF activities
BUI{1,3} = {'10100','10101','10110','10111'}; % BUI of the AR drone RF activities
BUI{1,4} = {'11000'};                         % BUI of the Phantom drone RF activities

% 定义文件夹映射（使用绝对路径）
folder_map = containers.Map();
% 背景
folder_map('00000') = 'D:\code\dronerf\DroneRF\Background\';
% Bebop
folder_map('10000') = 'D:\code\dronerf\DroneRF\Bepop\';
folder_map('10001') = 'D:\code\dronerf\DroneRF\Bepop\';
folder_map('10010') = 'D:\code\dronerf\DroneRF\Bepop\';
folder_map('10011') = 'D:\code\dronerf\DroneRF\Bepop\';
% AR
folder_map('10100') = 'D:\code\dronerf\DroneRF\AR\';
folder_map('10101') = 'D:\code\dronerf\DroneRF\AR\';
folder_map('10110') = 'D:\code\dronerf\DroneRF\AR\';
folder_map('10111') = 'D:\code\dronerf\DroneRF\AR\';
% Phantom
folder_map('11000') = 'D:\code\dronerf\DroneRF\Phantom\';

M = 2048; % 频率分箱总数
L = 1e5;  % 每段样本总数
Q = 10;   % 频谱连续性的返回点数

%% 主程序
for opt = 1:length(BUI)
    % 加载和平均
    for b = 1:length(BUI{1,opt})
        bui_code = BUI{1,opt}{b};
        disp(['处理 BUI 代码: ' bui_code]);
        
        if(strcmp(bui_code,'00000'))
            N = 40; % 背景RF活动的段数
        elseif(strcmp(bui_code,'10111'))
            N = 17; % AR无人机的特定型号段数
        else
            N = 20; % 其他无人机RF活动的段数
        end
        
        data = [];
        cnt = 1;
        
        for n = 0:N
            % 加载原始csv文件
            try
                % 文件命名格式：10100H_0.csv 和 10100L_0.csv
                x_file = [folder_map(bui_code) bui_code 'L_' num2str(n) '.csv'];
                y_file = [folder_map(bui_code) bui_code 'H_' num2str(n) '.csv'];
                
                % 检查文件是否存在
                if ~exist(x_file, 'file') || ~exist(y_file, 'file')
                    warning(['文件不存在: ' x_file ' 或 ' y_file]);
                    continue;
                end
                
                x = csvread(x_file);
                y = csvread(y_file);
                
                % 重新分段和信号变换
                for i = 1:floor(length(x)/L)
                    st = 1 + (i-1)*L;
                    fi = i*L;
                    if fi <= length(x)
                        xf = abs(fftshift(fft(x(st:fi)-mean(x(st:fi)),M))); xf = xf(end/2+1:end);
                        yf = abs(fftshift(fft(y(st:fi)-mean(y(st:fi)),M))); yf = yf(end/2+1:end);
                        data(:,cnt) = [xf ; (yf*mean(xf((end-Q+1):end))./mean(yf(1:Q)))];
                        cnt = cnt + 1;
                    end
                end
                disp(['完成 ' num2str(100*n/N) '%']);
            catch e
                warning(['处理文件时出错: ' e.message]);
            end
        end
        
        if ~isempty(data)
            Data = data.^2;
            % 保存
            save([save_filename bui_code '.mat'],'Data');
            disp(['已保存数据到: ' save_filename bui_code '.mat']);
        else
            warning(['没有数据可保存: ' bui_code]);
        end
    end
end

disp('数据聚合完成！'); 