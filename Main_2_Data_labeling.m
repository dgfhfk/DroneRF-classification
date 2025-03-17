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
load_filename = 'D:\code\dronerf\output\aggregated_data\';      % 聚合数据的路径（文件1的输出）
save_filename = 'D:\code\dronerf\output\labeled_data\';         % 标记数据的保存路径

% 创建输出目录（如果不存在）
if ~exist(save_filename, 'dir')
    mkdir(save_filename);
end

%% 参数设置
% 定义数据类别和对应的BUI代码
BUI{1,1} = {'00000'};                         % BUI of RF background activities
BUI{1,2} = {'10000','10001','10010','10011'}; % BUI of the Bebop drone RF activities
BUI{1,3} = {'10100','10101','10110','10111'}; % BUI of the AR drone RF activities
BUI{1,4} = {'11000'};                         % BUI of the Phantom drone RF activities

%% 加载和连接RF数据
DATA = [];
LN   = [];

for t = 1:length(BUI)
    for b = 1:length(BUI{1,t})
        file_path = [load_filename BUI{1,t}{b} '.mat'];
        
        % 检查文件是否存在
        if ~exist(file_path, 'file')
            warning(['文件不存在: ' file_path]);
            continue;
        end
        
        disp(['正在加载数据: ' file_path]);
        
        % 加载数据
        try
            load(file_path);
            
            % 归一化数据
            Data = Data./max(max(Data));
            
            % 连接数据
            DATA = [DATA, Data];
            LN   = [LN size(Data,2)];
            
            clear Data;
            disp(['完成 ' num2str(100*t/length(BUI)) '%']);
        catch e
            warning(['加载文件时出错 ' file_path ': ' e.message]);
        end
    end
end

%% 标记
if isempty(DATA)
    error('没有加载数据。无法继续标记。');
end

% 创建标签
Label = zeros(3,sum(LN));

% 标签1：二分类（0=背景，1=无人机）
Label(1,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:end)))];

% 标签2：四分类（0=背景，1=Bebop，2=AR，3=Phantom）
Label(2,:) = [0*ones(1,LN(1)) 1*ones(1,sum(LN(2:5))) 2*ones(1,sum(LN(6:9))) 3*ones(1,LN(10))];

% 标签3：详细分类（每个具体数据集一个标签）
temp = [];
for i = 1:length(LN)
    temp = [temp (i-1)*ones(1,LN(i))];
end
Label(3,:) = temp;

%% 保存
csvwrite([save_filename 'RF_Data.csv'],[DATA; Label]);
disp(['数据已成功标记并保存到: ' save_filename 'RF_Data.csv']); 