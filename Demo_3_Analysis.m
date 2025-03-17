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
input_path = 'D:\code\dronerf\output\aggregated_data\';  % Main_1_Data_aggregation.m的输出路径
output_path = 'D:\code\dronerf\output\analysis\';        % 分析结果的保存路径

% 创建输出目录（如果不存在）
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

%% Parameters
opt = 2;                                      % 设置为2，使用四分类模式
BUI = {'00000','10000','10001','10010','10011',...
    '10100','10101','10110','10111','11000'}; % BUI of all RF data
M   = 2048;                                   % Total number of frequency bins
fs  = 80;                                     % Sampling frequency in MHz
f   = 2400+(0:fs/(M-1):fs);                   % Frequency array for plotting
S   = 10;                                     % Number of points in the moving average filter for smoothing 
c   = [0 0 1 ; 1 0 0 ; 0 1 0 ; 0 0 0.1724 ;...
    1 0.1034 0.7241 ; 1 0.8276 0 ; 0 0.3448 0 ;...
    0.5172 0.5172 1 ; 0.6207 0.3103 0.2759 ;...
    0 1 0.7586];                              % 10 distinct colours for plotting       

% 定义类别名称
class_names = {'Background', 'Bebop', 'AR', 'Phantom'};

%% Averaging spectra
s = zeros(length(BUI),M);
raw_data = cell(length(BUI),1); % 存储原始数据用于时域图

for i = 1:length(BUI)
    try
        x = load([input_path BUI{1,i} '.mat']);
        [M, N] = size(x.Data);
        s(i,:) = mean(x.Data,2);
        
        % 保存一部分原始数据用于时域图
        if N > 0
            % 只保存前1000个样本点的数据用于时域图
            sample_size = min(1000, N);
            raw_data{i} = x.Data(:, 1:sample_size);
        else
            raw_data{i} = zeros(M, 1);
        end
    catch e
        warning(['无法加载文件: ' input_path BUI{1,i} '.mat - ' e.message]);
        % 如果文件不存在，使用零填充
        s(i,:) = zeros(1,M);
        raw_data{i} = zeros(M, 1);
    end
end

%% Aggregating and smoothing RF spectra
if(opt == 1)
    sig             = zeros(2,M);
    sig_smooth      = zeros(2,M);
    sig(1,:)        = s(1,:);
    sig(2,:)        = mean(s(2:end,:));
    sig_smooth(1,:) = smooth(sig(1,:),S);
    sig_smooth(2,:) = smooth(sig(2,:),S);
    
    % 类别名称
    tt = {'Background', 'Drone'};
elseif(opt == 2)
    sig             = zeros(4,M);
    sig_smooth      = zeros(4,M);
    sig(1,:)        = s(1,:);
    sig(2,:)        = mean(s(2:5,:));
    sig(3,:)        = mean(s(6:9,:));
    sig(4,:)        = s(10,:);
    sig_smooth(1,:) = smooth(sig(1,:),S);
    sig_smooth(2,:) = smooth(sig(2,:),S);
    sig_smooth(3,:) = smooth(sig(3,:),S);
    sig_smooth(4,:) = smooth(sig(4,:),S);
    
    % 类别名称
    tt = class_names;
elseif(opt == 3)
    sig              = s;
    sig_smooth       = zeros(10,M);
    sig_smooth(1,:)  = smooth(sig(1,:),S);
    sig_smooth(2,:)  = smooth(sig(2,:),S);
    sig_smooth(3,:)  = smooth(sig(3,:),S);
    sig_smooth(4,:)  = smooth(sig(4,:),S);
    sig_smooth(5,:)  = smooth(sig(5,:),S);
    sig_smooth(6,:)  = smooth(sig(6,:),S);
    sig_smooth(7,:)  = smooth(sig(7,:),S);
    sig_smooth(8,:)  = smooth(sig(8,:),S);
    sig_smooth(9,:)  = smooth(sig(9,:),S);
    sig_smooth(10,:) = smooth(sig(10,:),S);
    
    % 类别名称
    tt = {'Background', 'Bebop 1', 'Bebop 2', 'Bebop 3', 'Bebop 4', ...
          'AR 1', 'AR 2', 'AR 3', 'AR 4', 'Phantom'};
end

%% 绘制时域图
figure('Color',[1,1,1],'position',[100, 60, 840, 600]);
a = [];

% 根据opt选择要显示的类别
if opt == 1
    % 二分类：背景和无人机
    time_data = cell(2,1);
    time_data{1} = raw_data{1}(:,1); % 背景
    
    % 合并所有无人机数据
    drone_data = [];
    for i = 2:length(raw_data)
        if ~isempty(raw_data{i}) && size(raw_data{i},2) > 0
            drone_data = [drone_data, raw_data{i}(:,1)];
            if size(drone_data,2) >= 1
                break;
            end
        end
    end
    
    if isempty(drone_data)
        time_data{2} = zeros(size(time_data{1}));
    else
        time_data{2} = drone_data(:,1);
    end
    
    plot_names = {'Background', 'Drone'};
    
elseif opt == 2
    % 四分类：背景、Bebop、AR、Phantom
    time_data = cell(4,1);
    time_data{1} = raw_data{1}(:,1); % 背景
    
    % Bebop
    bebop_data = [];
    for i = 2:5
        if ~isempty(raw_data{i}) && size(raw_data{i},2) > 0
            bebop_data = [bebop_data, raw_data{i}(:,1)];
            if size(bebop_data,2) >= 1
                break;
            end
        end
    end
    
    if isempty(bebop_data)
        time_data{2} = zeros(size(time_data{1}));
    else
        time_data{2} = bebop_data(:,1);
    end
    
    % AR
    ar_data = [];
    for i = 6:9
        if ~isempty(raw_data{i}) && size(raw_data{i},2) > 0
            ar_data = [ar_data, raw_data{i}(:,1)];
            if size(ar_data,2) >= 1
                break;
            end
        end
    end
    
    if isempty(ar_data)
        time_data{3} = zeros(size(time_data{1}));
    else
        time_data{3} = ar_data(:,1);
    end
    
    % Phantom
    if ~isempty(raw_data{10}) && size(raw_data{10},2) > 0
        time_data{4} = raw_data{10}(:,1);
    else
        time_data{4} = zeros(size(time_data{1}));
    end
    
    plot_names = class_names;
    
else
    % 详细分类：所有10个类别
    time_data = cell(10,1);
    for i = 1:10
        if ~isempty(raw_data{i}) && size(raw_data{i},2) > 0
            time_data{i} = raw_data{i}(:,1);
        else
            if i > 1
                time_data{i} = zeros(size(time_data{1}));
            else
                time_data{i} = zeros(M,1);
            end
        end
    end
    
    plot_names = {'Background', 'Bebop 1', 'Bebop 2', 'Bebop 3', 'Bebop 4', ...
                 'AR 1', 'AR 2', 'AR 3', 'AR 4', 'Phantom'};
end

% 绘制时域图
for i = 1:length(time_data)
    % 只显示前200个点以便观察
    display_length = min(200, length(time_data{i}));
    a(i) = plot(1:display_length, time_data{i}(1:display_length), 'Color', c(i,:), 'linewidth', 2); 
    hold on;
end

xlabel('样本点', 'fontsize', 18); 
grid on; grid minor;
ylabel('幅度', 'fontsize', 18);
title('时域信号对比', 'fontsize', 20);
legend(a, plot_names, 'orientation', 'horizontal', 'location', 'south');
set(gca, 'fontweight', 'bold', 'fontsize', 20, 'FontName', 'Times');
set(gcf, 'Units', 'inches'); 
screenposition = get(gcf, 'Position');
set(gcf, 'PaperPosition', [0 0 screenposition(3:4)], 'PaperSize', screenposition(3:4));

%% 绘制频谱图
figure('Color',[1,1,1],'position',[100, 60, 840, 600]);
a = [];
for i = 1:size(sig,1)
    a(i) = plot(f,20*log10(sig_smooth(i,:)./(max(sig_smooth(i,:)))),'Color',c(i,:),'linewidth',2); hold on;
end
xlabel('频率 (MHz)','fontsize',18); grid on; grid minor;
ylabel('功率 (dB)','fontsize',18);
ylim([-110 5]);
title('频谱对比', 'fontsize', 20);
legend(a, tt, 'orientation', 'horizontal', 'location', 'south');
set(gca,'fontweight','bold','fontsize',20,'FontName','Times');
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));

%% 绘制箱线图
figure('Color',[1,1,1],'position',[100, 60, 700, 570]);
h = boxplot(20*log10(sig_smooth./(max(sig_smooth,[],2)))', 'Labels', tt, 'Whisker', 200);
set(h,{'linew'},{2});
set(gca,'fontweight','bold','fontsize',18,'FontName','Times');
xlabel('RF信号类别','fontsize',18); grid on; grid minor;
ylabel('功率 (dB)','fontsize',20);
ylim([-100 5]);
title('箱线图对比', 'fontsize', 20);
set(gcf,'Units','inches'); screenposition = get(gcf,'Position');
set(gcf,'PaperPosition',[0 0 screenposition(3:4)],'PaperSize',screenposition(3:4));

%% 保存结果
% 自动保存结果到输出目录（PNG格式）
print(1, [output_path 'TimeDomain_' num2str(opt)], '-dpng', '-r300');
print(2, [output_path 'Spectrum_' num2str(opt)], '-dpng', '-r300');
print(3, [output_path 'Box_' num2str(opt)], '-dpng', '-r300');
disp(['分析结果已保存到: ' output_path]);
disp('保存的文件:');
disp(['  - TimeDomain_' num2str(opt) '.png (时域图)']);
disp(['  - Spectrum_' num2str(opt) '.png (频谱图)']);
disp(['  - Box_' num2str(opt) '.png (箱线图)']);