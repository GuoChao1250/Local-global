clc;clear;close all;
tic
%%
load('converted_STD.mat')
load('times_350.mat')
load chanlocs_64  % M1 M2  13 19
load chanlocs_62
chanlocs_64 = chanloc s;
Active_1000=Std_350;

fig_path = 'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\healthy_process\ERP\python_code\Matlab\Fig\STD\group';
t=times_350*1000;

choose_chan=6;
fig = figure;
set(fig, 'Color', 'w');
plot(t,squeeze(mean(Active_1000(:,1,choose_chan,:),1))*1e6,'r','LineWidth',2 )
hold on
plot(t,squeeze(mean(Active_1000(:,2,choose_chan,:),1))*1e6,'b','LineWidth',2 )
legend('active','passive','FontSize',15,'AutoUpdate', 'off')
hold on
title(['1000Hz standard stimuli'],'FontSize',20)
set(gca,'XTick',-100:50:350)
set(gca,'XTickLabel',{'-100', ' ','0',' ','100',' ','200',' ','300','350'})
set(gca,'YTick',-2:1:2)
set(gca,'YTickLabel',{'-2', '-1','0','1','2'})
xlabel('Time/ms','FontSize',15)
ylabel('Amplitude/μV','FontSize',15)
xlim([-100 350])
ylim([-4 4])
% 移除图形窗口的边框
box off;

Active_1000(:,:,[13 19],:) = [];
start_time = [20,116,160];
end_time = [100,144,224];
for i=1:length(start_time)
    fig11 = figure;
    ST = dsearchn(t',start_time(i));
    ET = dsearchn(t',end_time(i));
    set(fig11, 'Color', 'w');
    topoplot(  squeeze(mean(Active_1000(:,1,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
%     title([num2str(start_time(i)),'-' ,num2str(end_time(i)),'ms'],'FontSize',15)
    caxis([-2 2])
%             cb=colorbar;
%             set(cb,'tickdir','out')  % 朝外
%             set(cb,'YTick',-2:2:2); %色标值范围及显示间隔
%             set(cb,'YTickLabel',{'-2','0','2'},'FontSize',30) %具体刻度赋值
    
    print('-dpng','-r600',[fig_path filesep num2str(start_time(i)) '-' num2str(end_time(i)) '_active.png'])
end

start_time = [24,116,152];
end_time = [100,136,216];
for i=1:length(start_time)
    fig11 = figure;
    ST = dsearchn(t',start_time(i));
    ET = dsearchn(t',end_time(i));
    set(fig11, 'Color', 'w');
    topoplot(  squeeze(mean(Active_1000(:,2,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
%     title([num2str(start_time(i)),'-' ,num2str(end_time(i)),'ms'],'FontSize',15)
    caxis([-2 2])
%             cb=colorbar;
%             set(cb,'tickdir','out')  % 朝外
%             set(cb,'YTick',-2:2:2); %色标值范围及显示间隔
%             set(cb,'YTickLabel',{'-2','0','2'},'FontSize',30) %具体刻度赋值
    
    print('-dpng','-r600',[fig_path filesep num2str(start_time(i)) '-' num2str(end_time(i)) '_passive.png'])
end
close all
%%
clear ;
load converted_DEV_MMN
load times_350
load chanlocs_64  % M1 M2  13 19
load chanlocs_62
chanlocs_64 = chanlocs;

Active_1000=Dev_1-STD_1;

fig_path = 'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\healthy_process\ERP\python_code\Matlab\Fig\DEV\group';
t=times_350*1000;

Active_1000(:,:,[13 19],:) = [];
start_time = [64];
end_time = [240];
for i=1:length(start_time)
    fig11 = figure;
    ST = dsearchn(t',start_time(i));
    ET = dsearchn(t',end_time(i));
    set(fig11, 'Color', 'w');
    topoplot(  squeeze(mean(Active_1000(:,1,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
    caxis([-4 4])
    print('-dpng','-r600',[fig_path filesep '1500_' num2str(start_time(i)) '-' num2str(end_time(i)) '_active.png'])
end

start_time = [56];
end_time = [232];
for i=1:length(start_time)
    fig11 = figure;
    ST = dsearchn(t',start_time(i));
    ET = dsearchn(t',end_time(i));
    set(fig11, 'Color', 'w');
    topoplot(  squeeze(mean(Active_1000(:,2,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
    caxis([-4 4])
    print('-dpng','-r600',[fig_path filesep '1500_' num2str(start_time(i)) '-' num2str(end_time(i)) '_passive.png'])
end
close all

%%
clear ;
load converted_NOV_MMN
load times_350
load chanlocs_64  % M1 M2  13 19
load chanlocs_62
chanlocs_64 = chanlocs;

Active_1000=Dev_1-STD_1;

fig_path = 'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\healthy_process\ERP\python_code\Matlab\Fig\NOV\group';
t=times_350*1000;

Active_1000(:,:,[13 19],:) = [];
start_time = [60,204];
end_time = [180,348];
for i=1:length(start_time)
    fig11 = figure;
    ST = dsearchn(t',start_time(i));
    ET = dsearchn(t',end_time(i));
    set(fig11, 'Color', 'w');
    topoplot(  squeeze(mean(Active_1000(:,1,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
    caxis([-6 6])
    print('-dpng','-r600',[fig_path filesep 'num_' num2str(start_time(i)) '-' num2str(end_time(i)) '_active.png'])
end

start_time = [60,192];
end_time = [176,316];
for i=1:length(start_time)
    fig11 = figure;
    ST = dsearchn(t',start_time(i));
    ET = dsearchn(t',end_time(i));
    set(fig11, 'Color', 'w');
    topoplot(  squeeze(mean(Active_1000(:,2,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
    caxis([-6 6])
    print('-dpng','-r600',[fig_path filesep 'num_' num2str(start_time(i)) '-' num2str(end_time(i)) '_passive.png'])
end
close all

%%
clear ;
load converted_Global_effect
load times_800
load chanlocs_64  % M1 M2  13 19
load chanlocs_62
chanlocs_64 = chanlocs;
fig_path = 'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\healthy_process\ERP\demo\analysis_scripts\Matlab\Fig\Global\group';
t=times_800*1000;
DEV_data = {Dev_1,Dev_2};
STD_data = {STD_1,STD_2};
data_label = {'G1','G2'};
for isdata = 1:length(DEV_data)
    if isdata == 1
        Active_1000=DEV_data{isdata}-STD_data{isdata};
        Active_1000(:,:,[13 19],:) = [];
        start_time = [40,220,300];
        end_time = [192,300,400];
        for i=1:length(start_time)
            fig11 = figure;
            ST = dsearchn(t',start_time(i));
            ET = dsearchn(t',end_time(i));
            set(fig11, 'Color', 'w');
            topoplot(  squeeze(mean(Active_1000(:,1,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
            caxis([-4 4])
            print('-dpng','-r600',[fig_path filesep data_label{isdata} '_' num2str(start_time(i)) '-' num2str(end_time(i)) '_active.png'])
        end
        
        start_time = [96,200,300];
        end_time = [180,296,400];
        for i=1:length(start_time)
            fig11 = figure;
            ST = dsearchn(t',start_time(i));
            ET = dsearchn(t',end_time(i));
            set(fig11, 'Color', 'w');
            topoplot(  squeeze(mean(Active_1000(:,2,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
            caxis([-4 4])
            print('-dpng','-r600',[fig_path filesep data_label{isdata} '_' num2str(start_time(i)) '-' num2str(end_time(i)) '_passive.png'])
        end
        close all
    else
        Active_1000=DEV_data{isdata}-STD_data{isdata};
        Active_1000(:,:,[13 19],:) = [];
        start_time = [108,220,300];
        end_time = [188,300,400];
        for i=1:length(start_time)
            fig11 = figure;
            ST = dsearchn(t',start_time(i));
            ET = dsearchn(t',end_time(i));
            set(fig11, 'Color', 'w');
            topoplot(  squeeze(mean(Active_1000(:,1,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
            caxis([-4 4])
            print('-dpng','-r600',[fig_path filesep data_label{isdata} '_' num2str(start_time(i)) '-' num2str(end_time(i)) '_active.png'])
        end
        
        start_time = [120,204,308];
        end_time = [176,308,400];
        for i=1:length(start_time)
            fig11 = figure;
            ST = dsearchn(t',start_time(i));
            ET = dsearchn(t',end_time(i));
            set(fig11, 'Color', 'w');
            topoplot(  squeeze(mean(Active_1000(:,2,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
            caxis([-4 4])
            print('-dpng','-r600',[fig_path filesep data_label{isdata} '_' num2str(start_time(i)) '-' num2str(end_time(i)) '_passive.png'])
        end
        close all        
    end
end

for isdata = 1:length(DEV_data)
    if isdata == 1
        Active_1000=DEV_data{isdata}-STD_data{isdata};
        Active_1000(:,:,[13 19],:) = [];
        start_time = [272];
        end_time = [392];
        for i=1:length(start_time)
            fig11 = figure;
            ST = dsearchn(t',start_time(i));
            ET = dsearchn(t',end_time(i));
            set(fig11, 'Color', 'w');
            topoplot(  squeeze(mean(Active_1000(:,1,:,ST:ET), [1,4]))*1e6-squeeze(mean(Active_1000(:,2,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
            caxis([-4 4])
            print('-dpng','-r600',[fig_path filesep data_label{isdata} '_' num2str(start_time(i)) '-' num2str(end_time(i)) '_active-passive.png'])
        end
    else
        Active_1000=DEV_data{isdata}-STD_data{isdata};
        Active_1000(:,:,[13 19],:) = [];
        start_time = [272];
        end_time = [400];
        for i=1:length(start_time)
            fig11 = figure;
            ST = dsearchn(t',start_time(i));
            ET = dsearchn(t',end_time(i));
            set(fig11, 'Color', 'w');
            topoplot(  squeeze(mean(Active_1000(:,1,:,ST:ET), [1,4]))*1e6-squeeze(mean(Active_1000(:,2,:,ST:ET), [1,4]))*1e6 ,chanlocs_62,'electrodes','off')
            caxis([-4 4])
            print('-dpng','-r600',[fig_path filesep data_label{isdata} '_' num2str(start_time(i)) '-' num2str(end_time(i)) '_active-passive.png'])
        end  
    end
end
%%
clear ;
load converted_Global_effect
load times_800
load chanlocs_64  % M1 M2  13 19
chanlocs_64 = chanlocs;
standard_8ch = {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1','O2'};
standard_16ch = {'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2'};
standard_32ch = {'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'AF7', 'AF3', 'AF4', 'AF8'};
fig_path = 'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\healthy_process\ERP\demo\analysis_scripts\Matlab\Fig\topoplot';
% 8-16-32-64 label
all_labels = {chanlocs_64.labels};
[~, idx_8] = ismember(standard_8ch, all_labels);
chanlocs_8 = chanlocs_64(idx_8(idx_8 > 0));  
[~, idx_16] = ismember(standard_16ch, all_labels);
chanlocs_16 = chanlocs_64(idx_16(idx_16 > 0));
[~, idx_32] = ismember(standard_32ch, all_labels);
chanlocs_32 = chanlocs_64(idx_32(idx_32 > 0));

% size of channel points
marker_size = 20;  
% 8chan
fig11 = figure;
set(fig11, 'Color', 'w');
topoplot([], chanlocs_8, 'style', 'blank', 'electrodes', 'on', ...
    'emarker', { '.', [1.0 0.8 0.6], marker_size, 1}, ...
    'hcolor', [1.0 0.8 0.6], 'headrad', 0.6,'plotrad', 0.7 ...
    );
print('-dpng','-r600',[fig_path filesep 'chan_8.png'])
% 16chan
fig11 = figure;
set(fig11, 'Color', 'w');
topoplot([], chanlocs_16, 'style', 'blank', 'electrodes', 'on', ...
    'emarker', { '.', [1.0 0.6 0.3], marker_size, 1}, ...
    'hcolor', [1.0 0.6 0.3], 'headrad', 0.6,'plotrad', 0.7 ...
    );
print('-dpng','-r600',[fig_path filesep 'chan_16.png'])
% 32chan
fig11 = figure;
set(fig11, 'Color', 'w');
topoplot([], chanlocs_32, 'style', 'blank', 'electrodes', 'on', ...
    'emarker', { '.', [0.9 0.4 0.1], marker_size, 1}, ...
    'hcolor', [0.9 0.4 0.1], 'headrad', 0.6,'plotrad', 0.7 ...
    );
print('-dpng','-r600',[fig_path filesep 'chan_32.png'])
% 64chan
fig11 = figure;
set(fig11, 'Color', 'w');
topoplot([], chanlocs_64, 'style', 'blank', 'electrodes', 'on', ...
    'emarker', { '.', [0.7 0.2 0.0 ], marker_size, 1}, ...
    'hcolor', [0.7 0.2 0.0 ], 'headrad', 0.6,'plotrad', 0.7 ...
    );
print('-dpng','-r600',[fig_path filesep 'chan_64.png'])
%%
toc