clc;clear;close all
tic
%%
load converted_file_STD
load times_350
pathname = 'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\Re_data\healthy\Sub__01\test_1';
EEG = pop_loadset([pathname filesep  'Spatial_filtered_ICA_Format.set'] );
chanlocs_64 = EEG.chanlocs;

%%
DW_data = Std_1000-Std_350;
t = times_350*1000;
choose_chan = 6;
colors = {'r','b'};
fig = figure;
set(fig, 'Color', 'w');
for istype = 1:size(DW_data,2)
    data1 = squeeze(DW_data(:,istype,:,:));
    data1 = data1*1e6;    
    plot(t,mean(squeeze(data1(:,choose_chan,:)),1),colors{istype},'LineWidth',2 )
    hold on
end
title(['1000Hz standard stimuli'],'FontSize',20)
set(gca,'XTick',-100:50:350,'XTickLabel',{'-100', ' ','0',' ','100',' ','200',' ','300','350'})
set(gca,'YTick',-4:1:4,'YTickLabel',{'-4', ' ','-2',' ','0',' ','2',' ','4'})
xlabel('Time/ms','FontSize',15)
ylabel('Amplitude/μV','FontSize',15)
xlim([-100 350])
ylim([-4 4])
% 移除图形窗口的边框
box off;
%% 0-100-200-300-400-500-600-700-800ms topo
start_time = [0,100,200];
end_time = [100,200,300];
Cond_type = {'Active','Passive'};
for istype = 1:size(DW_data,2)
    data1 = squeeze(DW_data(:,istype,:,:));
    data1 = data1*1e6;
    for i=1:length(start_time)
        ST = dsearchn(t',start_time(i));
        ET = dsearchn(t',end_time(i));
        fig11 = figure;
        set(fig11, 'Color', 'w');
        topoplot(mean(mean(data1(:,:,ST:ET),3),1),chanlocs_64,'electrodes','off')
        % title(['FOC latency: ',num2str(floor(Low)),'-' ,num2str(floor(High)),'ms'],'FontSize',15)
        caxis([-2 2])
        cb=colorbar;
        set(cb,'tickdir','out')  % 朝外
        set(cb,'YTick',-2:2:2); %色标值范围及显示间隔
        set(cb,'YTickLabel',{'-2','0','2'},'FontSize',30) %具体刻度赋值
        saveas(fig11, [pwd filesep 'Fig\topoplot\STD\' Cond_type{istype} '_' num2str(start_time(i)) '-' num2str(end_time(i)) '.png']);
    end
end



%%
toc