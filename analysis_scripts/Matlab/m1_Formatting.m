clc;clear;close all
tic
%%
pathname = 'E:\MyWork\Graduation\ASAP_ICA_EEG_ArtifactRejection_v1\Re_data\healthy' ;
listing = dir(pathname) ;
filename = {listing(3:end).name} ;
test_type = {'test_1','test_2'};
% Std_350ms-Std_1000ms: s1 s4
% Dev-std: s11 s22
% Nov-std: s1 s2
ana_mark = {'s1', 's4', 's11','s22','s3','s2'};
for isSub = 1:length(filename)
    for istype =1:length(test_type)
        [num2str(isSub)]
        sub_pathname = [pathname filesep filename{isSub} filesep test_type{istype} filesep];
%         findset = dir([sub_pathname '*.set']);
        EEG = pop_loadset([sub_pathname  'Spatial_filtered_ICA_1.set'] );
        Type = {EEG.event.type};
               
        %% 去除不需要的mark
        not_in_ana_mark = setdiff(Type, ana_mark);
        if ~isempty(not_in_ana_mark)
            indexes_to_delete = [];
            for ii = 1:length(not_in_ana_mark)
                % 找到当前不在 ana_mark 中的元素在 Type 中的索引
                indexes_to_delete = [indexes_to_delete, find(strcmp(Type, not_in_ana_mark{ii}))];
            end
            EEG = pop_editeventvals(EEG,'delete',indexes_to_delete);
        end
        % 保存
        EEG = pop_saveset( EEG, 'filename','Spatial_filtered_ICA_Format.set','filepath',sub_pathname);
        clear Type
        Type = {EEG.event.type};
        
        %% 统计mark数量
        for ismark = 1:length(ana_mark)
            mark2 = ana_mark{ismark};
            index = find(strcmp(Type, mark2));
            Sub_mark(isSub,istype,ismark) = length(index);
            
        end
        clear Type
    end
end

%%
toc