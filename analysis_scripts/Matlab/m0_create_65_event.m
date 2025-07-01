



EEG = pop_editeventvals(EEG,'delete',31:2760);

Type = {EEG.event.type};
EEG = pop_editeventvals(EEG,'delete',[find(strcmp(Type,'S 12')),...
    find(strcmp(Type,'S 13'))]);

EEG = eeg_checkset( EEG );

%% 更新操作界面
eeglab redraw