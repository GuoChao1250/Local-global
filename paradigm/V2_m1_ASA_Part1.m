clc
clear
close all
tic
%%
prompt = {'被试序号','姓名[英文]','性别[1=男,2=女]','出生日期[19990309]'};
dlg_title = '被试信息 ';
num_lines = 1;
defaultanswer = {'','','',''};
subinfo = inputdlg(prompt,dlg_title,num_lines,defaultanswer);
subID = str2num([subinfo{1}]);
name= [subinfo{2}];
gender = str2num([subinfo{3}]);
birthyear = str2num([subinfo{4}]);
%mkdir([pwd filesep 'Behavior_data' filesep [subID,name,gender,birthyear]])
load sound_num.mat
load sound_tune.mat
Screen('Preference', 'SkipSyncTests', 1);
try
    global screenNumber w wRect a b
    screenNumber = max(Screen('Screens'));
    Screen('Resolution', screenNumber, [2560], [1600], [60]);
    fontname = 'Arial';
    TextSize = 48;
    flag = 0;
    [w, wRect]=Screen('OpenWindow',screenNumber, 200,[],32,2);
    [a,b]=WindowCenter(w);
    Screen('TextFont',w,fontname);
    Screen('TextSize',w,TextSize);
    %% pictures
    fixation_img = imread([pwd filesep 'Fig' filesep 'fixation.png']);
    ending_img = imread([pwd filesep 'Fig' filesep 'ending.jpg']);
    start_img = imread([pwd filesep 'Fig' filesep 'start.jpg']);
    first_img = imread([pwd filesep 'Fig' filesep 'first.jpg']);
    
    fixation = Screen('MakeTexture',w,fixation_img);
    start = Screen('MakeTexture',w,start_img);
    ending = Screen('MakeTexture',w,ending_img);
    first_part = Screen('MakeTexture',w,first_img);
    KbName('UnifyKeyNames');
    confirm =  KbName('space');
    escapekey = KbName('escape');
    enterket = KbName('Return');
    RestrictKeysForKbCheck([KbName('space'),KbName('ESCAPE'),KbName('Return')]);
    
    
    
    %% Setting the port
    try
        PsychPortAudio('close');
    end
    
    config_io ;
    global cogent;
    if( cogent.io.status ~= 0 )
        error('inp/outp installation failed');
    end
    address = hex2dec('03F8');
    
    %% setting the audioport
    InitializePsychSound(1);
    nrchannels = 1;
    freq = 48000;
    repetitions = 1;
    startCue = 0;
    waitForDeviceStart = 1;
    PsychPortAudio('Close')
    pahandle = PsychPortAudio('Open', [], 1, 1, freq, nrchannels);
    
    %%  准备进入实验
    
    HideCursor
    Screen('DrawTexture', w, start, [], []);
    Screen('Flip',w);
    KbWait;
    Screen('DrawTexture', w, first_part, [], []);
    Screen('Flip',w);
    WaitSecs(0.5)
    KbWait;
    Screen('DrawTexture', w, fixation, [], []);
    Screen('Flip',w);
    WaitSecs(1)
    %% 建立
    waitbox=[0.8 1.1 0.85 1 1.05 0.9 0.95 0.8 0.85 1];
    for nn = 1:3
        for index_globle = 1:10
            outp(address,0);
            start_time = GetSecs ;
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue, waitForDeviceStart);
            outp(address,3);
            WaitSecs(0.35)
            
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,3);
            WaitSecs(waitbox(index_globle)+0.2)
        end
    end
    for index_repeat = 1:15
        %% 数字1
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.8)
        %% 标准5
        waitbox=[1.3 1.1 1 1.2 0.9];
        for index_experiment = 1:5
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字2
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{2}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(1)
        %% 偏差4
        waitbox = [0.8 1.2 1.3 0.9];
        for index_experiment = 1:4
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字3
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{3}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(1.1)
        %% 标准4
        waitbox = [0.8 0.9 1.2 1];
        for index_experiment = 1:4
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字4
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{4}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,4);
        WaitSecs(0.8)
        %% 偏差3
        waitbox=[1.3 0.9 1.1];
        for index_experiment = 1:3
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字5
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{5}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(1)
        %% 标准3
        waitbox=[0.9 0.8 1.3];
        for index_experiment = 1:3
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字6
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{6}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(1)
        %% 偏差5
        waitbox=[0.8 1.1 0.9 1.2 0.9];
        for index_experiment = 1:5
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字7
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{7}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.8)
        %% 标准3
        waitbox=[0.9 1.2 1];
        for index_experiment = 1:3
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字8
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{8}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.9)
        %% 偏差4
        waitbox=[1 1.1 1.2 0.8];
        for index_experiment = 1:4
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字9
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{9}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(1)
        %% 标准5
        waitbox=[0.9 1.2 1.1 0.8 1.3];
        for index_experiment = 1:5
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字1
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.9)
        %% 偏差3
        waitbox=[0.8 1.1 1];
        for index_experiment = 1:3
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字2
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{2}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.8)
        %% 标准4
        waitbox=[1 1.2 0.9 1.1];
        for index_experiment = 1:4
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字3
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{3}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.9)
        %% 偏差5
        waitbox=[0.8 1 1.3 1.2 1.1];
        for index_experiment = 1:5
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字4
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{4}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.9)
        %% 标准4
        waitbox=[0.8 1.1 1.3 1.2];
        for index_experiment = 1:4
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字5
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{5}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.9)
        %% 偏差3
        waitbox=[1.3 0.8 1];
        for index_experiment = 1:3
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字6
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{6}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(1)
        %% 标准3
        waitbox=[1.2 0.8 0.9];
        for index_experiment = 1:3
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字7
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{7}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.8)
        %% 偏差5
        waitbox=[1.3 0.9 1.1 1 1.2];
        for index_experiment = 1:5
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字8
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{8}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.9)
        %% 标准5
        waitbox=[1.2 1.3 1 1.1];
        for index_experiment = 1:5
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,1);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,4);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
        %% 数字9
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,1);
        WaitSecs(0.35)
        outp(address,0);
        PsychPortAudio('FillBuffer', pahandle, num_seq{9}');
        PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
        outp(address,2);
        WaitSecs(0.9)
        %% 偏差4
        waitbox=[1.3 0.8 1 1.1];
        for index_experiment = 1:4
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{1}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,11);
            WaitSecs(0.35)
            outp(address,0);
            PsychPortAudio('FillBuffer', pahandle, sound_seq{2}');
            PsychPortAudio('Start', pahandle, repetitions, startCue,  waitForDeviceStart);
            outp(address,22);
            WaitSecs(waitbox(index_experiment)+0.2)
        end
    end
    WaitSecs(1)
    WaitSecs(0.005)
    outp(address,0);
    PsychPortAudio('Stop', pahandle, 1, 1);
    WaitSecs(1)
    
    Screen('Close',w);
    ShowCursor
catch
    Screen('Closeall')
    rethrow(lasterror)
end
%%
toc