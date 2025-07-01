clear all
close all
num_1 = audioread('E:\sound2\1.wav');
num_2 = audioread('E:\sound2\2.wav');
num_3 = audioread('E:\sound2\3.wav');
num_4 = audioread('E:\sound2\4.wav');
num_5 = audioread('E:\sound2\5.wav');
num_6 = audioread('E:\sound2\6.wav');
num_7 = audioread('E:\sound2\7.wav');
num_8 = audioread('E:\sound2\8.wav');
num_9 = audioread('E:\sound2\9.wav');

sound_1000hz = audioread('E:\sound2\1000hz.wav');
sound_1500hz = audioread('E:\sound2\1500hz-1.wav');
sound_1500hz_50ms = audioread('E:\sound2\1500hz.wav');

sound_hexian = audioread('E:\sound2\hexian.wav');
sound_hexian2= audioread('E:\sound2\hexian2.wav');

sound_seq = cell(5,1);
num_seq = cell(9,1);
num_seq{1,1}=num_1;
num_seq{2,1}=num_2;
num_seq{3,1}=num_3;
num_seq{4,1}=num_4;
num_seq{5,1}=num_5;
num_seq{6,1}=num_6;
num_seq{7,1}=num_7;
num_seq{8,1}=num_8;
num_seq{9,1}=num_9;

sound_seq{1,1} = sound_1000hz;
sound_seq{2,1} = sound_1500hz;
sound_seq{3,1} = sound_1500hz_50ms;
sound_seq{4,1} = sound_hexian;
sound_seq{5,1} = sound_hexian2;

