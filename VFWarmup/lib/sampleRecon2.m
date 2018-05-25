addpath(genpath('/mnt/gaiagpfs/users/homedirs/mbenguebila/veryfastFVA'))
addpath(genpath('/mnt/gaiagpfs/users/homedirs/mbenguebila/ACHR.cu'))

modelName ='recon2';
switch modelName
    case 'ecoli_core'
        load Ecoli_core_model;
        model=modelEcore;
        warmupPts = csvread('Ecoli_core.mpswarmup.csv');
    case 'ematrix'
        load('Thiele et al. - E-matrix_LB_medium.mat'); 
        warmupPts = csvread('Ematrix.mpswarmup.csv');
    case 'ecoli_k12'
        load('iAF1260.mat');
        model=modeliAF1260;
    case 'putida'   
        load('Pputida_model_glc_min.mat')
        warmupPts = csvread('P_Putida.mpswarmup.csv');
    case 'recon2'
        load('Recon205_20150515Consistent.mat')
        warmupPts = csvread('Recon2.mpswarmup.csv');
        model=modelConsistent;
end
%addpath(genpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1262'));
%changeCobraSolver('tomlab_cplex')
%tic;warmupPts = SimplifiedcreateHRWarmup(model,nWrmup);toc;
%csvwrite('ecoliwarmup501.csv',warmupPts);
%a=full(warmupPts)

%%
clc
for nFiles = [10]
	disp(nFiles)
	tic;ACHRSamplerMOD(model,warmupPts,'trial',nFiles,1000,1000);toc;
end




