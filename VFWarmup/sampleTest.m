%cd('P:\Projects\cudaSampler')
modelName ='ecoli_core';
switch modelName
    case 'ecoli_core'
        load Ecoli_core_model;
        model=modelEcore;
        warmupPts = csvread('Ecoli_core.mpswarmup.csv');
    case 'ematrix'
        load('Thiele et al. - E-matrix_LB_medium.mat'); 
    case 'ecoli_k12'
        load('iAF1260.mat');
        model=modeliAF1260;
    case 'putida'   
        load('Pputida_model_glc_min.mat')
        warmupPts = csvread('P_Putida.mpswarmup.csv');
end
%addpath(genpath('C:\Program Files\IBM\ILOG\CPLEX_Studio1262'));
%changeCobraSolver('tomlab_cplex')
%tic;warmupPts = SimplifiedcreateHRWarmup(model,nWrmup);toc;
%csvwrite('ecoliwarmup501.csv',warmupPts);
%a=full(warmupPts)

%%
clc
tic;ACHRSamplerMOD(model,warmupPts,'trial',1,1,1);toc;





