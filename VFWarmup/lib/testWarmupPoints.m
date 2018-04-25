load('p/Projects/veryfastFVA/data/models/P_Putida/Pputida_model_glc_min.mat');%model putida
putida=model;
%%
%load warmup points
%3,000 (MATLAB)
warmupPts3000=createHRWarmup(putida,3000);
%5,000 (MATLAB)
warmupPts5000=createHRWarmup(putida,5000);
%2,120
warmup2120=csvread('P_Putida.mps2120warmup.csv');
%3,000
warmup3000=csvread('P_Putida.mps3000warmup.csv');
%5,000
warmup5000=csvread('P_Putida.mps5000warmup.csv');
%10,000
warmup10000=csvread('P_Putida.mps10000warmup.csv');
%100,000
warmup100000=csvread('P_Putida.mps100000warmup.csv');
%%
nSteps=1000;
nPts=10000;
nFiles=10;
%sampling 100,000 points 1000 steps
%cd 3000_warmup_Matlab
%ACHRSampler(putida, warmupPts3000, 'Putida3000M', nFiles, nPts, nSteps, [], [], 3600*24, 1);
%cd ..
%cd 5000_warmup_Matlab
%ACHRSampler(putida, warmupPts5000, 'Putida5000M', nFiles, nPts, nSteps, [], [], 3600*24, 1);
%cd ..
%
cd 2120_warmup_VF
ACHRSampler(putida, warmup2120, 'Putida2120', nFiles, nPts, nSteps, [], [], 3600*24, 1);
cd ..
%
cd 3000_warmup_VF
ACHRSampler(putida, warmup3000, 'Putida3000', nFiles, nPts, nSteps, [], [], 3600*24, 1);
cd ..
%
cd 5000_warmup_VF
ACHRSampler(putida, warmup5000, 'Putida5000', nFiles, nPts, nSteps, [], [], 3600*24, 1);
cd ..
%
cd 10000_warmup_VF
ACHRSampler(putida, warmup10000, 'Putida10000', nFiles, nPts, nSteps, [], [], 3600*24, 1);
cd ..
%
%%
%Check if they are solutions Sv=0

%%
%The sampling without FVA bounds
[minFlux, maxFlux] = fluxVariability(putida,0);
%%
figure;
subplot(2,2,1)
hist(warmup3000(1,:))
subplot(2,2,2)
hist(warmup5000(1,:))
subplot(2,2,3)
hist(warmup10000(1,:))
%%
samples2120  = loadSamples('Putida2120',10,10000);
samples3000  = loadSamples('Putida3000',10,10000);
samples5000  = loadSamples('Putida5000',10,10000);
samples10000 = loadSamples('Putida10000',10,10000);
%check CBT code fro sample processing
%correct bounds of samples
%Solve instead of sampling 
%%
figure;
x=300;
for i=x+1:x+64
    subplot(8,8,i-x)
    h1=histogram(samples2120(i,:));
    hold on;
    h2=histogram(samples10000(i,:));
    h1.Normalization = 'probability';
    bWidth=abs(h1.BinLimits(1)-h1.BinLimits(2))/10;
    h1.BinWidth = bWidth;
    h2.Normalization = 'probability';
    h2.BinWidth = bWidth;
    plot([minFlux(i) maxFlux(i)],[0 0],'*r');
    %title(putida.rxns{i});
end
