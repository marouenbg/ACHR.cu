%Script for benchmarking the creation of 10000 warmup points in metabolic
%models 
modelList={'Ecoli_core','Ecoli_K12','Ematrix','Ematrix_coupled','P_Putida','Recon2'};
nPoints=30000;
runTimes=zeros(length(modelList),1);
iter=0;
for modelName = modelList
    iter=iter+1;
    switch modelName{1}
        case 'Ecoli_core'
            load ecoli_core_model.mat;%model ecoli_core
        case 'Ecoli_K12'
            load('P:\Projects\veryfastFVA\data\models\Ecoli_K12\iAF1260.mat');%modeliAF1260 k12
            model=modeliAF1260;
        case 'Ematrix'
            load('P:\Projects\veryfastFVA\data\models\Ematrix\Thiele et al. - E-matrix_LB_medium.mat');%model Ematrix
        case 'Ematrix_coupled'
            load('P:\Projects\veryfastFVA\data\models\Ematrix_coupled\EMatrix_LPProblemtRNACoupled90.mat');%LPProblemRNACoupled90 Ematrix coupled
            model=LPProblemRNACoupled90;
        case 'P_Putida'
            load('P:\Projects\veryfastFVA\data\models\P_Putida\Pputida_model_glc_min.mat');%model putida
        case 'Recon2'
            load('P:\Projects\veryfastFVA\data\models\Recon2\Recon205_20150515Consistent.mat');%modelConsistent Recon2
            model=modelConsistent;
    end
    tic
    [warmupPts] = createHRWarmup(model, nPoints);
    runTime=toc
    runTimes(iter)=runTime;
end