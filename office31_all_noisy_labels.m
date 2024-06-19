clear;clear global;clc;
warning off;
addpath('code')
global options
% Set algorithm parameters
options.rho = 1;
options.p = 5;
options.eta = 1; 
options.r = 1.1; 
options.T = 4;
options.kernel_type = 'primal';

srcStr = {'amazon','amazon','dslr','dslr','webcam','webcam'};
tgtStr = {'dslr','webcam','amazon','webcam','amazon','dslr'};

ffid = fopen('result_office31_all_noisy.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, ' rho = %.3f r = %.3f p = %d\n eta = %.3f\n',options.rho,options.r,options.p,options.eta);
datapath = '\Datasets\Office31_resnet50\';
for i = 1:3
for iData = 1:6
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    load(fullfile(datapath,[src,'_04','.mat']));
    X_src = zscore(feas',1)';
    X_src = normc(X_src);
    Y_src = [labels_err_1' labels_err_2' labels_err_3'];
    load(fullfile(datapath,[tgt,'_04','.mat']));
    X_tar = zscore(feas',1)';
    X_tar = normc(X_tar);
    Y_tar = labels;
    
    fprintf('$$$$$$$$$$$$$$$ --%s-- $$$$$$$$$$$$$$\n' ,options.data);
    
    %%
    [acc,acc_ite,~,~] = SP_TCL(X_src,Y_src(:,i),X_tar,Y_tar);
    ACCi(iData,i)=acc;
    acc = 100*acc;
    fprintf('******************************\n%s :\naccuracy: %.4f\n\n',options.data,acc);
    fprintf(ffid,'******************************\n%s :\naccuracy: ',options.data);
    fprintf(ffid,'%.2f\n', acc);
end
end
fclose(ffid);
mean(ACCi', 1)