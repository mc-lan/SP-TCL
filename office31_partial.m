clear;clear global;clc;
warning off;
addpath('code')
global options
% Set algorithm parameters
options.rho = 1;
options.p = 5;
options.eta = 1; 
options.r = 1.2; 
options.T = 4;
options.kernel_type = 'primal';

srcStr = {'amazon_amazon','amazon_amazon','dslr_dslr','dslr_dslr','webcam_webcam','webcam_webcam'};
tgtStr = {'amazon_dslr','amazon_webcam','dslr_amazon','dslr_webcam','webcam_amazon','webcam_dslr'};

ffid = fopen('result_office31_partial.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, ' rho = %.3f r = %.3f p = %d\n eta = %.3f\n',options.rho,options.r,options.p,options.eta);
datapath = 'Datasets\Office31_resnet50\';
for iData = 1:6
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    load(fullfile(datapath,[src,'.mat']));
    X_src = normc(fts);
    Y_src = labels;
    load(fullfile(datapath,[tgt,'.mat']));
    IN = [1,2,6,11,12,13,16,17,18,23];
    index = [];
    for in = 1:10
        indexi = find(labels==IN(in));
        index = [index;indexi];
    end
    X_tar = fts(:,index);
    X_tar = normc(X_tar);
    Y_tar = labels(index);
    
    fprintf('$$$$$$$$$$$$$$$ --%s-- $$$$$$$$$$$$$$\n' ,options.data);
    
    %%
    [acc,acc_ite,~,~] = SP_TCL(X_src, Y_src, X_tar,Y_tar);
    ACCi(iData)=acc;
    acc = 100*acc;
    fprintf('******************************\n%s :\naccuracy: %.4f\n\n',options.data,acc);
    fprintf(ffid,'******************************\n%s :\naccuracy: ',options.data);
    fprintf(ffid,'%.2f\n', acc);
end
fclose(ffid);
mean(ACCi)