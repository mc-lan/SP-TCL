clear;clear global;clc;
warning off;
addpath('code')
global options
% Set algorithm parameters
options.rho = 1;
options.p = 5;
options.eta = 0.01;
options.r = 1.2;
options.T = 4;
options.kernel_type = 'primal';

srcStr = {'Art_Art','Art_Art','Art_Art','Clipart_Clipart','Clipart_Clipart','Clipart_Clipart',...
    'Product_Product','Product_Product','Product_Product','RealWorld_RealWorld','RealWorld_RealWorld','RealWorld_RealWorld'};
tgtStr = {'Art_Clipart','Art_Product','Art_RealWorld','Clipart_Art','Clipart_Product','Clipart_RealWorld',...
    'Product_Art','Product_Clipart','Product_RealWorld','RealWorld_Art','RealWorld_Clipart','RealWorld_Product'};

ffid = fopen('result_office_home_partial.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, ' rho = %.3f r = %.3f p = %d\n eta = %.3f\n',options.rho,options.r,options.p,options.eta);
datapath = 'Datasets\Office-Home_resnet50\';

for iData = 1:12
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);

    load(fullfile(datapath,[src,'.mat']));
    X_src = normc(fts);
    Y_src = labels;
    load(fullfile(datapath,[tgt,'.mat']));
    index = [];
    for in = 1:25
        indexi = find(labels==in);
        index = [index;indexi];
    end
    X_tar = fts(:,index);
    X_tar = normc(X_tar);
    Y_tar = labels(index);
    
    fprintf('$$$$$$$$$$$$$$$ --%s-- $$$$$$$$$$$$$$\n' ,options.data);
    
    %%
    [acc,acc_ite,~] = SP_TCL(X_src,Y_src,X_tar,Y_tar);
    ACCi(iData)=acc;
    acc = 100*acc;
    fprintf('******************************\n%s :\naccuracy: %.2f\n\n',options.data,acc);
    fprintf(ffid,'******************************\n%s :\naccuracy: ',options.data);
    fprintf(ffid,'%.2f\n', acc);
end
fclose(ffid);
mean(ACCi)