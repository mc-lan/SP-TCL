clear;clear global;clc;
warning off;
addpath('code')
global options
% Set algorithm parameters
options.rho = 1;
options.p = 5;
options.eta = 0.5;
options.r = 1.1;
options.T = 4;
options.kernel_type = 'primal';

srcStr = {'Art','Art','Art','Clipart','Clipart','Clipart','Product','Product','Product','RealWorld','RealWorld','RealWorld'};
tgtStr = {'Clipart','Product','RealWorld','Art','Product','RealWorld','Art','Clipart','RealWorld','Art','Clipart','Product'};

ffid = fopen('result_office_home_partial_noisy_labels.txt','at');
fprintf(ffid, '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n', datestr(now));
fprintf(ffid, ' rho = %.3f r = %.3f p = %d\n eta = %.3f\n',options.rho,options.r,options.p,options.eta);
datapath = 'Datasets\Office-Home_resnet50\';
for i = 1:3
    for iData = 1:12
        src = char(srcStr{iData});
        tgt = char(tgtStr{iData});
        options.data = strcat(src,'_vs_',tgt);

        load(fullfile(datapath,[src,'_04','.mat']));
        X_src = normc(feas);
        Y_src = [labels_err_1' labels_err_2' labels_err_3'];
        load(fullfile(datapath,[tgt,'_04','.mat']));
        index = [];
        for in = 1:25
            indexi = find(labels==in);
            index = [index;indexi];
        end
        X_tar = feas(:,index);
        X_tar = normc(X_tar);
        Y_tar = labels(index);

        fprintf('$$$$$$$$$$$$$$$ --%s-- $$$$$$$$$$$$$$\n' ,options.data);


        %%
        [acc,acc_ite,~,~] = SP_TCL(X_src,Y_src(:,i),X_tar,Y_tar);
        ACCi(iData,i) = acc;
        acc = 100*acc;
        fprintf('******************************\n%s :\naccuracy: %.2f\n\n',options.data,acc);
        fprintf(ffid,'******************************\n%s :\naccuracy: ',options.data);
        fprintf(ffid,'%.2f\n', acc);
    end
end
fclose(ffid);
mean(ACCi', 1)