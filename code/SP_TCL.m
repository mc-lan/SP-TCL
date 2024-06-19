function [acc,acc_ite,W,Y_pred] = SP_TCL(X_src,Y_src,X_tar,Y_tar)
global options
SPLopts.Type = 'hard';
[~,ns] = size(X_src);
nt = size(X_tar,2);
Ys = bsxfun(@eq, Y_src(:), 1:max(Y_src));
Ys = Ys';
n = ns+nt;
X = [X_src X_tar];

knn_model = fitcknn(X_src',Y_src,'NumNeighbors',1);
Cls = knn_model.predict(X_tar');
acc = mean(Y_tar == Cls);
fprintf('1NN=%0.4f\n',acc);

C = size(Ys,1);
T = eye(C); %label encoding matrix

%% Construct graph Laplacian
if options.rho > 0
    Ls = zeros(ns,ns);
    manifold.k = options.p;
    manifold.Metric = 'Cosine';
    manifold.NeighborMode = 'KNN';
    manifold.WeightMode = 'Cosine';
    WWW = lapgraph(X_tar',manifold);
    Dw = diag(sparse(sqrt(1 ./ sum(WWW))));
    Lt = eye(nt) - Dw * WWW * Dw;
    L = blkdiag(Ls,Lt);
else
    L = 0;
end
%% initialization
Ps = Ys';
Pt = zeros(nt,C);
P = [Ps;Pt];
F = diag([ones(ns,1);ones(nt,1)])*(P.^options.r);
S = diag(sum(F,2));

if strcmp(options.kernel_type,'primal')
    W = (X * S  * X' + options.eta * eye(size(X,1))) \ X * F;
    Y_pred = W'*X;
else
    K = kernel_meda(options.kernel_type,X,sqrt(sum(sum(X .^ 2).^0.5)/(size(X,1) + size(X,2))));
    W = ( S  * K + options.eta * eye(size(K,1))) \  F;
    Y_pred = W'*K;
end

[~,Cls] = max(Y_pred',[],2);
acc = mean(Y_tar == Cls(ns+1:end));
fprintf('iter=0\t');
fprintf('My_DA=%0.4f\n',acc);
acc_ite = acc;
%acc_src_ite = mean(Y_src == Y_true);

iteration = 1;
%% self-paced transfer classifier learning
rankratio_src = [1:-0.25:0];
for iter_out = 1:5
    if iter_out == 1
        vv = ones(ns,1);
    else
        % Compute loss
        loss = sum(P.^options.r.*q,2);
        
        % Compute u
        if rankratio_src(iter_out)==0
            vv = zeros(ns,1);
        else
            [vv, ~] = SPLreweighting(loss(1:ns),rankratio_src(iter_out),SPLopts);
        end
    end
    for iter = 1:options.T
        % Compute P
        for j = 1:C
            v = Y_pred - repmat(T(:,j),1,n);
            q(:,j) = sum(v'.*v',2);
        end
        if options.r==1
            P = zeros(n,C);
            [~,idx] = min(q,[],2);
            for i = 1:n
                P(i,idx(i)) = 1;
            end
        else
            P = q.^(1/(1-options.r-eps))./repmat(sum(q.^(1/(1-options.r-eps)),2),1,C);
            P = min(P,1);   %aviod NaN
        end
        
        F = diag([vv;ones(nt,1)])*(P.^options.r);
        S = diag(sum(F,2));
        
        %Compute W
        if strcmp(options.kernel_type,'primal')
            W = (X * (S + options.rho * L) * X' + +options.eta * eye(size(X,1))) \ X * F;
            Y_pred = W'*X;
        else
            W = ((S + options.rho * L) * K  +options.eta * eye(size(K,1))) \ F;
            Y_pred = W'*K;
        end
        
        % Compute acc
        [~,Cls] = max(Y_pred',[],2);
        acc = mean(Y_tar == Cls(ns+1:end));
        %acc_src = mean(Y_true == Cls(1:ns));
        fprintf('iter=%d\t',iteration);
        fprintf('My_DA=%0.4f\n',acc);
        %fprintf('src: My_DA=%0.4f\n',acc_src);
        acc_ite = [acc_ite,acc];
        %acc_src_ite = [acc_src_ite,acc_src];
        iteration = iteration + 1;
    end
end
end