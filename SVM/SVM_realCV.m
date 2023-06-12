% =========================================================================
% For the real datasets without testing sets, the five-fold 
% cross validation is conducted to evaluate performance of iNALM.
% =========================================================================

clear all
clc

cd
path1 = pwd;

load(strcat(path1,"\dataset\","dexter_merge.mat"))

if ~exist('X','var')
    X = X_tr;
    Y = y_tr;
end

if find(Y == -1)
    y = Y;
else
    y = 3 - 2*Y;
end

fold = 5;
data_fold = gen_cv_dataset(X,y,fold);
res = zeros(numel(fold),3);

for i = 1:fold
    X_tr = data_fold{i}{1,1}; y_tr = data_fold{i}{1,2};
    X_te = data_fold{i}{2,1}; y_te = data_fold{i}{2,2};

    n_tr = numel(y_tr);
    n_te = numel(y_te);
    p = size(X_tr,2);
    
    X_tr1 = [X_tr, ones(n_tr,1)];
    X_te1 = [X_te, ones(n_te,1)];
    A_tr = -y_tr.*X_tr1; b_tr = ones(n_tr,1);
    A_te = -y_te.*X_te1;
    p = p + 1;
    para.testset = A_te;

    subproblem = @GSN;
    out1 = iNALM( A_tr, b_tr, subproblem, para );
    w = out1.w;
    if exist("y_te","var")
        acc = comp_acc(X_te,y_te,w);
    else
        acc = comp_acc(X_tr,y_tr,w);
    end
    res(i,:) = [acc,out1.time,out1.nsv];

end

%% a function for computing accuracy
function acc = comp_acc(X,y,w)
n = numel(y);
X1 = [X,ones(n,1)];
acc = 1 - nnz(sign(X1*w) - y)/n;
end