function [X_tr,Y_tr,X_te,Y_te,W] = randomData1_MLC(n,p,m)
% =========================================================================
% This code first generates a multi-lable classification dataset with n
% samples, p features and m classes.
%
% Then 90% samples are selected as training set, and the rest of samples
% are regarded as testing set.
%
% The code is written by Penghe Zhang 30/09/2022
% =========================================================================

prop = 0.9;
n_tr = ceil(prop*n);
n1 = 2*n;
% rng(1,"twister")
for i = 1:10
    X = randn(n1,p);
    X1 = [X,ones(n1,1)];
    W = 2*rand(p + 1,m) - 1;
    X1W = X1*W;
    Y = sign(X1W);

    ind = find( sum(Y,2) > -m );
    if numel(ind) >=n
        X = X(ind,:);
        Y = Y(ind,:);
        break
    end
end


T = randperm(n);
X = X(T,:);
Y = Y(T,:);
X_tr = X(1:n_tr,:);
Y_tr = Y(1:n_tr,:);
X_te = X(n_tr+1:end,:);
Y_te = Y(n_tr+1:end,:);

end