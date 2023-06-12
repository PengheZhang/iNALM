% =========================================================================
% This simulaltion aims to demonstrate the performance of iNALM on various 
% binary classification datasets.
%
% variables:
% n: The number of samples
% p: The number of features
% r: The noise ratio
%
% The code is written by Penghe Zhang on 30/09/2022
% =========================================================================

clear all
clc

%% fix p & r, change n
n = 2e3:2e3:1e4; 
p = 5e3;
r = 0;  % noise

res_n = zeros(numel(n),3); % 5 simulated datasets 

rounds = 5; 


for i = 1:numel(n)

    for k = 1:rounds
        [X_tr,y_tr,X_te,y_te] = randomData(n(i),p,r);
        n_tr = numel(y_tr);
        n_te = numel(y_te);
        X_tr1 = [X_tr, ones(n_tr,1)];
        X_te1 = [X_te, ones(n_te,1)];

        A_tr = -y_tr.*X_tr1; b_tr = ones(n_tr,1);
        subproblem = @GSN;
        out1 = iNALM( A_tr, b_tr, subproblem );
        w = out1.w;
        acc = comp_acc(X_te,y_te,w);
        res_n(i,:) = res_n(i,:) + [acc,out1.time,out1.nsv]; 

    end

end
res_n = res_n/rounds;

%% fix n & r, change p
n = 5e3; 
p = 2e3:2e3:1e4;
r = 0;  % noise

res_p = zeros(numel(p),3); 

rounds = 5;


for i = 1:numel(p)


    for k = 1:rounds
        [X_tr,y_tr,X_te,y_te] = randomData(n,p(i),r);
        n_tr = numel(y_tr);
        n_te = numel(y_te);
        X_tr1 = [X_tr, ones(n_tr,1)];
        X_te1 = [X_te, ones(n_te,1)];

        A_tr = -y_tr.*X_tr1; b_tr = ones(n_tr,1);
        subproblem = @GSN;
        out1 = iNALM( A_tr, b_tr, subproblem );
        w = out1.w;
        acc = comp_acc(X_te,y_te,w);
        res_p(i,:) = res_p(i,:) + [acc,out1.time,out1.nsv]; 

    end

end
res_p = res_p/rounds;


%% fix n & p, change r
n = 1e4; 
p = 1e2;
r = 0.02:0.02:0.1;  % noise

res_r = zeros(numel(r),3); % 5 simulated datasets with noise ration

rounds = 5; 


for i = 1:numel(r)

    for k = 1:rounds
        [X_tr,y_tr,X_te,y_te] = randomData(n,p,r(i));
        n_tr = numel(y_tr);
        n_te = numel(y_te);
        X_tr1 = [X_tr, ones(n_tr,1)];
        X_te1 = [X_te, ones(n_te,1)];

        A_tr = -y_tr.*X_tr1; b_tr = ones(n_tr,1);
        subproblem = @GSN;
        out1 = iNALM(A_tr, b_tr, subproblem);
        w = out1.w;
        acc = comp_acc(X_te,y_te,w);
        res_r(i,:) = res_r(i,:) + [acc,out1.time,out1.nsv]; 


    end
end

res_r = res_r/rounds;


%% a function for computing accuracy
function acc = comp_acc(X,y,w)
n = numel(y);
X1 = [X,ones(n,1)];
acc = 1 - nnz(sign(X1*w) - y)/n;
end