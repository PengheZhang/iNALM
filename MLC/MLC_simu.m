% =========================================================================
% This simulaltion aims to demonstrate the performance of iNALM on 
% multi-label classification data with different dimensions.
%
% variables:
% n: The number of samples
% p: The number of features
% m: The number of classes
%
% The code is written by Penghe Zhang on 30/09/2022
% =========================================================================

clear all
clc

eva_matr = ["hamming loss", "ranking loss","one-error","coverage","avergae precision","Time"];
%% fix p & m, change n
n =1e3:1e3:5e3; 
p =5e3;
m = 3;  

res_n = zeros(numel(n),numel(eva_matr)); % 5 simulated datasets with different dimension

rounds = 10; % repeat for several rounds to avoid the randomness


for i = 1:numel(n)

    for k = 1:rounds
        [X_tr,Y_tr,X_te,Y_te,~] = randomData1_MLC(n(i),p,m);
        [n_tr,~] = size(X_tr);
        [n_te,~] = size(X_te);
        X_tr1 = [X_tr,ones(n_tr,1)];
        X_te1 = [X_te,ones(n_te,1)];

        out1 = BR_ALM(X_tr,Y_tr,@iNALM_slp,@GSN_slp);
        W = out1.W;
        scoremat = X_te1*W; Y_pre = sign(scoremat);
        [hamming_loss,rank_loss,one_error,coverage,avergae_precision] = MLC_metric(Y_te,Y_pre,scoremat);
        res_n(i,:) = res_n(i,:) + [hamming_loss,rank_loss,one_error,coverage,avergae_precision,out1.time];
       
    end
end

res_n = res_n/rounds;

%% fix n & m, change p
n = 1e3; 
p = 6e3:1e3:1e4;
m = 3;  

res_p = zeros(numel(p),numel(eva_matr)); % 5 simulated datasets with different dimension

rounds = 10; % repeat for several rounds to avoid the randomness


for i = 1:numel(p)

    for k = 1:rounds
        [X_tr,Y_tr,X_te,Y_te,~] = randomData1_MLC(n,p(i),m);
        [n_tr,~] = size(X_tr);
        [n_te,~] = size(X_te);
        X_tr1 = [X_tr,ones(n_tr,1)];
        X_te1 = [X_te,ones(n_te,1)];

        out1 = BR_ALM(X_tr,Y_tr,@iNALM_slp,@GSN_slp);
        W = out1.W;
        scoremat = X_te1*W; Y_pre = sign(scoremat);
        [hamming_loss,rank_loss,one_error,coverage,avergae_precision] = MLC_metric(Y_te,Y_pre,scoremat);
        res_p(i,:) = res_p(i,:) + [hamming_loss,rank_loss,one_error,coverage,avergae_precision,out1.time];
    end
end

res_p = res_p/rounds;


%% fix n & p, change m
n =1e3; 
p =5e3;
m = 6:2:14;  

res_m = zeros(numel(m),numel(eva_matr)); % 5 simulated datasets with different dimension

rounds = 10; % repeat for several rounds to avoid the randomness


for i = 1:numel(m)

    for k = 1:rounds
        [X_tr,Y_tr,X_te,Y_te,~] = randomData1_MLC(n,p,m(i));
        [n_tr,~] = size(X_tr);
        [n_te,~] = size(X_te);
        X_tr1 = [X_tr,ones(n_tr,1)];
        X_te1 = [X_te,ones(n_te,1)];

        out1 = BR_ALM(X_tr,Y_tr,@iNALM_slp,@GSN_slp);
        W = out1.W;
        scoremat = X_te1*W; Y_pre = sign(scoremat);
        [hamming_loss,rank_loss,one_error,coverage,avergae_precision] = MLC_metric(Y_te,Y_pre,scoremat);
        res_m(i,:) = res_m(i,:) + [hamming_loss,rank_loss,one_error,coverage,avergae_precision,out1.time];
    end

end
res_m = res_m/rounds;

