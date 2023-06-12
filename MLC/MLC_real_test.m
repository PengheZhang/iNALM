clear all
clc

cd
path1 = pwd;
load(strcat(path1,"\dataset\","Slashdot.mat"))

if exist("data","var")
    X_tr = data;
    Y_tr = target';
elseif exist("train_data","var")
    X_tr = train_data;
    Y_tr = train_target';
end
[n_tr, m] = size(Y_tr);

X_tr1 = [X_tr, ones(size(X_tr,1),1)];

if ~exist("X_te","var")
    X_te = X_tr;
    X_te1 = X_tr1;
    Y_te = Y_tr;
else
    X_te1 = [X_te, ones(size(X_te,1),1)];
end

out1 = BR_ALM(X_tr,Y_tr,@iNALM_slp,@GSN_slp);
W = out1.W;
scoremat = X_te1*W; 
Y_pre = sign(scoremat);
[hamming_loss,rank_loss,avergae_precision] = MLC_metric(Y_te,Y_pre,scoremat);
res = [hamming_loss,rank_loss,avergae_precision,out1.time];
