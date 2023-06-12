function data_fold = gen_cv_dataset(X,y,fold)
% =========================================================================
% This code is used for dividing datasets (X,y) into several parts for
% cross validation.
% 
% Written by Penghe Zhang 30/09/2022.
% =========================================================================

rng('default')
[n,p] = size(X);
temp0 = randperm(n);
X = X(temp0,:); y = y(temp0);
n_ef = floor(n/fold); 
ind_pos = find(y>0);
ind_neg = find(y<0);
prop_pos = numel(ind_pos)/(numel(ind_pos) + numel(ind_neg));

data_fold = cell(fold,1);

% n_ef_pos = floor(n_ef/2);
% n_ef_neg = n_ef - n_ef_pos;
n_ef_pos = floor(n_ef*prop_pos);
n_ef_neg = n_ef - n_ef_pos;
for i = 1:fold
    data_fold{i} = cell(2,2);
    if i < fold
        temp1 = ind_pos((i-1)*n_ef_pos + 1:i*n_ef_pos);
        temp2 = ind_neg((i-1)*n_ef_neg + 1:i*n_ef_neg);
        temp = union(temp1,temp2);
        temp3 = setdiff(1:n,temp);
    
        data_fold{i}{1,1} = X(temp3,:); data_fold{i}{1,2} = y(temp3); 
        data_fold{i}{2,1} = X(temp,:); data_fold{i}{2,2} = y(temp);
    else
        temp1 = ind_pos((i-1)*n_ef_pos + 1:end);
        temp2 = ind_neg((i-1)*n_ef_neg + 1:end);
        temp = union(temp1,temp2);
        temp3 = setdiff(1:n,temp);
    
        data_fold{i}{1,1} = X(temp3,:); data_fold{i}{1,2} = y(temp3); 
        data_fold{i}{2,1} = X(temp,:); data_fold{i}{2,2} = y(temp);
    end
end

end