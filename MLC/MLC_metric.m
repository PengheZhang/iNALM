function [hamming_loss,rank_loss,average_precision] = MLC_metric(Y,Y_pre,scoremat)
% =========================================================================
% This code is used for computing some metrics of multi-lable
% classification.
%
% Written by Penghe Zhang on 30/09/2022
%==========================================================================
[n,m] = size(Y);
num_eva = 0;

hamming_loss = nnz(Y - Y_pre)/m/n;


[~,ind_order] = sort(scoremat,2,'descend');
rank = zeros(n,m);
% coverage = 0;
% one_error = 0;
average_precision = 0;
rank_loss = 0;
for i = 1:n
    rank(i,ind_order(i,:)) = 1:m;
        
    log_pos = Y(i,:) > 0;
    log_neg = Y(i,:) <= 0;
    rank_pos = rank(i,log_pos);

    if nnz(log_pos) && nnz(log_neg)
        num_eva = num_eva + 1;     % If the nnz(log_pos) = 0 or nnz(log_neg) = 0, this sample can't be used for rank-based evaluation

        score_neg = scoremat(i,log_neg); score_pos = scoremat(i,log_pos);
        temp = score_neg' - score_pos;
        temp1 = temp >=0;
        rank_loss = rank_loss + nnz(temp1)/nnz(log_neg)/nnz(log_pos);

        temp2 = rank_pos' - rank_pos;
        temp2 = temp2 >= 0;
        temp2 = sum(temp2,2)./rank_pos';
        average_precision = average_precision + sum(temp2)/nnz(log_pos); 
    end
        
end


rank_loss = rank_loss/num_eva;
average_precision = average_precision/num_eva; 
    


end