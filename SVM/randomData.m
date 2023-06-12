function [X_tr,y_tr,X_te,y_te] = randomData(n,p,r)
% =========================================================================
% This file aims at generating random data  
% Inputs:
%       n     -- number of samples
%       r     -- flipping ratio
% Outputs:
%       X_tr     --  training samples data,     R^{n/2 \times p}
%       y_tr     --  training samples classes,  {-1,1}^{n/2} 
%       X_te     --  testing  samples data,     R^{n/2 \times p}
%       y_te     --  testing  samples classes,  {-1,1}^{n/2}
% =========================================================================

n_tr = ceil(n/2);

mu = randn(p, 1);
mu1 = mu;
sigma1 = abs(randn(p,1));
sigma1 = diag(sigma1);
mu2 = randn( p,1 );
sigma2 = abs(randn(p,1));
sigma2 = diag(sigma2);


A   = [mvnrnd(mu1, sigma1, n_tr);
       mvnrnd(mu2, sigma2, n - n_tr)];
c   = [-ones(n_tr,1); ones(n - n_tr,1)];    
T   = randperm(n); 
X_tr   = A(T(1:n_tr),:); 
y_tr   = c(T(1:n_tr)); 
y_tr   = filp(y_tr,r);  

X_te = A(T(n_tr+1:n),:); 
y_te = c(T(1+n_tr:n));
y_te = filp(y_te,r);
clear A c T q
end

function fc = filp(fc,r)
      if r  > 0
         mc = length(fc) ;    
         T0 = randperm(mc);  
         fc(T0(1:ceil(r*mc)))=-fc(T0(1:ceil(r*mc)));
      end
end