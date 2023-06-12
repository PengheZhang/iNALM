% =========================================================================
% A linear conjugate gradient method for solving Ax = b 
% The code is written by Penghe Zhang on 30/09/2022
% =========================================================================
function [x,evec,solve_ok] = cgn(Matvec,b,tol,maxit,x0)
n = numel(b);

if ~exist('maxit','var'); maxit = numel(b); end
if ~exist('tol','var'); tol = 1e-6; end 
if ~exist('x0','var'); x0 = zeros(n,1); end

reltol = tol*max(1,norm(b)); 
solve_ok = 1; 
evec = zeros(maxit,1);

x = x0;
if (norm(x) > 1e-15)
    Ax = Matvec(x);   
    g = Ax - b;  
else
    g = -b; 
end
d = -g;
err = norm(g);
for k = 1:maxit
    evec(k) = err;
    if (err < reltol); break; end  
    if (k > 1000) 
        ratio = evec(k-9:k)./evec(k-10:k-1); 
        if (min(ratio) > 0.9995) && (max(ratio) < 1.0005) %%terminate when stagnation happen
           solve_ok = -1; 
           break;
        end
     end       

    Ad = Matvec(d);
    alpha = err^2/(d'*Ad);
    x = x + alpha*d;
    g = g + alpha*Ad;

    err_old = err;
    err = norm(g);
    beta = err^2/err_old^2;
    d = -g + beta*d;
end

end