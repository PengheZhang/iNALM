function AL = ALag(f, u, z, Aw, b, rho, lambda) 
% Computing augmented Lagrangian function
res = Aw + b - u;
nres2 = norm(res)^2;
L = f + lambda*nnz(u > 0) + z'*res;
AL = L + rho*nres2/2;             
end