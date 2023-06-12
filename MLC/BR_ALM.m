function out = BR_ALM(X,Y,AL_method,subproblem)

[n,m] = size(Y);
X = [X,ones(n,1)];
[~,p] = size(X);
W = zeros(p,m);

time0 = tic;
for k = 1:m
    y = Y(:,k);
    A = -y.*X;
    b = ones(n,1);
    para.m = m;
    outm = AL_method( A, b, subproblem, para );
    W(:,k) = outm.w;
end
time = toc(time0);

out.W = W;
out.time = time;
end