function out = iNALM_slp( A, b, subproblem, para )
% =========================================================================
% An inexact augmented Lagrangian method for solving a multi-label 
% classification problem with smooth l1 regularizer and 0/1 loss
% =========================================================================
% Inputs: 
%         A:            data matrix
%         b:            bias vector with all elements being 1
%         subproblem:   algorithm for solving inner subploblem of iNALM 
%         para:         Parameters are all OPTIONAL
%         
%         para.m        --  The number of classes
%         para.maxk     --  The maximum iterations     (default 1000)
%         para.tol      --  Tolerance of stopping (default 1e-3)
%         para.lambda   --  Penalty parameter (default 1)
%         para.gamma    --  Algorithm parameter (default 1e-1*sqrt(a))
%         para.mu       --  Lyapunov parameter (depending on dimension)
%         para.tau      --  Sufficient descent parameter (default 1e-2*mu)
%         para.rho      --  Lagrangian parameter (depending on dimension)
% =========================================================================
% Outputs:
%     out.w, out.u:      The solution 
%     out.z:             The multiplier
%     out.time:          CPU time
%     out.iter:          The number of iteration
% =========================================================================
% Variables:
%      n:   The number of samples
%      p:   The number of features
%      m:   The number of classes
%     wn:   The new iterate w^{k+1}
%      w:   The current iterate w
%     w0:   The old iterate w^{k-1}
% =========================================================================
% Written by Penghe Zhang on 28/09/2022 based on the algorithm proposed in
%     Penghe Zhang, Naihua Xiu, Houduo Qi, 
%     iNALM: An inexact Newton Augmented Lagrangian Method for Zero-One 
%     Composite Optimization.
% Send your comments and suggestions to <<< phzhang971001@gmail.com >>>                                  
% =========================================================================
[n,p] = size(A);

w = zeros(p,1);
u = zeros(n,1);
z = zeros(n,1);
w0 = w;

if nargin < 4;              para    = [];           end
if isfield(para,'m');       m       = para.m;       end 
if isfield(para,'maxk');    maxk    = para.maxk;    else;  maxk = 1e3;     end
if isfield(para,'tol');     tol     = para.tol;     else;  tol  = 1e-3;    end
if isfield(para,'gamma');   gamma   = para.gamma;   else;  a    = A.*A; a = sum(a,2); a = min(a); gamma = 1e-1*sqrt(a);  end
if isfield(para,'mu');      mu      = para.mu;      elseif (n <= 500 && p >= 1000) || m < 4;  mu = 1e-2; elseif n>=7000 && p >= 20000; mu = 1e3; else; mu = 1e2;  end
if isfield(para,'tau');     tau     = para.tau;     else;  tau  = 1e-2*mu; end
if isfield(para,'rho');     rho     = para.rho;     elseif m<4;  rho    = 1e0; else; rho    = 1e2;  end
if isfield(para,'lambda');  lambda  = para.lambda;  elseif m<4;  lambda = 1e0; else; lambda = 1e3;  end



Fnorm2  = @(var)norm(var)^2;

acc = zeros(maxk,1);
timevec = zeros(maxk,1);

c1 = 0.1;
c4 = 4*(mu + c1)^2/gamma^2;
theta_r = 1.5;

accmax = 0;

time0 = tic;
for k = 0:maxk
    timek = tic;

    for kl = 1:10 
        [wn, un, zn, Awn] = subproblem( w, u, z, A, b, rho, lambda, mu, c1);
        
        fn = fun_slp(wn,"fun");
        ALn = ALag( fn, un, zn, Awn, b, rho, lambda );
        beta = min(c4/rho,mu/4);
        Vn = ALn + beta*Fnorm2(wn - w)/2;

        if k >= 1 %line search
            f = fun_slp(w,"fun");
            AL = ALag( f, u, z, Aw, b, rho, lambda );       
            V = AL + beta*Fnorm2(w - w0)/2;

            if V - Vn >= tau*Fnorm2(wn - w)
                break;
            else
                rho = rho*theta_r;
            end

        else
            break;
        end
    end


    if k == 0
        timevec(k+1) = toc(timek);
    else
        timevec(k+1) = timevec(k) + toc(timek);
    end
    acc(k+1) = nnz(Awn < 0)/n;

    if acc(k+1) >= accmax
        accmax = acc(k+1);
        itermax = k+1;
        wmax = wn;
        umax = un;
        zmax = zn;  
    end

    %stopping criteria
    SC1 = (norm(wn - w) + norm(un - u) + norm(zn - z))/( norm(wn) + norm(un) + norm( zn ) + 1) < tol;
    SC2 = acc(k+1) > 0.95|| acc(k+1) < accmax - 1e-3;
    SC3 = k >= 2 && nnz(acc(k:k+1) - acc(k-1:k)) == 0;
    if SC1 || SC2 || SC3
        break;
    end
   
    w0 = w;   
    w = wn;
    Aw = Awn;
    u = un;
    z = zn;

end

time = toc(time0);

out.w = wmax;
out.u = umax;
out.z = zmax;
out.iter = itermax;
out.time = time;
timevec = timevec(1:k+1);
out.timevec = timevec;


end