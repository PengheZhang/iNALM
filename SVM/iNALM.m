function out = iNALM( A, b, subproblem, para )
% =========================================================================
% An inexact augmented Lagrangian method for solving the zero-one
% support vector machine 
%
%  min_{w \in R^p}  0.5*(\sum_{i=1}^{p-1} w_i^2 + \delta*w_p^2) + \lambda \|(Aw+b)_+ \|_0,
%
% where \delta > 0, \lambda > 0, A = -y.*X, and b = ones(n,1).
% X \in R ^{n \times p} is the sample matrix
% y \in R^n is the label
% =========================================================================
% Inputs: 
%         A:            data matrix
%         b:            bias vector with all elements being 1
%         subproblem:   algorithm for solving inner subploblem of iNALM 
%         para:         Parameters are all OPTIONAL
%         
%         para.maxk     --  The maximum iterations     (default 1000)
%         para.tol      --  Tolerance of stopping (default 1e-3)
%         para.lambda   --  Penalty parameter (default 1)
%         para.gamma    --  Algorithm parameter (default 1e-1*sqrt(a))
%         para.mu       --  Lyapunov parameter (default 1e-2)
%         para.tau      --  Sufficient descent parameter (default 1e-2*mu)
%         para.rho      --  Lagrangian parameter (defualt 1)
%         para.delta    --  A weighted parameter of function f 
% =========================================================================
% Outputs:
%     out.w, out.u:      The solution 
%     out.z:             The multiplier
%     out.time           CPU time
%     out.nsv:           Number of support vectors
% =========================================================================
% Variables:
%     wn:   The new iterate w^{k+1}
%      w:   The current iterate w
%     w0:   The old iterate w^{k-1}
% =========================================================================
% Written by Penghe Zhang on 28/09/2022 based on the algorithm proposed in
%     Penghe Zhang, Naihua Xiu, Houduo Qi, 
%     iNALM: An inexact Newton Augmented Lagrangian Method for 
%     Zero-One Composite Optimization
% Send your comments and suggestions to <<< phzhang971001@gmail.com >>>                                  
% =========================================================================

[n,p] = size(A);
w = zeros(p,1);
u = zeros(n,1);
z = zeros(n,1);

if nargin < 4;              para    = [];                                                                                        end
if isfield(para,'maxk');    maxk    = para.maxk;    else;  maxk             = 1e3;                                               end
if isfield(para,'tol');     tol     = para.tol;     else;  tol              = 1e-3;                                              end
if isfield(para,'gamma');   gamma   = para.gamma;   else;  a = A.*A;      a = sum(a,2); a = min(a); gamma = 1e-1*sqrt(a);        end
if isfield(para,'mu');      mu      = para.mu;      else;  mu               = 1e-2;                                              end
if isfield(para,'tau');     tau     = para.tau;     else;  tau              = mu*1e-2;                                           end                                     
if isfield(para,'rho');     rho     = para.rho;     else;  rho              = 1e-0;                                              end
if isfield(para,'lambda');  lambda  = para.lambda;  else;  lambda           = max(1e-0,rho/2);                                   end 
if isfield(para,'delta');   delta   = para.delta;  elseif  min(p,n) >= 120 && min(p,n) <= 5000;  delta  = 1; else; delta = 1e-2; end 

Fnorm2  = @(var)norm(var)^2;

COD1 = p<= 50;
COD2 = n >= 1e3 && p>=1e2 && p<= 1e4;
COD3 = ~issparse(A);
if COD1 || (COD2 && COD3) 
    w = init(w, u, z, A, b, rho, mu, delta);
end
w0 = w;

FOC = zeros(maxk,1);
acc = zeros(maxk,1);
nSV = zeros(maxk,1);
Lyap = zeros(maxk,1);
timevec = Lyap;

c1 = 0.1;
c4 = 4*(mu + c1)^2/gamma^2;
theta_r = 1.5;

fprintf('Start to run iNALM... \n')
fprintf('Iter     Lyapunov        rho        numT \n')
fprintf('-------------------------------------------------- \n')

time0 = tic;
for k = 0:maxk
    timek = tic;
    for kl = 1:10 
        [wn, un, zn, Awn, Tn] = subproblem( w, z, A, b, rho, lambda, mu, c1, delta );
        
        fn = fun_wl2(wn,"fun",delta);
        ALn = ALag( fn, un, zn, Awn, b, rho, lambda );
        beta = c4/rho;
        Vn = ALn + beta*Fnorm2(wn - w)/2;

        if k >= 1 %line search
            f = fun_wl2(w,"fun",delta);
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
    
    numT = nnz(Tn);
    fprintf('%4d    %7.4e    %5.2e    %4d\n', k+1, Vn, rho, numT)
    
    Lyap(k+1) = Vn;
    nSV(k+1) = numT;
    if k == 0
        timevec(k+1) = toc(timek);
    else
        timevec(k+1) = timevec(k) + toc(timek);
    end
    acc(k+1) = nnz(Awn < 0)/n;

    %first-order optimality condition
    FOC1 = fun_wl2(wn,"grad",delta) + (zn'*A)';
    FOC1 = norm(FOC1);
    rn = Awn + b + z/rho;
    Tn1 = find( rn < 0 | rn > sqrt(2*lambda/rho) );
    Tn2 = find( rn == 0 | rn == sqrt(2*lambda/rho) );
    temp = min(abs(un(Tn2)),abs(zn(Tn2))/rho);
    FOC2 = Fnorm2(un(Tn)) + Fnorm2(temp) + Fnorm2(zn(Tn1)/rho); 
    FOC2 = sqrt(FOC2);
    FOC3 = norm(Awn + b - un);
    FOC(k+1) = max([FOC1,FOC2,FOC3]);

    
    %stopping criteria for outer algorithm
    SC = (norm(wn - w) + norm(un - u) + norm(zn - z))/( norm(wn) + norm(un) + norm( zn ) + 1);
    dim = n < 200 || n > 1e3 || p > 1e4;
    SC1 = SC < tol;
    SC2 = dim && FOC(k+1) < tol;
    SC3 = dim && k >= 1 && acc(k+1) - acc(k) <= - 1e-4;
    SC4 = dim && k >= 4 && nnz(acc(k-2:k+1) - acc(k-3:k)) == 0;
    SC5 = dim && k >= 1 && abs(nSV(k+1) - nSV(k)) < 10 && acc(k+1) == acc(k);  

    if SC1 || SC2 || SC3 || SC4 || SC5
        break;
    end
    
    w0 = w;   
    w = wn;
    Aw = Awn;
    u = un;
    z = zn;

end
time = toc(time0);
fprintf('-------------------------------------------------- \n')

out.w = wn;
out.u = un;
out.z = zn;
out.time = time;
out.nsv = numT;

end

%% Initialization
function out = init(w, u, z, A, b, rho, mu, delta)
r = A*w + b - u + z/rho;
ArT = (r'*A)';
d = fun_wl2(w,"grad",delta) + rho*ArT;
Hsf = fun_wl2(w,"hessian",delta) + mu;
temp1 = Hsf.*d; temp1 = d'*temp1;
temp2 = A*d; temp2 = rho*norm(temp2)^2;
t = norm(d)^2/(temp1 + temp2);
w = w - t*d;

out = w;
end


