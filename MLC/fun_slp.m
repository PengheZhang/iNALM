function out = fun_slp(w,type,q,zeta)
% =========================================================================
% A function for caluculating f(w), \nabla f(w) and \nabla^2 f(w) of
% smooth l1 norm
% =========================================================================
if nargin < 3; q = 1; zeta = 1e-3; end
delta = 1e-2;
if type == "fun"
    out = w.^2 + zeta; out = out.^(q/2); out(end) = delta*out(end); out = sum(out);
elseif type == "grad"
    out = w.^2 + zeta; out = out.^(1-q/2); out = q*w./out; out(end) = delta*out(end);
else
    temp = w.^2; temp1 = q*(zeta - (1-q)*temp); temp2 = (zeta + temp).^(2-q/2);
    out = temp1./temp2; out(end) = delta*out(end);
end
end