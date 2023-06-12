function out = fun_wl2(w, type, delta) 
% =========================================================================
% A function for caluculating f(w), \nabla f(w) and \nabla^2 f(w) of
% weighted square l2 norm
% f(w) = 0.5*(\sum_{i=1}^{p-1} w_i^2 + \delta*w_p^2)
% =========================================================================

p = numel(w);


switch type
    case "cons"
        out = delta;
    case "fun"
        out = w; out(p) = delta*w(p); out = w'*out/2;
    case "grad"
        out = w; out(p) = delta*w(p); 
    otherwise
        out = ones(p,1); out(p) = delta;
end

end