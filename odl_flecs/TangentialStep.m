function [x, iters] = TangentialStep(Hess, Jac, grad, cnstr, x, m, tol, radius)
% Computes a step in the (approximate) null space of Jac using Projected-CG
% (see Nocedal and Wright, pg 461, for example)
% 
% inputs:
%  Hess - the Hessian
%  Jac - the Jacobian
%  grad - the NEGATIVE gradient at x0
%  cnstr - the NEGATIVE constraint value at x0
%  x - initial guess; must satisfy constraint (inexactly)
%  m - maximum number of iterations
%  tol - tolerance target
%  radius - trust region radius
% 
% outputs:
%  x - primal step that attempts to minimize QP in null space
%  iters - number of matrix-vector products
%--------------------------------------------------------------------------

nVar = size(Jac,2);
nCeq = size(Jac,1);
n = nVar+nCeq;
proj_tol = 1e-2*tol;

% r is the initial gradient, and g is the gradient projected onto null
% space; d is the search direction
r = Hess*x - grad; % negative sign because grad = -gradient
iters = 1;
[g, iters_proj] = ProjectNull(Jac, r, m, proj_tol);
iters = iters + iters_proj;
norm0 = norm(g);
d = -g;
for i = 1:m
    Hd = Hess*d;
    iters = iters + 1;
    dtHd = d.'*Hd;
    if (dtHd <= 0.0) 
        % negative curvature in null space
        display('TangentialStep: negative curvature detected');
        xtHd = x.'*Hd;
        gradtd = -grad.'*d;
        xtx = x.'*x;
        xtd = x.'*d;
        dtd = d.'*d;        
        [alpha] = SolveTrust1D(dtHd, xtHd, gradtd, xtx, xtd, dtd, radius);
        x = x + alpha*d;
        return;
    end;
    alpha = r.'*g./dtHd;
    x = x + alpha*d;
    if (norm(x) > radius)
        % step length exceeded
        display('TangentialStep: trust radius exceeded');
        x = x - alpha*d;
        xtHd = x.'*Hd;
        gradtd = -grad.'*d;
        xtx = x.'*x;
        xtd = x.'*d;
        dtd = d.'*d;        
        [alpha] = SolveTrust1D(dtHd, xtHd, gradtd, xtx, xtd, dtd, radius);
        x = x + alpha*d;
        return;
    end;
    rp = r + alpha*Hd;
    [gp, iters_proj] = ProjectNull(Jac, rp, m, proj_tol);
    iters = iters + iters_proj;
    if (norm(gp) < tol*norm0)
        return;
    end;
    beta = rp.'*gp ./ (r.'*g);
    d = -gp + beta*d;
    g = gp;
    r = rp;
end; 

end
%==========================================================================
function [alpha] = SolveTrust1D(dtHd, xtHd, gradtd, xtx, xtd, dtd, radius)
% Solves a 1D trust-region subproblem
% 
% inputs:
%  dtHd - product d'*H*d, where H is Hessian and d is step
%  xtHd - product x'*H*d, where H is Hessian, d is step, x is current sol
%  gradtd - product of gradient with step
%  xtx - product of current solution with itself
%  xtd - product of current solution and step
%  dtd - product of step with itself
%  radius - trust radius
% 
% outputs:
%  alpha - step length along d
%--------------------------------------------------------------------------

% the minimum must be on the trust radius for cases considered in CG
radius2 = radius*radius;
Dis = sqrt(max(0.0, xtd.^2 - (xtx - radius2)*dtd));
alpha_p = (-xtd + Dis)./dtd;
alpha_m = (-xtd - Dis)./dtd;

J_p = 0.5*alpha_p*alpha_p*dtHd + alpha_p*(xtHd + gradtd);
J_m = 0.5*alpha_m*alpha_m*dtHd + alpha_m*(xtHd + gradtd);
if (J_p <= J_m)
    alpha = alpha_p;
else
    alpha = alpha_m;
end;
end



