function [x, iters] = ...
    CompositeStep(Hess, Jac, grad, cnstr, m, tol, radius)
% Minimizes a QO using a composite-step trust-region approach
% 
% inputs:
%  Hess - the Hessian
%  Jac - the Jacobian
%  grad - the NEGATIVE gradient at x0
%  cnstr - the NEGATIVE constraint value at x0
%  m - maximum number of iterations
%  tol - tolerance target
%  radius - trust region radius
% 
% outputs:
%  x - the solution=
%  iters - number of matrix-vector products
%--------------------------------------------------------------------------

% solve for the normal step that satisfies the constraint to tol
[x_norm, iters_norm] = NormalStep(Jac, cnstr, m, tol, 0.8*radius);
% solve for the tangential step that satisfies tol, or until trust
% radius/negative curvature encountered
[x_comp, iters_tang] = TangentialStep(Hess, Jac, grad, cnstr, x_norm, m, tol, radius);
x = x_comp;
iters = iters_norm + iters_tang;
end
