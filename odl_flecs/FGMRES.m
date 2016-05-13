function [x, res, iters, hist] = FGMRES(Hess, Jac, grad, cnstr, x, m, tol)
% Flexible Generalized Minimal Residual Method
% 
% inputs:
%  Hess - the Hessian
%  Jac - the Jacobian
%  grad - the NEGATIVE gradient; the primal rhs
%  cnstr - the NEGATIVE constraint value; the dual rhs
%  x - initial guess (not presently used; x0 = 0 is assumed)
%  m - maximum number of iterations
%  tol - tolerance target
% 
% outputs:
%  x - the solution
%  res - residual norm of entire primal-dual system
%  iters - number of matrix-vector products
%  hist - array of relative residual values
%--------------------------------------------------------------------------

nVar = size(grad,1);
nCeq = size(cnstr,1);
n = nVar+nCeq;

% calculate the norm of the rhs vector
grad0 = norm(grad,2);
feas0 = norm(cnstr,2);
norm0 = sqrt(grad0^2 + feas0^2);

grad_scale = 1.0;
feas_scale = 1.0;
% uncomment to scale the primal and dual equations (not used in paper)
% if (grad0 > feas0)
%     grad_scale = feas0/grad0;
% else
%     feas_scale = grad0/feas0;
% end;
b = [grad.*grad_scale; cnstr.*feas_scale];

% calculate the initial residual, and use it to find V(:,1)
V = zeros(n, m+1);
Z = zeros(n, m);
H = zeros(m+1, m);
VtV_dual = zeros(m+1,m+1);
V(:,1) = b;
beta = norm(V(:,1),2);
omega = norm(V(1:nVar,1),2);
gamma = norm(V(nVar+1:n,1),2);
V(:,1) = V(:,1)./beta;
VtV_dual(1,1) = V(nVar+1:n,1)'*V(nVar+1:n,1);

% initalize the rhs of the reduced system
g = zeros(m+1,1);
g(1) = beta;

% loop over all search directions
y = zeros(m,1);
hist(1,1) = 1.0;
for i = 1:m
    iters = i;
    % precondition the vector V(:,i) and store in Z(:,i)
    Z(:,i) = V(:,i); % no preconditioning in Matlab version at this time
    
    % apply primal dual matrix
    V(1:nVar,i+1) = grad_scale.*Hess*Z(1:nVar,i) + ...
    		  feas_scale.*Jac'*Z(nVar+1:n,i);
    V(nVar+1:n,i+1) = grad_scale.*Jac*Z(1:nVar,i);
    
    % scale matvec
    V(1:nVar,i+1) = V(1:nVar,i+1).*grad_scale;
    V(nVar+1:n,i+1) = V(nVar+1:n,i+1).*feas_scale;
    
    % modified Gram-Schmidt
    [V, H] = ModGramSchmidt(i+1, V, H);
    
    % compute some inner prodcuts
    for k = 1:i        
        VtV_dual(k,i+1) = V(nVar+1:n,k)'*V(nVar+1:n,i+1);
        VtV_dual(i+1,k) = VtV_dual(k,i+1);
    end;
    g_tang(i+1) = V(1:nVar,i+1)'*b(1:nVar);
    VtV_dual(i+1,i+1) = V(nVar+1:n,i+1)'*V(nVar+1:n,i+1);
    
    % solve the reduced problem and compute the residual
    [y, beta, gamma] = SolveReduced(i, H, g, VtV_dual);
    omega = sqrt(max(beta.^2 - gamma.^2, 0.0));
    res_norm = (gamma/feas_scale).^2 + (omega/grad_scale).^2;
    res_norm = sqrt(max(0.0, res_norm));
    hist(i+1,1) = res_norm/norm0;
    
    % check for convergence
    if ( (omega < tol*grad0*grad_scale) && (gamma < tol*feas0*feas_scale) )
        break;
    end;
end;
% compute solution
x = Z(:,1:i)*y(1:i);
x(1:nVar) = grad_scale.*x(1:nVar);
x(nVar+1:n) = feas_scale.*x(nVar+1:n);
res = hist(iters,1);
end

%==========================================================================
function [y, beta, gamma] = SolveReduced(i, H, g, VtV_dual)
% Finds subspace step by minimizing combination of primal and dual
% residual norms.
% 
% inputs:
%  i - current iteration
%  H - upper Hessenberg matrix from Arnoldi's method
%  g - reduced problem rhs
%  VtV_dual - inner products involving V(nVar+1:n,:) and itself
% 
% outputs:
%  y - solution to primal-dual problem in subspace
%  beta - norm of (primal-dual) residual using y
%  gamma - norm of the feasiblity using y
%--------------------------------------------------------------------------

% solve the reduced (primal-dual) problem and compute the residual
y = H(1:i+1,1:i)\g(1:i+1);
res_red = H(1:i+1,1:i)*y(1:i) - g(1:i+1);
beta = norm(res_red, 2);
gamma = sqrt(res_red'*VtV_dual(1:i+1,1:i+1)*res_red);
end
