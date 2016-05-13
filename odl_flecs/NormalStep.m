function [x, iters] = NormalStep(Jac, cnstr, m, tol, radius)
% Computes a normal step that inexactly satisfies the constraint and the
% trust-region radius (uses GMRES, although MINRES is equivalent here)
% 
% inputs:
%  Jac - the Jacobian
%  cnstr - the NEGATIVE constraint value at x0
%  m - maximum number of iterations
%  tol - tolerance target
%  radius - trust region radius
% 
% outputs:
%  x - primal step that inexactly satisfies the constraint
%  iters - number of matrix-vector products (augmented system)
%--------------------------------------------------------------------------

nVar = size(Jac,2);
nCeq = size(Jac,1);
n = nVar+nCeq;
b = [zeros(nVar,1); cnstr];

% calculate the initial residual norm, and use it to find V(:,1)
V = zeros(n, m+1);
H = zeros(m+1, m);
V(:,1) = b;
beta = norm(V(:,1),2);
norm0 = beta;
V(:,1) = V(:,1)./beta;

% initalize the rhs of the reduced system
g = zeros(m+1,1);
g(1) = beta;

% loop over all search directions
y = zeros(m,1);
for i = 1:m
    iters = i;
    % precondition the vector V(:,i) and store in Z(:,i)
    Z(:,i) = V(:,i); % no preconditioning at this time
    
    % apply augmented-system matrix
    V(1:nVar,i+1) = V(1:nVar,i) + Jac'*V(nVar+1:n,i);
    V(nVar+1:n,i+1) = Jac*V(1:nVar,i);
    
    % modified Gram-Schmidt
    [V, H, lin_depend] = ModGramSchmidt(i+1, V, H);
    
    % solve the reduced problem and compute the residual
    y = H(1:i+1,1:i)\g(1:i+1);
    res_red = H(1:i+1,1:i)*y(1:i) - g(1:i+1);
    beta = norm(res_red, 2);
    
    % check for convergence
    if (beta < tol*norm0)
        break;
    end;
    if ( (lin_depend) && (beta > tol*norm0) )
        error('NormalStep: Arnoldi-process breakdown');
    end;
end;
% compute solution and scale if necessary
x = V(1:nVar,1:i)*y(1:i);
len = norm(x);
if (len > radius)
    x = x.*radius/len;
end;
end