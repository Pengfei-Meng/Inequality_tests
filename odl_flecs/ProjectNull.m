function [p, iters] = ProjectNull(Jac, x, m, tol)
% Inexactly projects a given vector onto the null space of the Jacobian by
% solving an augmented system (using GMRES)
% 
% inputs:
%  Jac - the Jacobian
%  x - the vector being projected
%  m - maximum number of iterations
%  tol - tolerance target
% 
% outputs:
%  p - x projected unto null space of Jac
%  iters - number of matrix-vector products (augmented system)
%--------------------------------------------------------------------------

nVar = size(Jac,2);
nCeq = size(Jac,1);
n = nVar+nCeq;

b = [x; zeros(nCeq,1)];

% calculate the initial residual, and use it to find V(:,1)
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
    [V, H] = ModGramSchmidt(i+1, V, H);
    
    % solve the reduced problem and compute the residual
    y = H(1:i+1,1:i)\g(1:i+1);
    res_red = H(1:i+1,1:i)*y(1:i) - g(1:i+1);
    beta = norm(res_red, 2);
    
    % check for convergence
    if (beta < tol*norm0)
        break;
    end;
end;
% compute solution
p = V(1:nVar,1:i)*y(1:i);
end