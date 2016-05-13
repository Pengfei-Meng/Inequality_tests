function [H, A, g, c, x0, norm_len] = BuildQO(n, m, kappa, convex, consistent)
% Creates a synthetic Hessian and Jacobian for a quadratic optimization
% problem
% 
% inputs:
%  n - number of variables
%  m - number of constraints
%  kappa - condition number of the Hessian
%  convex - if true, Hessian is PD in null space
%  consistent - if true, problem will be consistent with ||x|| < 1.0 
% 
% outputs:
%  H - Hessian
%  A - Jacobian
%  g - gradient at x0


%  c - constraint at x0
%  x0 - a random start point of unit length
%  norm_len - shortest distance from x0 to constraints
%--------------------------------------------------------------------------

if (n < m) 
    error('num variables must be greater than or equal to num constraints')
end;
if (kappa < 0)
    error('condition number must be positive')
end;

% generate the synthetic eigenvalues of the Hessian
eig = rand(n,1);
alpha = (1/kappa-1)/(max(eig)-min(eig));
beta = 1 - alpha*min(eig);
eig = eig.*alpha + beta;

% first m eigenvalues can be positive or negative
eig(1:m) = eig(1:m).*(2.*randi(2,m,1)-3.*ones(m,1));

% last n-m eigenvalues can be negative if ~convex
if (~convex)
    % if not convex, force eigenvalues to be negative in null space
    while (min(eig(m+1:n)) > 0.0)
        eig(m+1:n) = eig(m+1:n).*(2.*randi(2,n-m,1)-3.*ones(n-m,1));
    end;
end;

% generate synthetic eigenvectors
R = rand(n,n);
V = orth(R);
if (size(V,2) ~= n)
    % with random matrices, the likelihood of this is low
    error('Problem with BuildQO');  
end;

% generate the synthetic Hessian
H = V*diag(eig)*V';

% generate the synthetic Jacobian
A = rand(m,m)*V(1:n,1:m)';

% generate the start point
x0 = 2.*rand(n,1) - ones(n,1);
x0 = x0/norm(x0);

% generate the gradient
g = H*x0;

% generate the constraint
% c = Ax + b = Ax - Ax_perp
% where x_perp = x0 + Qy and Q is from the QR decomp of A', and y is a set
% of random values scaled such that the problem is feasible
[Q,R] = qr(A',0);
x_norm = Q*rand(m,1);
if (consistent)
    norm_len = 0.5*rand(1); % fac keeps QO feasible
else
    norm_len = 1 + rand(1); % infeasible QO
end;
x_norm = norm_len*x_norm./norm(x_norm);
c = -A*(x0 + x_norm);

end

