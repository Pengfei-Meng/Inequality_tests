function [x, iters, hist, pred_aug, pred_trust, step_ytZtZy, arnoldi_struct] = FLECS(Hess, Jac, grad, cnstr, x, m, tol, radius, mu)
% FLexible Equality-Constrained Subproblem solver for inexactly minimizing
% nonconvex quadratics with linear equality constraints
% 
% inputs:
%  Hess - Hessian of the quadratic objective (or Lagrangian)
%  Jac - the constraint Jacobian
%  grad - the NEGATIVE gradient; the primal rhs
%  cnstr - the NEGATIVE constraint value; the dual rhs
%  x - initial guess (not presently used; x0 = 0 is assumed)
%  m - maximum number of iterations
%  tol - relative tolerance target for the FGMRES primal and dual norms
%  radius - trust-region radius for primal problem
%  mu - quadratic-penalty parameter
% 
% outputs:
%  x - the primal-dual solution
%  iters - number of KKT-matrix-vector products
%  hist - array of FGMRES relative residual values
%--------------------------------------------------------------------------

nVar = size(grad,1); % number of variables
nCeq = size(cnstr,1); % number of constraints
n = nVar+nCeq;
iters = 0;

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
VtZ = zeros(m+1,m);
VtZ_prim = VtZ;
VtZ_dual = VtZ;
ZtZ_prim = zeros(m,m);
VtV_dual = zeros(m+1,m+1);
V(:,1) = b;
beta = norm(V(:,1),2); % the primal-dual system residual norm
gamma = norm(V(nVar+1:n,1),2); % the primal residual norm
omega = norm(V(1:nVar,1),2); % the dual residual norm
V(:,1) = V(:,1)./beta;
VtV_dual(1,1) = V(nVar+1:n,1)'*V(nVar+1:n,1);

% initalize the rhs of the FGMRES reduced system
g = zeros(m+1,1);
g(1) = beta;

% loop over all search directions
y = zeros(m,1);
hist(1,1) = 1.0;
step_violate = false;
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
%     V
%     Z
    % compute some inner prodcuts
    for k = 1:i
        VtZ_prim(k,i) = V(1:nVar,k)'*Z(1:nVar,i);
        VtZ_prim(i+1,k) = V(1:nVar,i+1)'*Z(1:nVar,k);
        VtZ_dual(k,i) = V(nVar+1:n,k)'*Z(nVar+1:n,i);
        VtZ_dual(i+1,k) = V(nVar+1:n,i+1)'*Z(nVar+1:n,k);
        VtZ(k,i) = VtZ_prim(k,i) + VtZ_dual(k,i);
        VtZ(i+1,k) = VtZ_prim(i+1,k) + VtZ_dual(i+1,k);
        ZtZ_prim(k,i) = Z(1:nVar,k)'*Z(1:nVar,i);
        ZtZ_prim(i,k) = ZtZ_prim(k,i);
        VtV_dual(k,i+1) = V(nVar+1:n,k)'*V(nVar+1:n,i+1);
        VtV_dual(i+1,k) = VtV_dual(k,i+1);
    end;
    VtV_dual(i+1,i+1) = V(nVar+1:n,i+1)'*V(nVar+1:n,i+1);
%     ZtZ_prim
    % solve the reduced problems and compute the residual

    [y, y_aug, y_mult, beta, gamma, ~, ~, step_violate, pred, pred_aug, step_ytZtZy, pred_trust] = ...
        ReducedSpaceSol(i, radius/grad_scale, H, g, mu, ZtZ_prim, VtZ, ...
        VtZ_prim, VtZ_dual, VtV_dual);

    % check convergence criteria
    omega = sqrt(max(beta.^2 - gamma.^2, 0.0));
    res_norm = (gamma/feas_scale).^2 + (omega/grad_scale).^2;
    res_norm = sqrt(max(0.0, res_norm));
    hist(i+1,1) = res_norm/norm0;
    
    arnoldi_struct = 0;

    if ( (omega < tol*grad0*grad_scale) && (gamma < tol*feas0*feas_scale) )  
        
%         field1 = 'H';  value1 = H;
%         field2 = 'g';  value2 = g;
%         field3 = 'i';  value3 = i;
%         field4 = 'ZtZ_prim';    value4 = ZtZ_prim;
%         field5 = 'VtZ';         value5 = VtZ;
%         field6 = 'VtZ_prim';  value6 = VtZ_prim;
%         field7 = 'VtZ_dual';  value7 = VtZ_dual;
%         field8 = 'VtV_dual';  value8 = VtV_dual; 
%         field9 = 'nVar';  value9 = nVar; 
%         field10 = 'n';  value10 = n; 
%         field11 = 'Z'; value11 = Z; 
%   
%         arnoldi_struct = struct(field1,value1,field2,value2,field3,value3,field4,value4, field5,value5,field6,value6,field7,value7,field8,value8, field9, value9, field10,value10, field11, value11); 
        arnoldi_struct = 0;
        sprintf('arnoldi_sturct complete')
        break;
    end;
%         if pred_aug > 1e-5   % this prevent negative pred_aug, but not tiny positive pred_aug, how to? 
%             break;           
%         else 
%             % check convergence criteria
%             omega = sqrt(max(beta.^2 - gamma.^2, 0.0));
%             res_norm = (gamma/feas_scale).^2 + (omega/grad_scale).^2;
%             res_norm = sqrt(max(0.0, res_norm));
%             hist(i+1,1) = res_norm/norm0;
%             if ( (omega < tol*grad0*grad_scale) && (gamma < tol*feas0*feas_scale) )        
%                break;
%             end    
%             
%         end
    % end;           
end;



% while pred_aug < 0
%     mu = 10*mu 
%     [y, y_aug, y_mult, beta, gamma, ~, ~, step_violate, pred, pred_aug, step_ytZtZy, pred_trust] = ...
%                     ReducedSpaceSol(i, radius/grad_scale, H, g, mu, ZtZ_prim, VtZ, ...
%                               VtZ_prim, VtZ_dual, VtV_dual);   
% 
%     pred_aug                                                           
% end 

if (step_violate)
    display('trust radius exceeded by FGMRES');
end;

% construct the solution
x(1:nVar) = Z(1:nVar,1:i)*y_aug(1:i);
x(nVar+1:n) = Z(nVar+1:n,1:i)*y_mult(1:i);
x(1:nVar) = grad_scale.*x(1:nVar);
x(nVar+1:n) = feas_scale.*x(nVar+1:n);

end

%==========================================================================
function [y, y_aug, y_mult, beta, gamma, beta_aug, gamma_aug, ...
    step_violate, pred, pred_aug, step_ytZtZy, pred_trust] = ...
    ReducedSpaceSol(i, radius, H, g, mu, ZtZ_prim, VtZ, VtZ_prim, ...
    VtZ_dual, VtV_dual)
% Finds a globalized step for FLECS by solving quadratic-penalty function
% minimization in subspace constructed by flexible Arnoldi
% 
% inputs:
%  i - current iteration
%  radius - the trust-region radius
%  H - upper Hessenberg matrix from Arnoldi's method
%  g - reduced problem rhs
%  mu = penalty parameter
%  ZtZ_prim - inner products involving Z(1:nVar,:) and itself
%  VtZ - inner products involving V(:,:) and Z(:,:)
%  VtZ_prim - inner products involving V(1:nVar,:) and Z(1:nVar,:)
%  VtZ_dual - inner products involving V(nVar+1:n,:) and Z(nVar+1:n,:)
%  VtV_dual - inner products involving V(nVar+1:n,:) and itself
% 
% outputs:
%  y - FGMRES subspace solution to primal-dual problem
%  y_aug - primal subspace solution to quadratic penalty subproblem
%  y_mult - dual subspace solution
%  beta - norm of FGMRES (primal-dual) residual using y
%  gamma - norm of the feasiblity using y
%  beta_aug - norm of (primal-dual) residual using y_aug
%  gamma_aug - norm of constraint equation using y_aug
%  step_violate - true if the primal-dual solution violates the radius
%  pred - the predicted objective function reduction using y
%  pred_aug - the predicted objective function reduction using y_aug
%--------------------------------------------------------------------------

% solve the reduced (primal-dual) problem and compute the residual
y = H(1:i+1,1:i)\g(1:i+1);
res_red = H(1:i+1,1:i)*y(1:i) - g(1:i+1);
beta = norm(res_red, 2);
gamma = sqrt(res_red'*VtV_dual(1:i+1,1:i+1)*res_red);

% check length of FGMRES primal step
ytZtZy = y'*ZtZ_prim(1:i,1:i)*y;
if ( sqrt(ytZtZy) > radius)
    step_violate = true;
    step_ytZtZy = sqrt(ytZtZy); 
else
    step_violate = false;
    step_ytZtZy = radius; 
end;

% build linear system for quadratic penalty subspace problem
Hred = VtZ(1:i+1,1:i)'*H(1:i+1,1:i) - VtZ_dual(1:i+1,1:i)'*H(1:i+1,1:i) ...
    - H(1:i+1,1:i)'*VtZ_dual(1:i+1,1:i);
ZtJactJacZ = H(1:i+1,1:i)'*VtV_dual(1:i+1,1:i+1)*H(1:i+1,1:i);
Haug = Hred + mu*ZtJactJacZ;
gaug = (-g(1)*VtZ_prim(1,1:i) - g(1)*mu*VtV_dual(1,1:i+1)*H(1:i+1,1:i))';

% find transformation to account for potentially linearly dependent Z_prim
[U,S,V] = svd(ZtZ_prim(1:i,1:i));
ZtZ_rank = rank(ZtZ_prim(1:i,1:i));
T = U(:,1:ZtZ_rank)*inv(sqrt(S(1:ZtZ_rank,1:ZtZ_rank)));
% T = eye(size(gaug,1));

% Horig = Haug; 
% gorig = gaug; 

Haug = T'*Haug*T;
gaug = T'*gaug;

% Solve reduced-space trust-region problem:
[y_aug, val, posdef, ~, ~] = trust(gaug, Haug, radius);
y_aug = real(T*y_aug);

% compute the norms
res_red = H(1:i+1,1:i)*y_aug(1:i) - g(1:i+1);
beta_aug = norm(res_red, 2);
gamma_aug = sqrt(max(res_red'*VtV_dual(1:i+1,1:i+1)*res_red,0.0));

% compute the dual reduced-space solution
y_mult = y;

% determine objective function reductions
pred = -0.5*y'*Hred*y + g(1)*VtZ_prim(1,1:i)*y;
% pred_aug = -0.5*y_aug'*Hred*y_aug + g(1)*VtZ_prim(1,1:i)*y_aug;

pred_aug = -val; 
% pred_test = -0.5*y_aug'*Horig*y_aug - gorig'*y_aug; 


pred_trust = pred_aug; 
% pred_aug = -0.5*y_aug'*Haug*y_aug + g(1)*VtZ_prim(1,1:i)*y_aug;

end
