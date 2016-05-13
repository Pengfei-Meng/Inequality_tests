function [x, pred_aug, pred_trust] = FLECS_resolve(arnoldi_struct, x, tol, radius, mu)

grad_scale = 1.0;
feas_scale = 1.0;

i = arnoldi_struct.i; 
H = arnoldi_struct.H; 
g = arnoldi_struct.g; 
ZtZ_prim = arnoldi_struct.ZtZ_prim;
VtZ  = arnoldi_struct.VtZ;
VtZ_prim = arnoldi_struct.VtZ_prim;
VtZ_dual = arnoldi_struct.VtZ_dual;
VtV_dual = arnoldi_struct.VtV_dual; 
nVar = arnoldi_struct.nVar; 
n = arnoldi_struct.n; 
Z = arnoldi_struct.Z; 

[y, y_aug, y_mult, beta, gamma, ~, ~, step_violate, pred, pred_aug, step_ytZtZy, pred_trust] = ...
    ReducedSpaceSol(i, radius/grad_scale, H, g, mu, ZtZ_prim, VtZ, ...
    VtZ_prim, VtZ_dual, VtV_dual);

% if ( (omega < tol*grad0*grad_scale) && (gamma < tol*feas0*feas_scale) )          
%     return;
% end;

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