% This script compares the iterative method FLECS to a composite-step 
% algorithm.  The algorithms are applied to quadratic optimization problems
% whose Hessians are nonconvex in the null space of the Jacobian.
close all;
clear all;

radius = 1.0; % trust radius
tol = 1e-1; % tolerance used in iterative solvers
kappa = 100; % condition number of the Hessian
num_sample = 100; % number of samples to take

null_dim = zeros(num_sample,1);
feas_qual = zeros(num_sample,1);
obj_qual = zeros(num_sample,1);
sol_qual = zeros(num_sample,1);
mult_qual = zeros(num_sample,1);
num_iter = zeros(num_sample,1);
norm_len = zeros(num_sample,1);
rel_res = zeros(num_sample,1);
prob_dim = zeros(num_sample,1);
compare = 0;
tic;
for k = 1:num_sample
    k
    n = randi([10,100]);
    m = randi([1,n-1]);
    null_dim(k) = (n-m)/n;

    % generate the QO
    [H, A, g, c, x0, norm_len(k)] = BuildQO(n, m, kappa, false, true);    
    obj0 = 0.5*x0'*H*x0;
    feas0 = norm(c + A*x0, 2);
    
    % solve the QO using composite step (projected CG)
    [dx_comp, iters_comp] = CompositeStep(H, A, -g, -c-A*x0, n+m, tol, radius);
        
    % define composite-step solution, objective, and feasibility
    x_comp = x0 + dx_comp;
    obj_comp = 0.5*x_comp'*H*x_comp;
    feas_comp = norm(A*x_comp + c, 2);
    
    % solve using FLECS
    mu = 100.0/feas0;
    [dx, iters, hist] = FLECS(H, A, -g, -c-A*x0, zeros(n+m,1), iters_comp, ...
        1e-10, radius, mu);
      
    num_iter(k) = iters./iters_comp;
    
    % define FLECS solution, objective, and feasibility
    x_flecs = x0 + dx(1:n);
    obj_flecs = 0.5*x_flecs'*H*x_flecs;
    feas_flecs = norm(A*x_flecs + c,2);    
    
    % define objective and feasibility quality
    obj_qual(k) = (obj_flecs-obj_comp)/abs(obj0-obj_comp);
    feas_qual(k) = (feas_flecs-feas_comp)/(feas0-feas_comp);
    rel_res(k) = norm(H*dx(1:n) + A'*dx(n+1:n+m) + g)./norm(g);
    prob_dim(k) = n+m;
    
    clear H A g c x0 hist;
end;
T = toc
[null_dim, sort_indx] = sort(null_dim);
obj_qual = obj_qual(sort_indx);
feas_qual = feas_qual(sort_indx);
num_iter = num_iter(sort_indx);
norm_len = norm_len(sort_indx);
rel_res = rel_res(sort_indx);
prob_dim = prob_dim(sort_indx);
%KKT_cond = KKT_cond(sort_indx);

% write output to file
dlmwrite('consistent.dat',null_dim','delimiter',' ','precision','%16.10e');
dlmwrite('consistent.dat',obj_qual','-append','delimiter',' ','precision','%16.10e')
dlmwrite('consistent.dat',feas_qual','-append','delimiter',' ','precision','%16.10e')
dlmwrite('consistent.dat',num_iter','-append','delimiter',' ','precision','%16.10e')
dlmwrite('consistent.dat',rel_res','-append','delimiter',' ','precision','%16.10e')
dlmwrite('consistent.dat',prob_dim','-append','delimiter',' ','precision','%16.10e')
