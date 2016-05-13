% This script compares the iterative method FLECS to FGMRES. The algorithms
% are applied to quadratic optimization problems whose Hessians are convex
% in the null space of the Jacobian.
close all;
clear all;

tol = 1e-1; % tolerance used in iterative solvers
kappa = 100; % condition number of the Hessian
num_sample = 100; % number of samples to take

null_dim = zeros(num_sample,1);
feas_qual = zeros(num_sample,1);
obj_qual = zeros(num_sample,1);
num_iter = zeros(num_sample,1);
norm_len = zeros(num_sample,1);
rel_res = zeros(num_sample,1);
prob_dim = zeros(num_sample,1);
tic;
for k = 1:num_sample
    k
    n = randi([10,100]);
    m = randi([1,n-1]);
    null_dim(k) = (n-m)/n;

    % generate the QO
    [H, A, g, c, x0, norm_len(k)] = BuildQO(n, m, kappa, true, true);    
    obj0 = 0.5*x0'*H*x0;
    feas0 = norm(c + A*x0, 2);
    
    % solve for stationary point using FGMRES
    [dx_gmres, res_gmres, iters_gmres, hist_gmres] = ...
        FGMRES(H, A, -g, -c-A*x0, zeros(n+m,1), n+m, tol);    
    
    % define FGMRES solution, objective, and feasibility
    x_gmres = x0 + dx_gmres(1:n);
    obj_gmres = 0.5*x_gmres'*H*x_gmres;
    feas_gmres = norm(A*x_gmres + c, 2);   
    
    % solve using FLECS
    radius = norm(x_gmres,2)*100.0;
    mu = 100.0./feas0;
    [dx, iters, hist] = FLECS(H, A, -g, -c-A*x0, zeros(n+m,1), ...
        min(iters_gmres,n+m), tol, radius, mu);    
    
    % define FLECS solution, objective, and feasibility
    x_flecs = x0 + dx(1:n);
    obj_flecs = 0.5*x_flecs'*H*x_flecs;
    feas_flecs = norm(A*x_flecs + c,2);         
    
    num_iter(k) = iters./iters_gmres;
    
    % define objective and feasibility quality
    obj_qual(k) = (obj_flecs-obj_gmres)/abs(obj0-obj_gmres);
    feas_qual(k) = (feas_flecs-feas_gmres)/(feas0-feas_gmres);    
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

% write output to file
dlmwrite('convex.dat',null_dim','delimiter',' ','precision','%16.10e');
dlmwrite('convex.dat',obj_qual','-append','delimiter',' ','precision','%16.10e');
dlmwrite('convex.dat',feas_qual','-append','delimiter',' ','precision','%16.10e');
dlmwrite('convex.dat',num_iter','-append','delimiter',' ','precision','%16.10e');
dlmwrite('convex.dat',rel_res','-append','delimiter',' ','precision','%16.10e');
dlmwrite('convex.dat',prob_dim','-append','delimiter',' ','precision','%16.10e');


