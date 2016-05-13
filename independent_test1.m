function independent_test1()
% To test:
% min f = 100*(x(1)^2 - x(2))^2 + (x(1) - 1)^2;
% s.t.  1 - x^2 - y^2 = 0
%
% with proper history output files

close all
clc
addpath('./odl_flecs')

plot_contour(-2, 2, -2, 2)
hold all

m_eq = 0;
m_ieq = 1;
m_con = m_eq + m_ieq;

% change here for starting point
% go back and forth with this point, for equality constrained case!
% xk =  [-0.3753, 0.1311];
xk = [-1, 1];
lambda = 0;    % -3.7624;

% starting close to the right boundary
% xk = [0.7846, 0.6148];
% lambda = -0.123;

mu_init = 5;      % 10;     % 0.01;
mu_pow = 1;       % 0.5;    % 1;
radius = 0.5;

mu_1 = 1e-4; 
%-------------------------------------------
plot_point(xk, 'y', 'x');
% subspace_size = (length(xk) + length(lambda));
xold = xk;
max_iter = 50*(length(xk) + length(lambda));    % n+m;
mu = mu_init;
krylov_tolerance = 1e-2;
reduced_nu = 0.95;
des_tol = 1e-6;
ceq_tol = 1e-6;

k = 0;

hisfile = 'hist_rosen.txt';
file = fopen(hisfile,'w');
fprintf(file, 'krylov_tolerance : %e,  des_tol : %e\n', krylov_tolerance, des_tol);

while k < max_iter
    k = k+1;
    [f, g, h] = objs(xk);
    [c, c_g, c_h] = cons(xk);
    
    % here activeness criteria is changed
    crit_vec = -lambda(m_eq+1 : m_con) - c(m_eq+1 : m_con);
    num_inactive = length(find( crit_vec < 0));
    
    if num_inactive > 0           % if inactive exists, inside feasible region
        
        idx_inactive = m_eq + find(crit_vec < 0);      % idx of inactive inequ
        
        % zero the inactive row of Jacob matrix and inequality constraints
        c_g(idx_inactive,:) = zeros(size(c_g(idx_inactive,:)));
        c(idx_inactive,:) = zeros(size(c(idx_inactive,:)));
        
        % for i=1:num_inactive
        c_h = zeros(size(c_h));
        % end
        
        lambda(idx_inactive) = 0;
        subspace_size = length(xk) + length(lambda) - num_inactive;
        
    end
    
    
    
    % the special block for only one inequality constraint!
    if c >= -1.0/mu                    % if inequ con inactive
        c_g = zeros(size(c_g));
        c = 0;
        c_h{1} = zeros(size(c_h{1}));
        subspace_size = length(xk); 6
    else
        subspace_size = (length(xk) + length(lambda));
    end
    %-------------------------------------------
    
    dLdX = (g + lambda*c_g)';
    dLdXX = h;
    for kc = 1:length(c)
        dLdXX = dLdXX + lambda(kc)*c_h{kc};
    end
    
    if (k == 1)
        grad_norm0 = norm(dLdX);
        grad_norm = grad_norm0;
        feas_norm0 = norm(c);
        feas_norm_old = feas_norm0;
        feas_norm = feas_norm0;
        kkt_norm0 = sqrt(feas_norm0*feas_norm0 + grad_norm0*grad_norm0);
        kkt_norm = kkt_norm0;
        grad_tol = des_tol * grad_norm0;
        feas_tol = ceq_tol * max(feas_norm0, 1e-6);
        fprintf(file, 'initial grad_norm : %e,  target grad_norm : %e\n', grad_norm0, grad_tol);
        fprintf(file, 'initial feas_norm : %e,  target feas_norm : %e\n', feas_norm0, feas_tol);
        
    else
        grad_norm = norm(dLdX);
        feas_norm = norm(c);
        kkt_norm = sqrt(feas_norm*feas_norm + grad_norm*grad_norm);
        
        % update mu
        if feas_norm > feas_norm_old     % increased constraint violation
            mu = 100*mu;
            % lambda unchanged;
        else                      % reduced constraint violation
            mu = max(mu, mu_init*(feas_norm0/feas_norm)^(mu_pow));
            lambda = lambda + mu * c;
        end
        
        % mu = max(mu, mu_init*(feas_norm0/(1e-6 + feas_norm))^(mu_pow))
        % mu = mu_init*(feas_norm0/(1e-6 + feas_norm))^(mu_pow);
        feas_norm_old = feas_norm;
    end
    
    fprintf(file, '-------------------------------------------------------------\n');
    fprintf(file, 'nonlinear iteration : %d\n', k);
    fprintf(file, 'current obj : %e, grad_norm : %e, feas_norm: %e\n',f, grad_norm, feas_norm);
    fprintf(file, 'lambda: %f, %f  ', lambda(:));
    
    
    if (grad_norm < grad_tol)  && (feas_norm < feas_tol)   %  (norm(dLdX) < 1e-6)
        fprintf('converged!')
        fprintf(file, '\n-------------------------------------------------------------\n');
        fprintf(file, 'optimization loop terminated because: grad_norm < grad_tol && feas_norm < feas_tol\n');
        fprintf(file, 'outer iter: %d,  xk: %f %f,  f: %f, c: %f %f \n', k, xk(:), f, c(:));
        fprintf(file, 'grad_norm : %e, feas_norm: %e \n', grad_norm, feas_norm);
        fprintf(file, 'final lambda: %f %f \n',  lambda(:));
        break
    end
    
    krylov_tol = krylov_tolerance * min(1, sqrt( kkt_norm/kkt_norm0 ));
    if 1
        krylov_tol = max(krylov_tol, min(grad_tol/grad_norm, feas_tol/feas_norm));
    else
        krylov_tol = max(krylov_tol, 0.001);
    end
    krylov_tol = krylov_tol * reduced_nu;
    fprintf(file, 'krylov_tol : %e \n', krylov_tol);
    
    
    [dx, iters, hist] = FLECS(dLdXX, c_g, -dLdX, -c, xk, subspace_size, krylov_tol, radius, mu);
    
    if k == 1
        % update xk anyway, 'cos initial lambda all zero. step only reduce
        % objective function
        alpha = 1;
        xk = xk + alpha.*dx(1:length(xk));
        lambda = lambda + alpha.*dx(length(xk)+1:end);
    else
        
        rho = 0.5;   max_alpha = 10;   alpha_min = 0.01;
        alpha = back_tracking(@merit_obj_grad, xk, dx, mu_1, rho,  max_alpha, alpha_min);
        merit_next = merit_obj_grad(alpha);
        
        rho_k = (merit_this - merit_next) /  (pred_aug) ;
        pred_aug;
        % pred_trust
        
        if rho_k < 0.1
            radius = radius/4;
        elseif (rho_k > 3/4)   %  && (norm(dx(1:length(xk)),2)==radius)
            radius = min(2*radius, 20);
        end
        
        if rho_k > eta
            xk = xk + alpha.*dx(1:length(xk));
            lambda = lambda + alpha.*dx(length(xk)+1:end);
        else
            
            [dx, pred_aug, pred_trust] = FLECS_resolve(arnoldi_struct, xk, krylov_tol, radius, mu);
            
            fprintf('xk, lambda don''t get updated at this iteration, %d, \n', k)
            %             if radius < 1e-5
            %                 break
            %             end
        end
    end
    
    
    plot_line(xold, xk, 'o')
    xold = xk;
    
    pause(1)
end

fclose(file);
plot_point(xk, 'g', '^')
end

function plot_point(x0, color, marker)
hold all
plot(x0(1), x0(2), 'Color',color,'Marker',marker,'LineWidth', 2, 'MarkerSize', 12)
hold all
end

function plot_line(xold, xk, marker)
hold all
line([xold(1), xk(1)], [xold(2), xk(2)],'Color','r','Marker', marker, 'LineWidth', 2, 'MarkerSize', 9);
hold all
end

function [f, f_g, f_h] = objs(x)
% n = length(x);
f = x(1) + x(2)^2;
f_g = [1, 2*x(2)];
f_h = [0, 0; 0, 2];
end

function [c, c_g, c_h] = cons(x)
% input x: row vector
% s.t.  e^x - 1 >= 0

c = exp(x(1)) - 1;
c_g = [exp(x(1)), 0];
c_h{1} = [exp(x(1)), 0; 0, 0];

end

function plot_contour(xmin, xmax, ymin, ymax)
% min  f = x + y^2
% s.t. e^x - 1 >= 0
% create a grid of points at which to evaluate the function and the
% constraint
% y_low = -1;  y_high = 0;

[X,Y] = meshgrid(xmin:.1:xmax, ymin:0.1:ymax);
f = zeros(size(X,1),size(X,2));

% evaluate the function and constraint at the points
for i = 1:size(X,1)
    for j = 1:size(X,2)
        f(i,j) = X(i,j) + Y(i,j)^2;
    end
end

% plot the contours of the objective function
[C,h] = contourf(X,Y,f,20);
colorbar
% surf(X,Y,f);
xlabel('X')
ylabel('Y')
hold all
% line([xmin, xmax], [y_high, y_high],'Color','b','LineWidth', 2, 'MarkerSize', 9);
% hold all

% % y > -1
% line([xmin, xmax], [y_low, y_low],'Color','b','LineWidth', 2, 'MarkerSize', 9);  % 'Marker','o',
% hold all


% y = exp(x) - 1;
for i = 1:size(X,1)
    for j = 1:size(X,2)
        c(i,j) = exp(X(i,j)) - 1;
    end
end

contour(X,Y,c,[0 0],'Color','b','LineWidth', 2);
hold all

end

function alpha = back_tracking(merit_fun, xk, dx, mu_1, rho,  max_iter, alpha_min)
 
alpha = 0;  
[merit_0, grad_merit_0] = merit_fun(alpha);
df_pk = grad_merit_0'*dx(1:length(xk)); 


alpha = 1;  
[merit_alpha, ~] = merit_fun(alpha); 
n_iter = 0; 

while (alpha > alpha_min) && (n_iter < max_iter)
   
    n_iter = n_iter+1;     
    if merit_alpha <= merit_0 + mu_1*alpha*df_pk     
       % fprintf('backtracking succeed \n');
       % alpha
       break
    else
       alpha = alpha*rho;  
       [merit_alpha, ~] = merit_fun(alpha); 
    end 
end
end
