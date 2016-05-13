function opt_test()
% testing the inequality framework on CUT%
% To test:
% min f = 100*(x(1)^2 - x(2))^2 + (x(1) - 1)^2;
% s.t.  1 - x^2 - y^2 = 0
%
% with proper history output files

clear all
% close all

addpath('/home/pengfei/Developer/Inequality_tests/odl_flecs')

%------------------------------------
m_eq_gen = 0;
m_ieq_gen = 1;                     % general inequ constraints doubled
m_con = m_eq_gen + m_ieq_gen;


xk = [1,1];
lambda = 0;
xold = xk; 

x_old = xk;
max_iter = 50*(length(xk) + length(lambda));

% penalty parameter updating scheme
mu_init = 10;      % 10;     % 0.01;
mu_pow = 1;       % 0.5;    % 1;
mu = mu_init;

eta = 3/16;
radius = 1;               % radius can be changed
krylov_tolerance = 1e-2;
reduced_nu = 0.95;
des_tol = 1e-6;
ceq_tol = 1e-6;

grad_tol = 0;
feas_tol = 0;
kkt_norm0 = 0;

mu_1 = 1e-4;
mu_2 = 0.6;
alpha_init = 1;
alpha_max = 1.5;

k = 0;

hisfile = 'temp'; 
% hisfile = [prob_name '.log'];
file = fopen(hisfile,'w');
fprintf(file, 'krylov_tolerance : %e,  des_tol : %e\n', krylov_tolerance, des_tol);

plot_contour(-2, 2, -2, 2)
hold all
plot_point(xk, 'y', 'x');
hold all


while k < max_iter
    k = k+1;
    
    [f, c, L, dLdX, dLdXX, c_reformat, cg_reformat, lambda, subspace_size] = lag_obj_grad(xk, lambda);
    
    merit_this = L + mu/2*norm(c_reformat)^2;
    
%     % plotting -----------------------
%     subplot(2,2,1)
%     plot(k, L, 'ro')
%     hold all
%     
%     subplot(2,2,2)
%     plot(k, f, 'gd')
%     hold all
%     
%     subplot(2,2,3)
%     plot(k, lambda'*c_reformat, 'bx')
%     hold all
    % plotting end --------------------
    
    [break_idx, krylov_tol] = writing_file(file, k, f, c, dLdX, lambda, c_reformat, des_tol, ceq_tol, mu, krylov_tolerance, reduced_nu);
    % lambda; %+ alpha.*dx(le
    if break_idx
        fprintf(file, '\n xk: %f, ', xk);
        
        break
    end
    
    [dx, iters, hist, pred_aug, pred_trust, step_ytZtZy, arnoldi_struct] = FLECS(dLdXX, cg_reformat, -dLdX, -c_reformat, xk, subspace_size, krylov_tol, radius, mu);
    
    %% Merit function and trust region block
    % alpha = linesearch(@merit_obj_grad, xk, lambda, dx, mu_1,...
    % mu_2, alpha_init, alpha_max, 10);
    
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
            
            % [dx, pred_aug, pred_trust] = FLECS_resolve(arnoldi_struct, xk, krylov_tol, radius, mu);
            
            fprintf('xk, lambda don''t get updated at this iteration, %d, \n', k)
        end
    end
    
    plot_line(xold, xk, 'o')
    xold = xk;
    pause(1)
        
    fprintf(file, 'back_tracking alpha: %f  \n', alpha);
    
end


% feasibility or infeasibility
if isempty(find(c_reformat < -1e-6, 1))    % no infeasibility constraints
    fprintf('the final point is feasible \n')
else
    infeas_idx = find(c_reformat < 0);
    fprintf('Infeasible points exist! \n')
    fprintf('Infeasible index: %d, \n', infeas_idx)
end
fclose(file);

plot_point(xk, 'g', '^')

%% Auxilarating functions

    function [merit, grad_merit] = merit_obj_grad(alpha)
        trial_xk = xk + alpha.*dx(1:length(xk));
        trial_lambda = lambda;         %+ alpha.*dx(length(xk)+1:end);
        
        [~, ~, L_loc, dLdX_loc, ~, c_reformat_loc, cg_reformat_loc, ~, ~] = lag_obj_grad(trial_xk, trial_lambda);
        
        merit = L_loc + mu/2*norm(c_reformat_loc)^2;
        grad_merit = dLdX_loc + ((mu/2*2).*c_reformat_loc'*cg_reformat_loc)';
    end

    function [f_loc, c_loc, L_loc, dLdX_loc, dLdXX_loc, c_reformat_loc, cg_reformat_loc, lambda_loc, subspace_size_loc] = lag_obj_grad(xk_loc, lambda_loc)
        % calculate dLdX, dLdXX at the current xk, lambda point
        [f_loc, g_loc, h_loc] = objs(xk_loc);
        [c_loc, c_g_loc, c_h_loc] = cons(xk_loc);
                                
        c_reformat_loc = c_loc; 
        cg_reformat_loc = c_g_loc; 
        ch_reformat_loc = c_h_loc; 
                
        %% check and zero inactive inequalities
        
        crit_vec = -lambda_loc(m_eq_gen+1 : m_con) - c_loc;
        
        subspace_size_loc = length(xk_loc) + length(find(crit_vec > 0));
        
        num_inactive = length(find( crit_vec < 0));
        
        if num_inactive > 0               % if inactive exists, inside feasible region
            
            idx_inactive = m_eq_gen + find(crit_vec < 0);      % idx of inactive inequ
            
            % zero the inactive row of Jacob matrix and inequality constraints
            cg_reformat_loc(idx_inactive,:) = zeros(size(cg_reformat_loc(idx_inactive,:)));
            c_reformat_loc(idx_inactive,:) = zeros(size(c_reformat_loc(idx_inactive,:)));
            
            % zero the inactive constraint Hessian
            for i=1:num_inactive
                ch_reformat_loc{idx_inactive(i)} = zeros(size(ch_reformat_loc{idx_inactive(i)}));
            end
            
            lambda_loc(idx_inactive) = 0;
            subspace_size_loc = length(xk_loc) + length(lambda_loc) - num_inactive;
            
        end
        
        %% assemble the gradient and hessian of the Lagrangian for active cons
        L_loc = f_loc + lambda_loc'*c_reformat_loc;
        dLdX_loc = (g_loc' + lambda_loc'*cg_reformat_loc');
        dLdXX_loc = h_loc;
        for kc = 1:(m_eq_gen + m_ieq_gen)
            dLdXX_loc = dLdXX_loc + lambda_loc(kc)*ch_reformat_loc{kc};  %c_h is in a different format
        end
        
    end

    function [break_idx, krylov_tol] = writing_file(file, k, f, c, dLdX, lambda, c_reformat, des_tol, ceq_tol, mu,  krylov_tolerance, reduced_nu)
        
        break_idx = 0;
        
        if (k == 1)
            grad_norm0 = norm(dLdX);
            grad_norm = grad_norm0;
            feas_norm0 = norm(c_reformat);
            feas_norm = feas_norm0;
            feas_norm_old = feas_norm0;
            kkt_norm0 = sqrt(feas_norm0*feas_norm0 + grad_norm0*grad_norm0);
            kkt_norm = kkt_norm0;
            grad_tol = des_tol * grad_norm0;
            feas_tol = ceq_tol * max(feas_norm0, 1e-6);
            fprintf(file, 'initial grad_norm : %e,  target grad_norm : %e\n', grad_norm0, grad_tol);
            fprintf(file, 'initial feas_norm : %e,  target feas_norm : %e\n', feas_norm0, feas_tol);
            
        else
            grad_norm = norm(dLdX);
            feas_norm = norm(c_reformat);  % the NEGATIVE gradient; the primal
            %     mu = max(mu, mu_init*(feas_norm0/feas_norm)^(mu_pow));
            
            %     if feas_norm > 1.05*feas_norm_old    % increased violation, so increase penalty
            %         mu = 5*mu;
            %     end
            
            % feas_norm_old = feas_norm;
            kkt_norm = sqrt(feas_norm*feas_norm + grad_norm*grad_norm);
        end
        
        fprintf(file, '-------------------------------------------------------------\n');
        fprintf(file, 'nonlinear iteration : %d\n', k);
        fprintf(file, 'current obj : %e, grad_norm : %e, feas_norm: %e\n',f, grad_norm, feas_norm);
        fprintf(file, 'mu : %f, \n', mu);
        fprintf(file, 'norm(lambda, Inf): %f  \n', norm(lambda, Inf));
        
        if (grad_norm < grad_tol)  && (feas_norm < feas_tol)
            fprintf('converged! \n')
            fprintf(file, '\n-------------------------------------------------------------\n');
            fprintf(file, 'optimization loop terminated because: grad_norm < grad_tol && feas_norm < feas_tol\n');
            fprintf(file, 'outer iter: %d, ', k);
            fprintf(file, 'f: %f, ', f);
            fprintf(file, 'c: %f, ', c);
            fprintf(file, 'grad_norm : %e, feas_norm: %e \n', grad_norm, feas_norm);
            
            break_idx = 1;
        end
        
        krylov_tol = krylov_tolerance * min(1, sqrt( kkt_norm/kkt_norm0 ));
        if 1
            krylov_tol = max(krylov_tol, min(grad_tol/grad_norm, feas_tol/feas_norm));
        else
            krylov_tol = max(krylov_tol, 0.001);
        end
        krylov_tol = krylov_tol * reduced_nu;
        fprintf(file, 'krylov_tol : %e \n', krylov_tol);
        % fprintf('norm of dLdX: %f \n', norm(dLdX));
    end

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
df_pk = grad_merit_0'*dx(1:length(xk))';


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