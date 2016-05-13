function opt_cute1()
% testing the inequality framework on CUT%

clear all
% close all


% addpath is only for directory containing matlab routines
% to add directory containing LINUX routine, use setenv... 

addpath('/home/pengfei/Developer/Inequality_tests/odl_flecs')
addpath('/home/pengfei/Developer/CUTEst/cutest/src/matlab')
% addpath('/home/pengfei/Developer/CUTEst/sif')

%% THIS IS VERY IMPORTANT, COST MANY HOURS.....
% at every MATLAB restart, do thelambda; %+ alpha.*dx(le following path setting
% PATH=getenv('PATH')
% setenv('PATH',[PATH ':/home/pengfei/Developer/CUTEst/cutest/bin']);
% setenv('PATH',[PATH ':/home/pengfei/Developer/CUTEst/sif']);
% setenv('PATH',[PATH ':/home/pengfei/MATLAB/R2013a/bin']);

% change the SIF problem name  
prob_name = 'BT11';     % 'BLOCKQP1';    %  'AUG2D';         %   % ; % 'HS76';    

% routines for using SIF problem set
unix(['cutest2matlab ', prob_name])

prob = cutest_setup();

%------------------------------------
m_eq_gen = sum(prob.equatn); 
m_ieq_gen = 2*(prob.m - m_eq_gen);    % general inequ constraints doubled 
m_ieq_bound = 2*prob.n;               % bound constraints, two sides
m_con = m_eq_gen + m_ieq_gen + m_ieq_bound;             % need to count in bound constraints?

xk = prob.x; 
lambda = [prob.v(prob.equatn); prob.v(~prob.equatn); prob.v(~prob.equatn); zeros(m_ieq_bound,1)]; 

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

hisfile = [prob_name '.log']; 
file = fopen(hisfile,'w');
fprintf(file, 'krylov_tolerance : %e,  des_tol : %e\n', krylov_tolerance, des_tol);

figure()
subplot(2,2,1)
xlabel('iteration')
ylabel('Lagrangian value')
title('Lagrangian')
hold all

subplot(2,2,2)
xlabel('iteration')
ylabel('Objective value')
title('f')
hold all

subplot(2,2,3)
xlabel('iteration')
ylabel('Lambda*Constraint value')
title('\lambda c')
hold all


while k < max_iter
    k = k+1;

    [f, c, L, dLdX, dLdXX, c_reformat, cg_reformat, lambda, subspace_size] = lag_obj_grad(xk, lambda);
    
    merit_this = L + mu/2*norm(c_reformat)^2; 
    
    % plotting -----------------------
    subplot(2,2,1)
    plot(k, L, 'ro')
    hold all
    
    subplot(2,2,2)
    plot(k, f, 'gd')
    hold all
    
    subplot(2,2,3)
    plot(k, lambda'*c_reformat, 'bx')
    hold all
    % plotting end --------------------
       
    [break_idx, krylov_tol] = writing_file(file, k, f, c, dLdX, lambda, c_reformat, des_tol, ceq_tol, mu, krylov_tolerance, reduced_nu);
                 % lambda; %+ alpha.*dx(le
    if break_idx
       fprintf(file, '\n xk: %f, ', xk);        
        
       break 
    end
   
    [dx, iters, hist, pred_aug, pred_trust, step_ytZtZy, arnoldi_struct] = FLECS(dLdXX, cg_reformat, -dLdX, -c_reformat, xk, subspace_size, krylov_tol, radius, mu);
    
    [dx2, pred_aug2, pred_trust2] = FLECS_resolve(arnoldi_struct, xk, krylov_tol, radius, mu); 
    
    %% Merit function and trust region block         
         

                      
                      
                      
    
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
    
    [f_loc, g_loc] = cutest_obj(xk_loc);         
    h_loc = cutest_ihess(xk_loc, 0);        % Hessian of the objective function       
    [c_loc, c_g_loc] = cutest_cons(xk_loc);

%     plot(k, f_loc, 'ro')
%     pause(1)
%     hold all

    c_h_loc = cell(1,prob.m); 
    for i_con = 1 : prob.m
        c_h_loc{i_con} = cutest_ihess(xk_loc, i_con);   
    end
    %%  Reformulate inequality & equality constraints            
    ch_reformat_loc = cell(1,m_con);               % bound consraint no hessian
    if m_ieq_gen == 0                      % zero of inequality constraints 
          c_ieq_loc = [xk_loc - prob.bl;
                      -xk_loc + prob.bu ];      
          cg_ieq_loc = [eye(prob.n);
                       -eye(prob.n)];               

    elseif m_ieq_gen > 0
         c_ieq_gen_loc = c_loc(~prob.equatn);             % general inequality        
             c_ieq_loc = [ c_ieq_gen_loc - prob.cl(~prob.equatn); 
                          -c_ieq_gen_loc + prob.cu(~prob.equatn);
                              xk_loc - prob.bl;
                             -xk_loc + prob.bu ]; 

         cg_ieq_gen_loc = c_g_loc(~prob.equatn, :);       % general inequality Jacobin            
             cg_ieq_loc = [ cg_ieq_gen_loc;
                           -cg_ieq_gen_loc;             
                               eye(prob.n);
                              -eye(prob.n)];    
         ch_reformat_loc(m_eq_gen+1 : prob.m) = c_h_loc(~prob.equatn); 
         
        for idx = 1:m_ieq_gen/2
            ch_reformat_loc{prob.m + idx} = -c_h_loc{m_eq_gen + idx}; 
        end
    end

    if m_eq_gen == 0           % in case no equality general constraints
        c_reformat_loc = c_ieq_loc; 
        cg_reformat_loc = cg_ieq_loc; 
 
    elseif m_eq_gen > 0               
        c_reformat_loc = [c_loc(prob.equatn);
                       c_ieq_loc]; 
        cg_reformat_loc = [c_g_loc(prob.equatn,:);
                       cg_ieq_loc ];                    
        ch_reformat_loc(1 : m_eq_gen) = c_h_loc(prob.equatn); 
    end

    for idx = (m_eq_gen+m_ieq_gen+1):m_con
        ch_reformat_loc{idx} = zeros(size(c_h_loc{1})); 
    end

    %% check and zero inactive inequalities 
    
    crit_vec = -lambda_loc(m_eq_gen+1 : m_con) - c_ieq_loc;     

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
    dLdX_loc = (g_loc' + lambda_loc'*cg_reformat_loc)';
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