function temp6_rosen_final()
% To test: 
% min f = 100*(x(1)^2 - x(2))^2 + (x(1) - 1)^2;
% s.t.  1 - x^2 - y^2 = 0
% 
% with proper history output files 

close all
clc
addpath('./odl_flecs')

% contourplot()
openfig('contour_rosen_con.fig')
hold all

m_eq = 1; 
m_ieq = 0;  
m_con = m_eq + m_ieq; 

% change here for starting point
% go back and forth with this point, for equality constrained case! 
% xk =  [-0.3753, 0.1311];
xk = [-1, 1]; 
lambda = 0;    % -3.7624; 

% starting close to the right boundary
% xk = [0.7846, 0.6148];
% lambda = -0.123;

mu_init = 10;      % 10;     % 0.01;  
mu_pow = 1;       % 0.5;    % 1;
radius = 0.25;


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
       
    %--------------------------------------------
%     % zero out the inactive inequality constraints    
%     subspace_size = length(xk) + sum(find(c(m_eq+1 : m_con)<0));
%     
%     row_keep = m_eq + find(c(m_eq+1:m_con)<=0);  % index of active inequ cons
%     m_active = length(row_keep);              % num of active inequ cons
% 
%     if find(c(m_eq+1 : m_con)>0) ~= 0                % if inactive exists, inside feasible region
%         row_delte = m_eq + find(c(m_eq+1:m_con)>0);  % idx of inactive inequ
%         m_inactive = length(row_delte);           % num of inactive inequ
% 
%         % zero the inactive row of Jacob matrix and inequality constraints
%         c_g(row_delte,:) = zeros(size(c_g(row_delte,:)));
%         c(row_delte,:) = zeros(size(c(row_delte,:)));
% 
%         for i=1:m_inactive
%             c_h{row_delte(i)} = zeros(size(c_h{row_delte(i)}));
%         end
%         
%         % subspace_size = length(xk) + length(lambda) - m_inactive; 
%     end

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
    
    
    % do something on lambda?     
    xk = xk + dx(1:length(xk));    
    lambda = lambda + dx(length(xk)+1:end);
    
%     if lambda > 0       % incorrect sign  
%        lambda = 0   -lambda;  
%     end
    
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

f = 100*(x(1)^2 - x(2))^2 + (x(1) - 1)^2;
f_g = zeros(1,2);
f_h = zeros(2,2);

f_g(1) = 400*x(1)*(x(1)^2-x(2)) + 2*(x(1)-1);
f_g(2) = -200*(x(1)^2 - x(2));

f_h(1,1) = 400*(3*x(1)^2 - x(2)) + 2;
f_h(1,2) = -400*x(1);
f_h(2,1) = -400*x(1);
f_h(2,2) = 200;

end

function [c, c_g, c_h] = cons(x)
% input x: row vector
% 2 constraints
% s.t.  1 - x^2 - y^2 = 0

c = zeros(1,1);  
c_g = zeros(1,2); 
c_h = cell(1,1);

% uncomment the following for constrained case
c(1) = 1 - x(1)^2 - x(2)^2; 
c_g(1,:) = -2.*x; 
c_h{1} = -2.*diag([1,1]);

% uncomment the following for unconstrained case
% c(1) = 0;      
% c_g(1,:) = [0,0]; 
% c_h{1} = zeros(2,2);

end

