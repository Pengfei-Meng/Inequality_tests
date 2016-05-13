function temp7_equ_rosen()
% opt+flecs:  2D rosenbrock with 1 equality constraint, working all right! 
% 2D rosenbrock: 
% min f = 100*(x(1)^2 - x(2))^2 + (x(1) - 1)^2;
% s.t.  1 - x^2 - y^2 = 0
% for equality constrained case
% it goes back and forth with a ceratin starting point! 


% starting point matters! change the result a lot! 
close all
clc
addpath('./odl_flecs')

% contourplot()
openfig('contour_rosen_con.fig')
hold all


n = 2;
nVar = n; 
m_eq = 1; 
m_ieq = 0;  
m_con = m_eq + m_ieq; 

% xk =  [-0.50,0.50]; 
% lambda = -0.1.*ones(1, m_con); 

% go back and forth with this point, for equality constrained case! 
xk =  [-0.3753, 0.1311];

% xk = [-1, 1.5]; 
lambda = 0;    % -3.7624; 

% starting close to the right boundary
% xk = [0.7846, 0.6148];
% lambda = -0.123;


plot(xk(1), xk(2), 'Color','y','Marker','x','LineWidth', 2, 'MarkerSize', 12)
hold all
xold = xk; 

max_iter = 100*(length(xk) + length(lambda));    % n+m;
f_c_hist = zeros(max_iter, 4); 

subspace_size = (length(xk) + length(lambda));
mu_init = 10;      % 10;     % 0.01;  
mu_pow = 1.0;       % 0.5;    % 1;

mu = mu_init;
radius = .5;
krylov_tolerance = 1e-2;
reduced_nu = 0.95;
des_tol = 1e-6;
ceq_tol = 1e-6;

k = 0;

while k < max_iter
    k = k+1;
    [f, g, h] = objs(xk);
    [c, c_g, c_h] = cons(xk);
    
    % --- bookeeping ----
    f_c_hist(k, 1) = xk(1); 
    f_c_hist(k, 2) = xk(2); 
    f_c_hist(k, 3) = f; 
    f_c_hist(k, 4) = c; 
     
    % the special block for only one inequality constraint! 
    if c >= 1e-2                    % if inequ con inactive
       c_g = zeros(size(c_g));
       c = 0; 
       c_h{1} = zeros(size(c_h{1})); 
       subspace_size = length(xk); 
    else
       subspace_size = (length(xk) + length(lambda));
    end
    
%     if lambda >= 0
%         lambda = 0;
%     end
    
    dLdX = (g + lambda*c_g)';
    dLdXX = h;
    for kc = 1:length(c)
        dLdXX = dLdXX + lambda(kc)*c_h{kc};
    end

      
    if (k == 1)
        grad_norm0 = norm(dLdX);
        grad_norm = grad_norm0;
        feas_norm0 = norm(c);
        feas_norm = feas_norm0;
        kkt_norm0 = sqrt(feas_norm0*feas_norm0 + grad_norm0*grad_norm0);
        kkt_norm = kkt_norm0;
        grad_tol = des_tol * grad_norm0;
        feas_tol = ceq_tol * max(feas_norm0, 1e-6);
        
    else
        grad_norm = norm(dLdX);
        feas_norm = norm(c);
        mu = max(mu, mu_init*(feas_norm0/feas_norm)^(mu_pow));
        kkt_norm = sqrt(feas_norm*feas_norm + grad_norm*grad_norm);
        
    end
    
    % dLdX_save = dLdX;
    
    krylov_tol = krylov_tolerance * min(1, sqrt( kkt_norm/kkt_norm0 ));
    if 1
        krylov_tol = max(krylov_tol, min(grad_tol/grad_norm, feas_tol/feas_norm));
    else
        krylov_tol = max(krylov_tol, 0.001);
    end
    krylov_tol = krylov_tol * reduced_nu;
            
    if  (grad_norm < grad_tol)  && (feas_norm < feas_tol)   %  (norm(dLdX) < 1e-6)   
        fprintf('converged!')
        break
    end

    [dx, iters, hist] = FLECS(dLdXX, c_g, -dLdX, -c, xk, subspace_size, krylov_tol, radius, mu);
    
    xk = xk + dx(1:length(xk));
    lambda = lambda + dx(length(xk)+1:end)';
    
    hold all
    line([xold(1), xk(1)], [xold(2), xk(2)],'Color','r','Marker','o','LineWidth', 2, 'MarkerSize', 9);
    hold all
    xold = xk; 

    pause(1)
    
end
plot(xk(1), xk(2), 'Color','g','Marker','^','LineWidth', 2, 'MarkerSize', 9)
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
