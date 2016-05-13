function temp5_inequ_con()
% To test: 
% min f = x^2 - y^2;
% s.t. -1 < y < 0
% this is working quite well!
% with proper history output files 

close all
clc
addpath('./odl_flecs')

plot_contour(-2, 2, -2, 2)

m_eq = 0; 
m_ieq = 2;  
m_con = m_eq + m_ieq; 

% xk =  [0.51, 0.52]; 
 xk = [-1, -0.5]; 
% xk = [-0.5, -1.5];
% xk = [1.5, -0.5]; 
% xk = [-0.5, 1]; 
    
lambda = 0.1.*ones(1, m_con); 
% lambda = zeros(1, m_con); 
plot_point(xk, 'y', 'x');

xold = xk; 
max_iter = 50*(length(xk) + length(lambda));    % n+m;

mu_init = 5;      % 10;     % 0.01;  
mu_pow = 1;       % 0.5;    % 1;

mu = mu_init;
radius = 0.25;
krylov_tolerance = 1e-2;
reduced_nu = 0.95;
des_tol = 1e-6;
ceq_tol = 1e-6;

k = 0;

hisfile = 'hist_inequ_1.txt'; 
file = fopen(hisfile,'w');
fprintf(file, 'krylov_tolerance : %e,  des_tol : %e\n', krylov_tolerance, des_tol);

while k < max_iter
    k = k+1;
    [f, g, h] = objs(xk);
    [c, c_g, c_h] = cons(xk);
       
    %% --------------------------------------------
    % zero out the inactive inequality constraints
    
    subspace_size = length(xk) + sum(find(c(m_eq+1 : m_con)<0));
    
    row_keep = m_eq + find(c(m_eq+1:m_con)<=0);  % index of active inequ cons
    m_active = length(row_keep);              % num of active inequ cons

    if find(c(m_eq+1 : m_con)>0) ~= 0                % if inactive exists, inside feasible region
        row_delte = m_eq + find(c(m_eq+1:m_con)>0);  % idx of inactive inequ
        m_inactive = length(row_delte);           % num of inactive inequ

        % zero the inactive row of Jacob matrix and inequality constraints
        c_g(row_delte,:) = zeros(size(c_g(row_delte,:)));
        c(row_delte,:) = zeros(size(c(row_delte,:)));

        for i=1:m_inactive
            c_h{row_delte(i)} = zeros(size(c_h{row_delte(i)}));
        end
        
        % subspace_size = length(xk) + length(lambda) - m_inactive; 
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
        mu = max(mu, mu_init*(feas_norm0/feas_norm)^(mu_pow));
        kkt_norm = sqrt(feas_norm*feas_norm + grad_norm*grad_norm);
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
    
    xk = xk + dx(1:length(xk));
    lambda = lambda + dx(length(xk)+1:end);
    
    plot_line(xold, xk, 'o')
    xold = xk; 
    
    pause(1)
end

fclose(file);
plot_point(xk, 'g', '^')
end


function [f, f_g, f_h] = objs(x)
n = length(x); 
f = x(1)^2 - x(2)^2; 
f_g = [2*x(1), -2*x(2)]; 
f_h = [2,0; 0,-2];
end

function [c, c_g, c_h] = cons(x)
% input x: row vector
% s.t.  -1 < y < 0
% ~     y + 1 > 0;  -y > 0 
n = length(x);
c = zeros(n,1);
c(1) = x(2) + 1; 
c(2) = -x(2); 

c_g = zeros(2, n); 
c_h = cell(1,2);
  
c_g(1,:) = [0, 1];
c_g(2,:) = [0, -1];

c_h{1} = zeros(2,2);
c_h{2} = zeros(2,2);


% s.t.  -1 < y < 0
% ~     y > x^2 - 1;  -y > 0 
% n = length(x);
% c = zeros(n,1);
% c(1) = x(2) - x(1)^2 + 1; 
% c(2) = -x(2); 
% 
% c_g = zeros(2, n); 
% c_h = cell(1,2);
%   
% c_g(1,:) = [-2*x(1), 1];
% c_g(2,:) = [0, -1];
% 
% c_h{1} = [-2,0; 0,0];
% c_h{2} = zeros(2,2);

end

function plot_contour(xmin, xmax, ymin, ymax)
% min  f = x^2 - y^2
% s.t. -1 < y < 0
% create a grid of points at which to evaluate the function and the
% constraint
y_low = -1;  y_high = 0; 

[X,Y] = meshgrid(xmin:.1:xmax, ymin:0.1:ymax);
f = zeros(size(X,1),size(X,2));

% evaluate the function and constraint at the points
for i = 1:size(X,1)
    for j = 1:size(X,2)
        f(i,j) = X(i,j)^2 - Y(i,j)^2;       
    end
end

% plot the contours of the objective function
[C,h] = contourf(X,Y,f,20);
colorbar
% surf(X,Y,f);
xlabel('X')
ylabel('Y')
hold all
line([xmin, xmax], [y_high, y_high],'Color','b','LineWidth', 2, 'MarkerSize', 9);
hold all

% y > -1
line([xmin, xmax], [y_low, y_low],'Color','b','LineWidth', 2, 'MarkerSize', 9);  % 'Marker','o',
hold all


% y = x^2 - 1
% for i = 1:size(X,1)
%     for j = 1:size(X,2)
%         c(i,j) = Y(i,j) - X(i,j)^2 + 1;       
%     end
% end
% 
% contour(X,Y,c,[0 0],'Color','b','LineWidth', 2);
% hold all

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
