function as4_fmincon
% figure(1)
% plot_obj_con
% hold all

options = optimoptions(@fmincon,'Algorithm','sqp','GradObj','on','GradConstr', 'on',...
    'PlotFcns',{@optimplotx_my,@optimplotfirstorderopt_my});   % @optimplotfval,} 


x0 = [20; 20]; 
[f, df] = myfun(x0)
[c, ceq, gc, gceq] = mycon(x0)

lb = [5; 5];
ub = [25;25];   

k=0; kg = 0; 

[x, fval, exitflag, output, lambda, grad, hessian] = fmincon(@myfun, x0, [],[],[],[],lb,ub, @mycon, options)


function stop = optimplotx_my(x,optimValues,state,varargin)

stop = false;
switch state
    case 'iter'
        k=k+1;
        x_progress(:,k) = x; 
end
end


function stop = optimplotfirstorderopt_my(x,optimValues,state,varargin)

stop = false;

switch state
    case 'iter'
        kg= kg+1; 
        gk_progress(kg) = optimValues.firstorderopt;  
end

end

save gk_fmincon.mat gk_progress

% x_progress = x_progress(:,any(x_progress,1));   % recording the iteration design points
figure(1)
hold all
plot(x_progress(1,:), x_progress(2,:),'yx-','linewidth',2.0,'MarkerSize',4)

end


function [f, df] = myfun(AS)

f = total_drag(AS); 
df = Drag_grad_complx(AS); 

end


function [D] = total_drag(AS)
A = AS(1); S = AS(2); 

rho = 1.23;
mu = 17.8e-6;
V = 35; 
Swet = 2.05*S; 
k = 1.2; 
e = 0.96; 

%------------------------
W = total_weight(AS); 
%------------------------

CL = 2*W/(rho*V^2*S); 

Re = rho*V/mu*sqrt(S/A);  
Cf = 0.074/(Re^0.2);
CD_1 = k*Cf*Swet/S + CL^2/(pi*A*e); 

CD = 0.03062702/S + CD_1; 

D = W*(CD/CL); 

end

function [dfdAS] = Drag_grad_complx(AS)
h = 1e-60; 

A_i = AS + [complex(0, h); 0]; 

D_i_A = total_drag(A_i); 
dfdA = imag(D_i_A)/h; 

S_i = AS + [0; complex(0, h)]; 
D_i_S = total_drag(S_i); 
dfdS = imag(D_i_S)/h; 

dfdAS = [dfdA; dfdS];

end


function [c, ceq, gc, gceq] = mycon(AS)

c = constr(AS); 
gc = con_grad_complx(AS); 

ceq = []; 
gceq = []; 
end


function con = constr(AS)

rho = 1.23;
V_min = 22;        % m/s 
CL_max = 2.0;  

A = AS(1); S = AS(2); 
W = total_weight(AS); 

con = 2*W/(rho*V_min^2*CL_max) - S; 

end


function [dCon_dAS] = con_grad_complx(AS)

h = 1e-60; 

A_i = AS + [complex(0, h); 0]; 

D_i_A = constr(A_i); 
dfdA = imag(D_i_A)/h; 

S_i = AS + [0; complex(0, h)]; 
D_i_S = constr(S_i); 
dfdS = imag(D_i_S)/h; 

dCon_dAS = [dfdA; dfdS];

end


function W = total_weight(AS)
A = AS(1); S = AS(2); 

W_0 = 4940; 
N_ult = 2.5; 
t_c = 0.12; 

c = W_0 + 45.42*S; 
d = 8.71*1e-5 * N_ult / t_c * A * sqrt(S*A); 

% a * x^2 + b * x + c = 0
a = 1; b = -(2*c + d^2*W_0); e_coef = c^2;     

W = (-b + sqrt(b^2 - 4*a*e_coef))/(2*a); 

end


function plot_obj_con

% create a grid of points at which to evaluate the function and the
% constraint
[X,Y] = meshgrid(5:.1:25,5:0.1:25);
f = zeros(size(X,1),size(X,2));
c = zeros(size(X,1),size(X,2));

% evaluate the function and constraint at the points
for i = 1:size(X,1)
    for j = 1:size(X,2)
        f(i,j) = total_drag([X(i,j) Y(i,j)]');
        c(i,j) = constr([X(i,j) Y(i,j)]');
    end;
end;

% plot the contours of the objective function

% v = [1.e-30];
% plot the constraint
[obj_fun,h] = contourf(X,Y,f,50);
colormap gray
hold on;
[cnstr_fun,h2] = contour(X,Y,c,[0 0],'-b','linewidth',1.5, 'ShowText', 'on');

end
