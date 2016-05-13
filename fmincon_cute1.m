function fmincon_cute1()
% using fmincon to optimize on CUTEr test problem
% used for comparison with RSNK result
clear all
% close all
% figure()

addpath('/home/pengfei/Developer/Inequality_tests/odl_flecs')
addpath('/home/pengfei/Developer/CUTEst/cutest/src/matlab')

% change the SIF problem name  
prob_name = 'CHAIN';         % 'BT11'  BT6;  % 'BLOCKQP1';   % 'AUG2D';

% routines for using SIF problem set
unix(['cutest2matlab ', prob_name])

prob = cutest_setup();

options = optimoptions(@fmincon,'Algorithm','active-set','GradObj','on','GradConstr', 'on',...
    'PlotFcns',{@optimplotfval,@myoptimplotfirstorderopt},'Display','iter');   

m_eq_gen = sum(prob.equatn); 
m_ieq_gen = 2*(prob.m - m_eq_gen);    % general inequ constraints doubled 
% m_ieq_bound = 2*prob.n;               % bound constraints, two sides
% m_con = m_eq_gen + m_ieq_gen + m_ieq_bound;             % need to count in bound constraints?

x0 = prob.x; 
 
[x, fval, exitflag, output,lambda] = fmincon(@myfun, x0, [], [], [], [], prob.bl, prob.bu, @mycon, options) 


% output the activeness for inequality constraints at the final point
[c_ieq, ceq, cg_ieq, gceq] = mycon(x); 

% inactive_idx = find(c_ieq>0)         % inactive constraints 

if isempty(find(c_ieq > 0, 1))    % no infeasibility constraints
    fprintf('the final point is feasible \n') 
else 
    infeas_idx = find(c_ieq > 0); 
    fprintf('Infeasible points exist! \n') 
    fprintf('Infeasible index: %d, ', infeas_idx)
end

%
    function [f,g] = myfun(x)       
        [f, g] = cutest_obj(x);          
    end

    function [c_ieq, ceq, cg_ieq, gceq] = mycon(x)
        
        [c_temp, cg_temp] = cutest_cons(x);
        
        if m_ieq_gen > 0            
            c_ieq_gen = c_temp(~prob.equatn);           
                c_ieq = (-1).*[ c_ieq_gen - prob.cl(~prob.equatn); 
                         -c_ieq_gen + prob.cu(~prob.equatn)]; 
                     
            cg_ieq_gen = cg_temp(~prob.equatn, :);     
                cg_ieq = (-1).*[ cg_ieq_gen;
                                -cg_ieq_gen]';                  
        else
            c_ieq = [];
            cg_ieq = []; 
        end
               
        if m_eq_gen > 0    
            ceq = c_temp(prob.equatn); 
            gceq = cg_temp(prob.equatn, :)'; 
        else
           ceq = []; 
           gceq = []; 
        end        
    end
end


function stop = myoptimplotfirstorderopt(x,optimValues,state,varargin)
% OPTIMPLOTFIRSTORDEROPT Plot first-order optimality at each iteration.
%
%   STOP = OPTIMPLOTFIRSTORDEROPT(X,OPTIMVALUES,STATE) plots
%   OPTIMVALUES.firstorderopt.
%
%   Example:
%   Create an options structure that will use OPTIMPLOTFIRSTORDEROPT as the
%   plot function
%     options = optimset('PlotFcns',@optimplotfirstorderopt);
%
%   Pass the options into an optimization problem to view the plot
%      fmincon(@(x) 3*sin(x(1))+exp(x(2)),[1;1],[],[],[],[],[0 0],[],[],options)

%   Copyright 2006-2011 The MathWorks, Inc.
%   $Revision: 1.1.6.8 $  $Date: 2012/08/21 01:07:46 $

persistent plotavailable
stop = false;

switch state
    case 'iter'
        if optimValues.iteration == 1
            if isfield(optimValues,'firstorderopt') && ~isempty(optimValues.firstorderopt)
                plotavailable = true;

                % The 'iter' case is  called during the zeroth iteration, but
                % firstorderopt may still  be empty.  Start plotting at the
                % first iteration.
                plotfirstorderopt = plot(optimValues.iteration,optimValues.firstorderopt,'kd', ...
                    'MarkerFaceColor',[1 0 1]);
                title(getString(message('optim:optimplot:TitleFirstOrderOpt', ...
                    sprintf('%g',optimValues.firstorderopt))),'interp','none');
                xlabel(getString(message('optim:optimplot:XlabelIter')),'interp','none');
                ylabel(getString(message('optim:optimplot:YlabelFirstOrderOpt')),'interp','none');
                set(plotfirstorderopt,'Tag','optimplotfirstorderopt');
            else % firstorderopt field does not exist or is empty
                plotavailable = false;
                title(getString(message('optim:optimplot:TitleFirstOrderOpt', ...
                    getString(message('optim:optimplot:NotAvailable')))),'interp','none');
            end
        else
            if plotavailable
                plotfirstorderopt = findobj(get(gca,'Children'),'Tag','optimplotfirstorderopt');
                newX = [get(plotfirstorderopt,'Xdata') optimValues.iteration];
                newY = [get(plotfirstorderopt,'Ydata') optimValues.firstorderopt];
                set(plotfirstorderopt,'Xdata',newX, 'Ydata',newY);
                set(get(gca,'Title'),'String', ...
                    getString(message('optim:optimplot:TitleFirstOrderOpt', ...
                    sprintf('%g',optimValues.firstorderopt))));
            end
        end
        
       % sprintf('First-Order-Opt, %f', optimValues.firstorderopt)
end
end
