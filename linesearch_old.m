function [alpha, fnew, feval, dfeval] = linesearch(merit_fun, xk, lambda, pk, mu_1,...
    mu_2, alpha_init, alpha_max, max_iter)
% Purpose: returns a step length that satisfies the strong-Wolfe conditions
% Inputs:
%   obj - a function handle for the objective
%   grad - a function handle for the objective gradient
%   xk - the location in the design space where the line search starts
%   pk - the search direction; must be a descent direction for obj
%   mu_1 - the sufficient decrease parameter; 0 < mu_1 < 1
%   mu_2 - the curvature condition parameter; mu_1 < mu_2 < 1
%   alpha_init - the initial step length
%   alpha_max - the maximum allowable step length
%   max_iter = the maximum number of iterations allowed
%   obj0 (optional) = function value at xk
%   grad0 (optional) = gradient at xk (must be given if obj0 is given)
% Outputs:
%   alpha - a step satisfying the strong-Wolfe conditions
%   fnew - function value at xk + alpha*pk
%   feval, dfeval - number of function and gradient evaluations
%--------------------------------------------------------------------------

% perform some sanity checks
assert(size(xk,1) + size(lambda,1) == size(pk,1),...
    'check that xk+lambda and pk are the same dimensions');
assert(size(xk,2) == 1,...
    'xk must be given as a column vector');
assert(mu_1 > 0 && mu_1 <= mu_2 && mu_2 <= 1,...
    'check that mu_1 and mu_2 are valid');
assert(alpha_init > 0 && alpha_init < alpha_max,...
    'check that alpha_init and alpha_max are valid');
assert(max_iter > 0,...
    'max_iter must be positive');

% initialize some data before beginning bracketing stage
feval = 0;
dfeval = 0;

alpha = 0; 
[phi0, grad0] = merit_fun(alpha);
dphi0 = grad0'*pk;
    
fnew = phi0;
alpha_old = 0.0;
phi_old = phi0;
dphi_old = dphi0;

% check search direction
if (dphi0 > 0)
    error('Error in linesearch: search direction is not a descent direction');
end

alpha = alpha_init;
for i = 1:max_iter
    if (alpha > alpha_max)
        error('Error in linesearch: alpha > alpha_max');
    end
    [phi, ~] = merit_fun(alpha);
    feval = feval + 1;
    
    % check if new step violates the sufficient decrease condition, or
    % (when i > 2) if new phi is greater than old phi.  If so, call zoom
    if ( (phi > phi0 + mu_1*alpha*dphi0) || ...
	    ( (i > 1) && (phi > phi_old) ) )
        str = sprintf('linesearch switching to zoom (1): iteration %d\n',i);
        disp(str);
        [alpha, fnew] = zoom(alpha_old, alpha, phi_old, phi, dphi_old);
        return;
    end
    
    % get new derivative and check curvature condition
    [~, grad] = merit_fun(alpha);
    dphi = grad'*pk;
    dfeval = dfeval + 1;
    if ( abs(dphi) <= -mu_2*dphi0 )
        % curvature condition is satisfied;
        fnew = phi;
        return;
    end
    
    if (dphi > 0)
        % if we get here, the curvature condition is not satisfied, but 
        % derivative is positive and phi_new < phi_old
        str = sprintf('linesearch switching to zoom (2): iteration %d\n',i);
        disp(str);
        [alpha, fnew] = zoom(alpha, alpha_old, phi, phi_old, dphi);
        return;
    end
    
    % update the old variables
    alpha_old = alpha;
    phi_old = phi;
    dphi_old = dphi;
    
    % increase alpha; this can be changed if necessary
    alpha = 2*alpha;
end

% if we get here...
error('Error in lineSearch: exceeded maximum number of iterations');

%--------------------------------------------------------------------------
    function [alpha, phi] = zoom(alpha_low, alpha_hi, phi_low, phi_hi, dphi_low)
        % Purpose: finds a step that satisfies the strong Wolfe conditions
        % Inputs:
        %   alpha_low, alpha_hi - steps that satisfy phi(alpha_low) < phi(alpha_hi)
        %   phi_low, phi_hi - function values at alpha_low, alpha_hi
        %   dphi_low - derivative at alpha_low
        % Outputs:
        %   alpha - step that satisfies strong Wolfe conditions
        %   phi - function value at alpha 
        if (alpha_low == alpha_hi) && (alpha_low == alpha_max);
            alpha = alpha_max;
            [phi, ~] = merit_fun(alpha)
            return
        end
        
        for j = i:max_iter
            str = sprintf('iter %i : zoom interval = [%13.10f,%13.10f]',j,alpha_low,alpha_hi);
            disp(str);
            
            % use interpolation to get new step, then find the new function value    
            alpha = quadraticStep(alpha_low, alpha_hi, phi_low, phi_hi, dphi_low);
            [phi, ~] = merit_fun(alpha);
            feval = feval + 1;
        
            % check if phi_new violates the sufficient decrease condition, 
            % or if phi > phi_low; if so, this step gives the new alpha_hi value
            if ( (phi > phi0 + mu_1*alpha*dphi0) && (phi >= phi_low) )
                alpha_hi = alpha;
                phi_hi = phi;
            else
                % the sufficent decrease is satisfied and phi < phi_low
                [~, grad] = merit_fun(alpha);
                dphi = grad'*pk;
                dfeval = dfeval + 1;
                if ( abs(dphi) <= -mu_2*dphi0 )
                    % curvature condition has been satisfied, so stop
                    return;
                elseif ( dphi*(alpha_hi - alpha_low) >= 0 )
                    % in this case, alpha_low and alpha bracket a minimum
                    alpha_hi = alpha_low;
                    phi_hi = phi_low;
                end;
                % the new low step is alpha_new
                alpha_low = alpha;
                phi_low = phi;
                dphi_low = dphi;
            end
        end
        
        if (phi < phi0 + mu_1*alpha*dphi0);
            print('WARNING: Step found but curvature condition not met.');
            return;
        end
        
        % if we get here, the maximum number of iterations was exceeded
        error('Error in zoom: exceeded maxiter');
    end
end
%==========================================================================
function [alpha] = quadraticStep(alpha_low, alpha_hi, f_low, f_hi, df_low)
% Purpose: finds step between alpha_low and alpha_hi using quad interp.
% Input:
%   alpha_low - the evaulation point at the low function value
%   alpha_hi - the evaluation point at the hi function value
%   f_low, f_hi - the function values corresponding to alpha_low, alpha_hi
%   df_low - the derivative values corresponding to alpha_low
% Output:
%   alpha - the new step length
%--------------------------------------------------------------------------
alpha = alpha_low - 0.5*df_low*( (alpha_hi - alpha_low)/((f_hi - f_low)...
    /(alpha_hi - alpha_low) - df_low) );
if ( (alpha < min(alpha_low,alpha_hi)) || ...
        (alpha > max(alpha_low,alpha_hi)) )
    % check that step is between alpha_low and alpha_hi; if not error
    error('Error in quadraticStep: step out of range');
end
end

