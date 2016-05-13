function [V, H, lin_depend] = ModGramSchmidt(i, V, H)
% orthogonalizes V(:,i) with respect to V(:,1:i-1) 
% inputs:
%  i - index of vector to orthogonalize
%  V - set of vectors
%  H - upper Hessenberg matrix
%--------------------------------------------------------------------------
reorth = 0.98;

% get the norm of the vector being orthogonalized, and find threshold for
% re-orthogonalization
nrm = V(:,i)'*V(:,i);
thr = nrm*reorth;
if (nrm <= 0.0)
    error('Error in ModGramSchmidt: inner_prod(V,V) < 0.0');
end;
if (nrm ~= nrm)
    error('Error in ModGramSchmidt: norm(V) = nan');
end;

if (i == 1)
    % just normalize and return
    V(:,i) = V(:,i)/sqrt(nrm);
    return;
end;

% Begin main Gram-Schmidt loop
for k = 1:i-1
    prod = V(:,i)'*V(:,k);
    H(k,i-1) = prod;
    V(:,i) = V(:,i) - prod.*V(:,k);
    % check if reorthogonalization is necessary
    if (prod*prod > thr)
        prod = V(:,i)'*V(:,k);
        H(k,i-1) = H(k,i-1) + prod;
        V(:,i) = V(:,i) - prod.*V(:,k);
    end;
    % update the norm and check its size
    nrm = nrm - H(k,i-1)*H(k,i-1);
    if (nrm < 0.0)
        nrm = 0.0;
    end;
    thr = nrm*reorth;
end;

% test the resulting vector
nrm = norm(V(:,i),2);
H(i,i-1) = nrm;
lin_depend = false;
if (nrm <= 0.0)
    lin_depend = true;
    %error('Error in ModGramSchmidt: V(i) is linearly dependent');
else
    V(:,i) = V(:,i)./nrm;
end;
end