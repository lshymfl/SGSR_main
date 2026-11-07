function [D,Z,err] = TSC_ADMM(X,paras)
% This function solves the optimization problem in TSC.
% X: temporal data
% Z: coding matrix
% Author: Sheng Li (shengli@ece.neu.edu)
% Date: Feb. 11, 2015

%%%%--Initialization--%%%%%
tol = paras.tol;
maxIter = paras.maxIter;
[d, n_x] = size(X);
rho = paras.stepsize;
dsize = paras.n_d;
lambda1 = paras.lambda1;
lambda2 = paras.lambda2;

D = rand(d, dsize);
Z = rand(dsize, n_x);
V = zeros(dsize, n_x);

Y1 = zeros(d, dsize);
Y2 = zeros(dsize, n_x);

alpha = 1e-1;
beta = 1e-1;

%%--Define the weight matrix--%%
k = paras.ksize;
k2 = (k-1)/2;
W = zeros(n_x, n_x);
for i = 1:n_x
    W(i, max(1,i-k2):min(n_x,i+k2)) = 1;
end
for i = 1:size(W,1)
    W(i,i) = 0;
end

%%%%--Start main loop--%%%%
disp('Optimization...');

% func_vals = zeros(maxIter,1);
err = zeros(maxIter,1);
for loop = 1:maxIter
    
    %%--Construct Laplacian matrix--%%
    DD = diag(sum(W));
    L = DD - W;
    
    f_old = norm(X-D*Z, 'fro');
    
    %%--Update U--%%
    U = (X*V' - Y1 + alpha*D) / (V*V' + alpha*eye(dsize));
    
    %%--Update V--%%
    kron_Xt_X = kron(speye(n_x,n_x), (U' * U));
    kron_R_Rt = kron(L, speye(dsize,dsize));
    left = kron_Xt_X  + kron(speye(n_x,n_x),(lambda1+beta)*speye(dsize,dsize)) + lambda2*kron_R_Rt;
    right = U'*X - Y2 + beta*Z;
    right = reshape(right,dsize*n_x,1);
    Vtmp = left \ right;
    V = reshape(Vtmp, dsize, n_x);
    
    %%--Update D--%%
    D = U + Y1 / alpha;
    D(D<0) = 0;
    for kk = 1:size(D,2)
        D(:,kk) = D(:,kk) ./ norm(D(:,kk));
    end
    
    %%--Update Z--%%
    Z = V + Y2 / beta;
    Z(Z<0) = 0;
    
    f_new = norm(X-D*Z, 'fro');
%     func_vals(k) = 0.5*norm(X - D*Z, 'fro')^2 + lambda1*norm(Z,'fro')+lambda2*trace(Z*L*Z') 
    if (abs(f_new - f_old)/max(1,abs(f_old)) < tol) && (nnz(D)>0)
        break;
    else
             
        err(loop) = abs(f_new - f_old)/max(1,abs(f_old));
        Y1 = Y1 + rho*alpha*(U - D);
        Y2 = Y2 + rho*beta*(V - Z);
    end
end