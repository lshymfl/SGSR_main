function [D,E,C,time,chgC_hist] = GTSC_PAMcutSSC( ...
    X,beta,lambda,alph,rho,s,tol,max_iter)
% function [D,E,C,time] = GTSC_PAMcutSSC(X,beta,lambda,alph,rho,s,tol,max_iter)
%Solve the Joint Sparse Subspace Clustering by PAM algorithm
%
% min_{B,E,C} \frac{1}{2}\|X-XC-E\|_{F}^{2}+\lambda\|E\|_{2,1}~\\
%s.t. B=C,~~~\|B\|_{0,2}\le s,\\
%&C^{\top}\mathbf{1}=\mathbf{1},~\textrm{diag}(C)=\mathbf{0}.,
% 
% ---------------------------------------------
%
% initialization
obj=zeros(max_iter,1);
MRes1=zeros(max_iter,1);
CRes1=zeros(max_iter,1);
Re =zeros(max_iter,1);
[d,N] = size(X);
affine = 0;
alpha = 20;
chgC_hist = zeros(max_iter,1);  

%%
R = -eye(N) + diag(ones(N-1, 1), -1);
R(:, N) = [];
RRT = R*R';

%%
tic
CMat = admmOutlier_mat_func(X, affine, alpha); 
%size(CMat)
C_TSC = CMat(1:N,:);
E = zeros(d, N);
C = C_TSC;
D =C_TSC*R;


t0  = tic;

for iter = 1 : max_iter
    Dk = D;
    Ek = E;
    Ck = C;
      
    %% update D by proj_l02
    P=(beta/(beta+rho))*Ck*R+(rho/(beta+rho))*Dk;
    D=proj_l02(P,s);
    %% update E
    theta = lambda / (1 + rho);                    
    U = (X - X*C + rho * E) / (1 + rho);            
    col_norm = sqrt(sum(U.^2, 1));                  
    E = U;                                         
    E(:, col_norm <= sqrt(2*theta)) = 0;  

    %% update C
    Txx    = find( sum(abs(X),2)>0 ); 
    XX   = X(Txx,:)'*X(Txx,:);
    A1 = XX+rho*eye(N);
    A2 = beta*RRT;
    C = XX-X(Txx,:)'*E(Txx,:)+beta*D*R'+rho*Ck;
    C = lyap(A1,A2,-C);
    C = C - diag(diag(C));
    %% check convergence
    chgC    = norm(Ck(:)-C(:),"fro")/norm(Ck(:),"fro");
    chgC_hist(iter) = chgC;
  
    if iter > 10
        if chgC < tol
            break;
        end
    end
    chgC_hist = chgC_hist(1:iter);

    %%
    E02=0;
    for e=1:size(E,2)
        E02=E02+norm(E(:,e),2);
    end
    obj(iter)=norm(X-X*C-E,'fro')^2+lambda*E02; 
    if (iter-1)==0
        Re(iter)=obj(iter);
        MRes1(iter)=norm(D,'fro');
        CRes1(iter)=norm(C,'fro');
    else
        Re(iter)   =abs(obj(iter)-obj(iter-1));
        MRes1(iter) =norm(D-Dk,'fro');
        CRes1(iter) =norm(C-Ck,'fro');
    end
    fprintf('%4d  %5.2e  %5.2e %5.2e %5.2e %5.2e %5.2e\n',iter,obj(iter),Re(iter),MRes1(iter),CRes1(iter),norm(E-Ek,'fro'),chgC);

%     disp(['iter ' num2str(iter) ',chgC=' num2str(chgC,'%2.1e')])
    time=toc(t0);
end

end