function [ U,D,V,rankA] = RFSVT6( A, lambda, K, npoweriter, gamma,hfun_sg,L)
    [m, n] = size(A);
    vOmega = randn(m,n);
    maxrank = min(m,n);
    randidx = randi(maxrank-K+1);
    %%  Left Range Space
        Omega = vOmega(randidx:randidx+K-1,:);
        Y = A*Omega';
        Q= qr(Y,0);
    % Power iteration
%     for q = 1:npoweriter
% 
% %       [Q, ~] = qr(A.*(Q'.*A)',0);
%         [Q, ~] = qr(A.*(Q'.*A)',0);
%     end
    [U,D,V,rankA] = Shrinkage_by_Decomposition(Q, A, lambda, gamma, hfun_sg,L);
end

function [U,diagD,V,svp] = Shrinkage_by_Decomposition(Q, A, lambda, gamma, hfun_sg,L)
    [Qm,Qn] = size(Q);
    B  = Q'*A;
    [H, C] = qr(B',0);
    [Up, Dp, Vp] = mexPolarSVD(C);
    diagDp = diag(Dp);
    w     = hfun_sg(diagDp,gamma,lambda) ;
    diagD = diagDp - w/L ;
    svp   = length(find(diagD > 0));
    idSv  = Qn-svp+1;
    U     = Q*Vp;
    U     = U(:,idSv:Qn);
    V     = (H*(Up*(Vp)));
    V     = V(:,idSv:Qn);
    diagD = diagD(idSv:Qn);
end
