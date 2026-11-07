function [lables]=Nclu(M,k)
%M is the a affinity matrix

D=diag(sum(M,2));
L=D-M;
[U,~,~]=svd(D^(-0.5)*L*D^(-0.5));
F=U(:,end-k+1:end);
Q=D^(-0.5)*F;
lables=kmeans(Q,k);
end