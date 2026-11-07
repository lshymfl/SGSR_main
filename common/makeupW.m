function W = makeupW (X,t)
[~,n] = size(X);
for i=1 : n-1
    W(i) = exp(-norm(X(:,i+1)-X(:,i))^2/t);
end
