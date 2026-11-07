function X = proxl21(B,lambda)

X = zeros(size(B));
for i = 1 : size(X,1)
    nxi = norm(B(i,:));
    if nxi > lambda  
        X(i,:) = (1-lambda/nxi)*B(i,:);
    end
end