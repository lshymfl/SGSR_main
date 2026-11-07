function [ E ] = solve_l2l1( W, lambda )
%% Solves the following
%    RoSeSC  l21定义为行向量二范数的和
%   arg min_{x} || X - W ||_2^2 + lambda || X ||_2/1
%
% Created by Stephen Tierney
% stierney@csu.edu.au
%

[m, n] = size(W);

E = W;

for i = 1 : m
    
    norm_col = norm(W(i,:));
    
    if (norm_col > lambda)
        E(i,:) = (norm_col - lambda) * W(i,:) / norm_col;
    else
        E(i,:) = zeros(1, n);
    end
    
end

end