function D = proj_l02(Q, t)
    [M, ~] = size(Q);
    D = zeros(size(Q));
    col_norms = sqrt(sum(Q.^2, 1));
    [~, sorted_indices] = sort(col_norms, 'descend');
    active_cols_indices = sorted_indices(1:min(t, M));
    D(:, active_cols_indices) = Q(:, active_cols_indices);
end