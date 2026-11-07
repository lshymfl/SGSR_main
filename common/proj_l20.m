function B = proj_l20(P, s)
    [~, N] = size(P);
    B = zeros(size(P));
    row_norms = sqrt(sum(P.^2, 2));
    [~, sorted_indices] = sort(row_norms, 'descend');
    active_rows_indices = sorted_indices(1:min(s, N));
    B(active_rows_indices, :) = P(active_rows_indices, :);
end