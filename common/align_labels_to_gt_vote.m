% function aligned = align_labels_to_gt_vote(pred, gt)
%     [~,~,gt_ids]   = unique(gt(:));
%     [~,~,pred_ids] = unique(pred(:));
% 
%     Kp = max(pred_ids);
%     map = zeros(Kp,1);
%     for j = 1:Kp
%         gj = gt_ids(pred_ids==j);
%         map(j) = mode(gj);  % 多数投票
%     end
%     aligned = reshape(arrayfun(@(x) map(x), pred_ids), size(pred));
% end
function aligned = align_labels_vote_no_merge(pred, gt)

    gt = gt(:);
    pred = pred(:);

    [pred_ids,~,pred_idx] = unique(pred);
    [gt_ids,~,gt_idx]     = unique(gt);

    Kp = numel(pred_ids);
    map = zeros(Kp,1);

    used_gt = false(numel(gt_ids),1);

    for j = 1:Kp
        gj = gt_idx(pred_idx == j);
        cnt = histcounts(gj, 1:numel(gt_ids)+1);
        cnt(used_gt) = -inf;      % 禁止重复使用
        [~, id] = max(cnt);
        map(j) = gt_ids(id);
        used_gt(id) = true;
    end

    aligned = map(pred_idx);
    aligned = reshape(aligned, size(pred));
end