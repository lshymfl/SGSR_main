function [accuracy, map_pred2true, y_pred_aligned] = compute_accuracy(NcutDiscrete, label)
% Robust accuracy for many clusters (K > 100) using Hungarian assignment.
% Handles rectangular confusion matrices, unmatched clusters, and corner cases.
%
% Inputs:
%   NcutDiscrete : [m x K_pred] indicator/soft matrix
%   label        : [m x 1] ground-truth labels (any integer set)
%
% Outputs:
%   accuracy        : scalar in [0,1]
%   map_pred2true   : length-K_pred vector, value in 0..K_true (0=unassigned)
%   y_pred_aligned  : [m x 1] labels mapped back to original label space

% ---- sanitize inputs ----
if size(label,2) > 1, label = label(:); end
[m, k] = size(NcutDiscrete);
if length(label) ~= m
    error('Label length (%d) must match number of rows in NcutDiscrete (%d).', length(label), m);
end
if k == 0
    accuracy = 0; map_pred2true = []; y_pred_aligned = zeros(m,1);
    return;
end

% ---- predicted labels from NcutDiscrete ----
% If a row is all zero/NaN, put it to cluster 1 to avoid empty argmax.
Nc = NcutDiscrete;
Nc(~isfinite(Nc)) = -inf;
rowAllNegInf = all(~isfinite(Nc),2);
Nc(rowAllNegInf,1) = 0;
[~, y_pred] = max(Nc, [], 2);
K_pred = max(y_pred);

% ---- compress ground-truth labels to 1..K_true ----
[uniq_true, ~, y_true] = unique(label, 'stable');
K_true = max(y_true);

% ---- confusion matrix C (K_pred x K_true) ----
C = accumarray([y_pred, y_true], 1, [K_pred, K_true]);

% ---- build cost matrix for Hungarian (maximize C -> minimize cost) ----
if isempty(C)
    accuracy = 0;
    map_pred2true = zeros(K_pred,1);
    y_pred_aligned = uniq_true(ones(m,1));
    return;
end
maxC = max(C(:));
if ~isfinite(maxC), maxC = 0; end
costMat = maxC - C;                 % non-negative
costMat(~isfinite(costMat)) = max(1e6, 1 + max(costMat(:)));

% ---- Hungarian assignment (robust) ----
assign = munkres_robust(costMat);   % length = K_pred, assign(i) in {0..K_true}

% ---- compute accuracy safely ----
matched = find(assign > 0);
if isempty(matched)
    % no matches: fall back to majority mapping (each pred -> global majority class)
    [~, majTrue] = max(sum(C,1));               % index in 1..K_true
    map_pred2true = zeros(K_pred,1);
    if ~isempty(majTrue), map_pred2true(:) = majTrue; end
    correct = sum(C(:,majTrue));
else
    picked_cols = assign(matched);
    map_pred2true = zeros(K_pred,1);
    map_pred2true(matched) = picked_cols;
    correct = sum(C(sub2ind(size(C), matched, picked_cols)));
end
accuracy = correct / m;

% ---- aligned predicted labels in original label space ----
% map_pred2true(i)=0 means unassigned -> send to global majority true class (or 1)
if any(map_pred2true==0)
    if exist('majTrue','var') && ~isempty(majTrue)
        fillCol = majTrue;
    else
        [~, fillCol] = max(sum(C,1));
        if isempty(fillCol), fillCol = 1; end
    end
    mp = map_pred2true; mp(mp==0) = fillCol;
else
    mp = map_pred2true;
end
idx = mp(y_pred);                % guaranteed >=1 now
idx(~(idx>=1 & idx<=K_true)) = 1;
y_pred_aligned = uniq_true(idx);
end


% ==========================================================
% ROBUST HUNGARIAN (supports rectangular, avoids 0-indexing)
% Returns assignment (length r) with values in 0..c.
% ==========================================================
function assignment = munkres_robust(costMat)
[r,c] = size(costMat);
n = max(r,c);
C = inf(n); C(1:r,1:c) = costMat;
C(~isfinite(C)) = 1e12;  % large cost for invalid

% Step 1: subtract row minima
for i = 1:n
    v = min(C(i,:));
    if isfinite(v), C(i,:) = C(i,:) - v; end
end
% Step 2: subtract column minima
for j = 1:n
    v = min(C(:,j));
    if isfinite(v), C(:,j) = C(:,j) - v; end
end

mask = zeros(n);            % 0=none,1=star,2=prime
rowCover = false(n,1);
colCover = false(n,1);

% Step 3: star independent zeros
for i = 1:n
    for j = 1:n
        if C(i,j)==0 && ~rowCover(i) && ~colCover(j)
            mask(i,j)=1; rowCover(i)=true; colCover(j)=true;
        end
    end
end
rowCover(:)=false; colCover(:)=false;

% Cover columns with starred zeros
for j = 1:n
    if any(mask(:,j)==1), colCover(j)=true; end
end

% Main loop
while sum(colCover) < n
    [r0,c0,found] = findZero(C,rowCover,colCover);
    if ~found
        % Step 6: adjust matrix
        a = min(C(~rowCover, ~colCover), [], 'all');
        if ~isfinite(a) || isempty(a)
            break; % no more progress possible
        end
        C(~rowCover, ~colCover) = C(~rowCover, ~colCover) - a;
        C(rowCover, colCover)   = C(rowCover, colCover) + a;
        continue;
    end
    mask(r0,c0) = 2; % prime
    sc = find(mask(r0,:)==1, 1);
    if ~isempty(sc)
        % cover row r0 and uncover the column with the star
        rowCover(r0)=true; colCover(sc)=false;
    else
        % augment path
        [mask, rowCover, colCover] = augmentPath(mask, r0, c0, rowCover, colCover);
        % re-cover columns with stars
        colCover(:)=false;
        for j = 1:n
            if any(mask(:,j)==1), colCover(j)=true; end
        end
        rowCover(:)=false;
    end
end

% Build assignment for original r x c
assignment = zeros(r,1);
for i = 1:r
    j = find(mask(i,1:c)==1, 1);
    if ~isempty(j), assignment(i)=j; else, assignment(i)=0; end
end
end

function [r0, c0, found] = findZero(C, rowCover, colCover)
n = size(C,1);
found = false; r0 = 1; c0 = 1; % init to valid indices to avoid 0-indexing
for i = 1:n
    if ~rowCover(i)
        j = find(C(i,:)==0 & ~colCover', 1);
        if ~isempty(j)
            r0=i; c0=j; found=true; return;
        end
    end
end
end

function [mask, rowCover, colCover] = augmentPath(mask, r0, c0, rowCover, colCover)
% Alternating path starting from primed zero (r0,c0).
path = [r0, c0];
done = false;
while ~done
    r = find(mask(:, path(end,2)) == 1, 1); % star in the column
    if isempty(r)
        done = true;
    else
        path(end+1,:) = [r, path(end,2)];
        c = find(mask(path(end,1), :) == 2, 1); % prime in the row
        if isempty(c)
            % If no prime is found (rare), stop the augmentation to avoid empty index.
            done = true;
        else
            path(end+1,:) = [path(end,1), c];
        end
    end
end

% Flip along path
for t = 1:size(path,1)
    i = path(t,1); j = path(t,2);
    if mask(i,j)==1
        mask(i,j)=0;
    else
        mask(i,j)=1;
    end
end
% Clear all primes
mask(mask==2) = 0;
% reset covers
rowCover(:)=false; colCover(:)=false;
end
