clc;
clear;
close all;
load('data/YB10.mat');
addpath('L1_ADMM/');
addpath('toolbox/');
addpath('common\');
addpath('algorithm\');
addpath('ncut\');
addpath('Codes\');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Data processing
X=fea;
data_num = size(X, 1);
class_num = length(unique(gnd));
label = gnd;
nCluster = length(unique(label));
X = NormalizeFea(X, 1);
times = struct();
%% run_lrr
lambda1=0.1;
tic;
[Z_lrr] = lrr_relaxed(X, lambda1);    
times.lrr = toc;
W_lrr = (abs(Z_lrr) + abs(Z_lrr)') / 2; % already symmetric affinity

Y_lrr= SpectralClustering1(W_lrr, nCluster);
%Evaluation lrr
accr_lrr = evalAccuracy(Y_lrr, label);
nmi_lrr = nmi_new(label,Y_lrr );

[~,~,gt_ids]   = unique(label);
imagesc(W_lrr);
set(gca, 'XTick', [], 'YTick', []);

aligned_lrr = align_labels_to_gt_vote(Y_lrr,gt_ids);
plotClusters(aligned_lrr);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
%% run_ssc
lambda2=0.1;
tic;
[ Z_ssc] = ssc_relaxed( X, lambda2);    
times.ssc = toc;
W_ssc = (abs(Z_ssc) + abs(Z_ssc)') / 2; % already symmetric affinity

Y_ssc= SpectralClustering1(W_ssc, nCluster);

%Evaluation ssc
accr_ssc = evalAccuracy(label, Y_ssc);
nmi_ssc = nmi_new(Y_ssc, label);

[~,~,gt_ids]   = unique(label);
imagesc(W_ssc);
set(gca, 'XTick', [], 'YTick', []);

aligned_ssc = align_labels_to_gt_vote(Y_ssc,gt_ids);
plotClusters(aligned_ssc);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
%% run_spatSC
lambda_3=0.01;lambda_4=150;
tic;
[ Z_spatSC] = spatsc_relaxed( X, lambda_3, lambda_4);
times.spatSC = toc;
W_spatSC = (abs(Z_spatSC) + abs(Z_spatSC)') / 2; % already symmetric affinity

Y_spatSC= SpectralClustering1(W_spatSC, nCluster);
%Evaluation spatSC
accr_spatSC = evalAccuracy(label, Y_spatSC);
nmi_spatSC = nmi_new(Y_spatSC, label);

[~,~,gt_ids]   = unique(label);
imagesc(W_spatSC);
set(gca, 'XTick', [], 'YTick', []);

aligned_spatsc = align_labels_to_gt_vote(Y_spatSC,gt_ids);
plotClusters(aligned_spatsc);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
%% run_osc
lambda_5=0.1;lambda_6=0.1;   
tic;
[ Z_osc] = osc_relaxed( X, lambda_5, lambda_6);  
times.osc = toc;
W_osc = (abs(Z_osc) + abs(Z_osc)') / 2; % already symmetric affinity

Y_osc= SpectralClustering1(W_osc, nCluster);
%Evaluation osc
accr_osc = evalAccuracy(label, Y_osc);
nmi_osc = nmi_new(Y_osc, label);

[~,~,gt_ids]   = unique(label);
imagesc(W_osc);
set(gca, 'XTick', [], 'YTick', []);

aligned_osc = align_labels_to_gt_vote(Y_osc,gt_ids);
plotClusters(aligned_osc);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);

% run_tsc
paras = [];
% paras.lambda1 = 0.01;
% paras.lambda2 = 100;
paras.lambda1 = 10;
paras.lambda2 = 0.001;
paras.n_d = 80;
paras.ksize = 7;
paras.tol = 1e-4;
paras.maxIter = 200;
paras.stepsize = 0.1;
tic;
%%---Learn representations Z---%%%
[D, Z_tsc, err] = TSC_ADMM(X,paras);
times.tsc = toc;
vecNorm = sum(Z_tsc.^2);
W_tsc= (Z_tsc'*Z_tsc) ./ (vecNorm'*vecNorm + 1e-6);
Y_tsc= SpectralClustering1(W_tsc, nCluster);
% Evaluation tsc
accr_tsc = evalAccuracy(label, Y_tsc);
nmi_tsc = nmi_new(Y_tsc, label);

[~,~,gt_ids]   = unique(label);
imagesc(W_tsc);
set(gca, 'XTick', [], 'YTick', []);

aligned_tsc = align_labels_to_gt_vote(Y_tsc,gt_ids);
plotClusters(aligned_tsc);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
%% running FPEMS-WOSC  0.5  0.1
% H=1;
% acc=zeros(H,2);
% tim=zeros(H,1);
% for i=1:H
rho = 0.1;
t = 100;
[~,n] = size(X);
% V = X' + N;
V = X + zeros;
H1=makeupW(V,t);
H = diag(H1);
R = (triu(ones(n,n-1),1) - triu(ones(n, n-1))) + (triu(ones(n, n-1),-1)-triu(ones(n, n-1)));
% gamma=1.01*(norm(R*H)^2/mu+2*(norm(X')^2));
lambda7 = 0.5;
lambda8 = 0.5;
tic;
[Z_FPEMS_WOSC] = FPP_RoSeSC(X,H,lambda7,lambda8,rho);
times.FPEMS_WOSC = toc;
W_FPEMS_WOSC = (abs(Z_FPEMS_WOSC) + abs(Z_FPEMS_WOSC)') / 2;% already symmetric affinity

Y_FPEMS_WOSC= SpectralClustering1(W_FPEMS_WOSC, nCluster);

%Evaluation FPEMS_WOSC
accr_FPEMS_WOSC = evalAccuracy(label, Y_FPEMS_WOSC);
nmi_FPEMS_WOSC = nmi_new(Y_FPEMS_WOSC, label);

[~,~,gt_ids]   = unique(label);
imagesc(W_FPEMS_WOSC);
set(gca, 'XTick', [], 'YTick', []);

aligned_FPEMS = align_labels_to_gt_vote(Y_FPEMS_WOSC,gt_ids);
plotClusters(aligned_FPEMS);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
%% running tsc_sc
lambda9=1;lambda10=10;alpha=0.5;n_space=40;
tic;
[Z_tsc_sc,Q,iter,ziter]=TSCSS(X, lambda9,lambda10,alpha,n_space);
times.tsc_sc = toc;

W_tsc_sc = (abs(Z_tsc_sc) + abs(Z_tsc_sc)') / 2;% already symmetric affinity

Y_tsc_sc = SpectralClustering1(W_tsc_sc, nCluster);
%Evaluation tsc_sc
accr_tsc_sc = evalAccuracy(label, Y_tsc_sc);
nmi_tsc_sc = nmi_new(Y_tsc_sc, label);

[~,~,gt_ids]   = unique(label);
imagesc(W_tsc_sc);
set(gca, 'XTick', [], 'YTick', []);

aligned_tsc_sc = align_labels_to_gt_vote(Y_tsc_sc,gt_ids);
plotClusters(aligned_tsc_sc);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
%%  running AGCSC
alpha=0.01;beta=0.01;
tic;
[F, C, time_used] =  agcsc_x(X', alpha, beta);
times.AGCSC = toc;
W_AGCSC = (abs(C) + abs(C)') / 2;% already symmetric affinity

Y_AGCSC = SpectralClustering1(W_AGCSC, nCluster);
%Evaluation AGCSC
accr_AGCSC = evalAccuracy(label, Y_AGCSC);
nmi_AGCSC = nmi_new(Y_AGCSC, label);

[~,~,gt_ids]   = unique(label);
imagesc(W_AGCSC);
set(gca, 'XTick', [], 'YTick', []);

colormap('parula');
aligned_AGCSC = align_labels_to_gt_vote(Y_AGCSC,gt_ids);
plotClusters(aligned_AGCSC);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
%% JSSCSSC
% K=[4,5];
K_kk = 4;
beta_bb=0.025;
lambda_ll = 0.1;
alph = 0.0001;
rho_rr =0.01;
tol = 1e-6;
maxiter_ii=500;
tic;
%normalizeColumn = @(data) cnormalize_inplace(data);
% [B,E,C_JSSC,time_JSSC] = JSSC_PAMnoalpha(X',beta_bb,lambda_ll,0,rho_rr,K_kk,tol,maxiter);
[BSSC,ESSC,C_JSSCSSC,time_JSSCSSC] = JSSC_PAMcutSSC(X,beta_bb,lambda_ll,0,rho_rr,K_kk,tol,maxiter_ii);
times.JSSCSSC = toc;
%C_JSSC = C_JSSC(1:N,:);
% C_JSSCSSC = C_JSSCSSC(1:size(X,1),:);
W_JSSCSSC = (abs(C_JSSCSSC) + abs(C_JSSCSSC'))/2;   %%%  affinity matrix
Y_JSSCSSC = SpectralClustering1(W_JSSCSSC, nCluster);
%EvaluationJSSC
accr_JSSCSSC  = evalAccuracy(label, Y_JSSCSSC);
nmi_JSSCSSC  = nmi_new(Y_JSSCSSC, label);

[~,~,gt_ids]   = unique(label);
W = NormalizeFea(W_JSSCSSC, 1);
imagesc(W);
set(gca, 'XTick', [], 'YTick', []);

aligned_JSSCSSC = align_labels_to_gt_vote(Y_JSSCSSC,gt_ids);
plotClusters(aligned_JSSCSSC );
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
%% Temporal subspace clustering gradient l02norm GTSC
betaD = 0.01;
lambda = 0.0001;
rho = 0.01;  
T_ss = 4;
tol = 1e-4;
maxiter_ii=200;
tic;
% [BTSC,ETSC,C_GTSC,time_GTSC,chgC_hist] = ...
%     GTSC_PAMcutSSC(X,betaD,lambda,0,rho,T_ss,tol,maxiter_ii);

[BTSC,ETSC,C_GTSC,time_GTSC] =GTSC_PAMcutSSC(X,betaD,lambda,0,rho,T_ss,tol,maxiter_ii);
times.GTSC = toc;
% [BTSC,ETSC,C_GTSC,time_GTSC] = GSTSC_PAMcutSSC(X,betaC,betaD,lambda,0,rho,T_kk,T_ss,tol,maxiter_ii);

W_GTSC =(abs(C_GTSC) + abs(C_GTSC'))/2;   %%%  affinity matrix
Y_GTSC = SpectralClustering1(W_GTSC, nCluster);
%EvaluationJSSC
accr_GTSC  = evalAccuracy(label, Y_GTSC);
nmi_GTSC  = nmi_new(Y_GTSC, label);

[~,~,gt_ids]   = unique(label);
W = NormalizeFea(W_GTSC, 1);
imagesc(W);
movegui(imagesc(W), 'center');
set(gca, 'XTick', [], 'YTick', []);
                        
% colormap('parula');
aligned_GTSC = align_labels_to_gt_vote(Y_GTSC,gt_ids);
plotClusters(aligned_GTSC);
set(gca, 'XTick', [], 'YTick', []);
plotClusters(gt_ids);
set(gca, 'XTick', [], 'YTick', []);
% save('GTSC_convergence.mat','chgC_hist');
%% ========== 打印各方法运行时间 ==========
fprintf('\n========== Running Time Summary (dataset: %s) ==========\n', 'YALEB10');
method_names = {'lrr','ssc','spatSC','osc','tsc','FPEMS_WOSC','tsc_sc','AGCSC','JSSCSSC','GTSC'};
for i = 1:numel(method_names)
    name = method_names{i};
    if isfield(times, name)
        fprintf('%-12s : %8.4f s\n', name, times.(name));
    end
end
fprintf('=========================================================\n');
