%-------------------------------------------------------------------------|
% This is a demo for the SC-SRGF algorithm. If you find it useful in your |
% research, please cite the paper below.                                  |
%                                                                         |
% Xiaosha Cai, Dong Huang, Chang-Dong Wang, and Chee-Keong Kwoh.          |
% Spectral Clustering by Subspace Randomization and Graph Fusion for      |
% High-Dimensional Data. In Proc. of the 24th Pacific-Asia Conference on  |
% Knowledge Discovery and Data Mining (PAKDD), 2019.                      |
% ------------------------------------------------------------------------|

function label = SCSRGF_function(fea, clsNum, Knn, m, r)
% Input:
% fea:      The data matrix. Each row is a data instance.
% clsNum:   The number of clusters.
% Knn:      The number of nearest neighbors in the KNN graph.
% m:        The number of similarity matrices.
% r:        The sampling ratio.

if nargin < 3
    Knn = 5;
end
if nargin < 4
    m = 20;
end
if nargin < 5
    r = 0.5;
end

%% Get similarity matrices
% Generate m random subspaces and calculate m similarity matrices
allMatrices = cell(1,m);
for i = 1:m
    rand('state',sum(100*clock)*rand(1));
    seq=randperm(size(fea,2)); % Random permutation
    subFea = fea(:,seq(1:int64(size(fea,2)*r)));
    allMatrices{i} = KNNsparse(subFea,subFea,Knn);
    allMatrices{i} = max(allMatrices{i},(allMatrices{i})');
end

%% Fuse the similarity matrices
newMatrix = SNF_sparse(allMatrices,Knn);

%% Get the final clustering result
label = SpectralClustering(newMatrix,clsNum);
