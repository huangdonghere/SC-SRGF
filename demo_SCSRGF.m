%-------------------------------------------------------------------------|
% This is a demo for the SC-SRGF algorithm. If you find it useful in your |
% research, please cite the paper below.                                  |
%                                                                         |
% Xiaosha Cai, Dong Huang, Chang-Dong Wang, and Chee-Keong Kwoh.          |
% Spectral Clustering by Subspace Randomization and Graph Fusion for      |
% High-Dimensional Data. In Proc. of the 24th Pacific-Asia Conference on  |
% Knowledge Discovery and Data Mining (PAKDD), 2019.                      |
% ------------------------------------------------------------------------|

function demo_SCSRGF()

%% Load the data
dataName = 'COIL20';
% dataName = 'MF';
load(['data_',dataName,'.mat'],'data');
gt = data(:,1);% Ground truth
fea = data(:,2:end);% Ddata features
clsNum = numel(unique(gt)); % Number of clusters

%% run the SC-SRGF algorithm
disp(['Running the SC-SRGF algorithm on the ',dataName,' dataset...']);
%% You can use the default parameters.
tic;
clsResult = SCSRGF_function(fea, clsNum);
toc;
%% Or customize the parameters.
% Knn = 5;    % Number of nearest neighbors in the KNN graph.
% m = 20;     % Number of similarity matrices.
% r = 0.5;    % Sampling ratio.
% clsResult = SCSRGF_function(fea, clsNum, Knn, m, r);

%% Compute NMI
nmiScore = NMImax(gt,clsResult);
disp(['NMI = ',num2str(nmiScore)]); 