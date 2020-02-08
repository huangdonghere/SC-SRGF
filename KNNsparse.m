%-------------------------------------------------------------------------|
% This function constructs the KNN similarity graph. If you find it useful|
% in your research, please cite the paper below.                          | 
%                                                                         |
% Xiaosha Cai, Dong Huang, Chang-Dong Wang, and Chee-Keong Kwoh.          |
% Spectral Clustering by Subspace Randomization and Graph Fusion for      |
% High-Dimensional Data. In Proc. of the 24th Pacific-Asia Conference on  |
% Knowledge Discovery and Data Mining (PAKDD), 2019.                      |
% ------------------------------------------------------------------------|

function output = KNNsparse(data1,data2,K)

N1 = size(data1,1);
N2 = size(data2,1);

% Distance matrix
D = pdist2(data1,data2);

sigma = mean(mean(D));

if data1==data2
    for i = 1:N1
        D(i,i)=1e100;
    end
end

dump = zeros(N1,K);
idx = dump;
for i = 1:K
    [dump(:,i),idx(:,i)] = min(D,[],2);
    temp = (idx(:,i)-1)*N1+[1:N1]';
    D(temp) = 1e100; 
end

dump = exp(-dump/(2*sigma));
Gsdx = dump;
Gidx = repmat([1:N1]',1,K);
Gjdx = idx;
output = sparse(Gidx(:),Gjdx(:),Gsdx(:),N1,N2);