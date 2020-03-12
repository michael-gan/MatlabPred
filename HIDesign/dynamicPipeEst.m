function [x3, x4, x5, Qest] = dynamicPipeEst(dp, Q, I)
% 该方法用来估计动态管道方程（根据方程用矩阵形式反算参数）在不同泵转速设定下的方程参数
% I: 转速区间设定的索引

Q = Q(I);
dp = dp(I);
R1 = [0; Q(1:end-1)];
R2 = dp; R2(R2<0) = 0; R2 = sqrt(R2);
R = [ones(size(R2)), R2, R1];

% 移除未运行在定义区间内的样本
ii = find(I);
j = find(diff(ii)~=1);
R = R(2:end,:); R(j,:) = [];
y = Q(2:end); y(j) = [];
x = R\y;

x3 = x(1);
x4 = x(2);
x5 = x(3);

Qest = R*x;
end