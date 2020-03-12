function [x1, x2, dpest] = staticPumpEst(w, dp, I)
% 该方法用来估计静态泵方程在不同泵转速设定下的方程参数
% I: 转速区间设定的索引

w1 = [0; w(I)];
dp1 = [0; dp(I)];
R1 = [w1.^2 w1];
x = pinv(R1)*dp1;
x1 = x(1);  
x2 = x(2);  

dpest = R1(2:end,:)*x;
end