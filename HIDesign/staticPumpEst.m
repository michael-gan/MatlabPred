function [x1, x2, dpest] = staticPumpEst(w, dp, I)
% �÷����������ƾ�̬�÷����ڲ�ͬ��ת���趨�µķ��̲���
% I: ת�������趨������

w1 = [0; w(I)];
dp1 = [0; dp(I)];
R1 = [w1.^2 w1];
x = pinv(R1)*dp1;
x1 = x(1);  
x2 = x(2);  

dpest = R1(2:end,:)*x;
end