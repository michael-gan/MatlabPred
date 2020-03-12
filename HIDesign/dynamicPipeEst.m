function [x3, x4, x5, Qest] = dynamicPipeEst(dp, Q, I)
% �÷����������ƶ�̬�ܵ����̣����ݷ����þ�����ʽ����������ڲ�ͬ��ת���趨�µķ��̲���
% I: ת�������趨������

Q = Q(I);
dp = dp(I);
R1 = [0; Q(1:end-1)];
R2 = dp; R2(R2<0) = 0; R2 = sqrt(R2);
R = [ones(size(R2)), R2, R1];

% �Ƴ�δ�����ڶ��������ڵ�����
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