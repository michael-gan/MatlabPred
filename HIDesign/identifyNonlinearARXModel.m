function syse = identifyNonlinearARXModel(Mmot,w,Q,Ts,N)
%identifyNonlinearARXModel ʶ������Ե��Իع��̬����ģ�Ͷ���˫����(w, Q), �����(Mmot)����
% ���룺
%  w: ת��
%  Q: Һ��������
%  Mmot: ���ת��
%  N: ʹ�������ĸ���
% �����
%  syse: ʶ�𵽵�ģ��
%
% ������ʹ������ϵͳʶ�𹤾����е�NLARX������

sys = idnlarx([2 2 1 0 1],'','CustomRegressors',{'u1(t-2)^2','u1(t)*u2(t-2)','u2(t)^2'});
data = iddata(Mmot,[w Q],Ts);
opt = nlarxOptions;
opt.Focus = 'simulation';
opt.SearchOptions.MaxIterations = 500;
syse = nlarx(data(1:N),sys,opt);
end