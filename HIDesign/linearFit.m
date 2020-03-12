function varargout = linearFit(Form, Data)
%linearFit Linear least squares solution for Pump Head and Torque parameters.
%
% If Form==0, accept separate inputs and return separate outputs. For one experiment only.
% If Form==1, accept an ensemble and return compact parameter vectors. For several experiments (ensemble).
if Form==0
   w = Data{1};
   Q = Data{2};
   H = Data{3};
   M = Data{4};
   n = length(Q);
   if isscalar(w), w = w*ones(n,1); end
   Q = Q(:); H = H(:); M = M(:);
   Predictor = [w.^2, w.*Q, Q.^2];
   Theta1 = Predictor\H;
   hnn =  Theta1(1);
   hnv = -Theta1(2);
   hvv = -Theta1(3);
   Theta2 = Predictor\M;
   k0 =  Theta2(2);
   k1 = -Theta2(3);
   k2 =  Theta2(1);
   varargout = {hnn, hnv, hvv, k0, k1, k2};
else
   H = cellfun(@(x)x.Head,Data,'uni',0);
   Q = cellfun(@(x)x.Discharge,Data,'uni',0);
   M = cellfun(@(x)x.Torque,Data,'uni',0);
   W = cellfun(@(x)x.Speed,Data,'uni',0);
   N = numel(H);

   Theta1 = zeros(3,N);
   Theta2 = zeros(3,N);
   
   for kexp = 1:N
      Predictor = [W{kexp}.^2, W{kexp}.*Q{kexp}, Q{kexp}.^2];
      X1 = Predictor\H{kexp};
      hnn =  X1(1);
      hnv = -X1(2);
      hvv = -X1(3);
      X2 = Predictor\M{kexp};
      k0 =  X2(2);
      k1 = -X2(3);
      k2 =  X2(1);
      
      Theta1(:,kexp) = [hnn; hnv; hvv];
      Theta2(:,kexp) = [k0; k1; k2];
   end
   varargout = {Theta1', Theta2'};
end
end