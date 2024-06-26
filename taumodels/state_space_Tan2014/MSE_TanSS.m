function [f, y_pred, B] = MSE_TanSS(Z,NbTrials,StartIdx,ModelBeforeSwitch,Perturb,Error)


%function [y_pred, B] = ErrorModel_TanSS(A,c,Bstart,X0,
%   NbPreviousTrials,StartIdx,ModelBeforeSwitch,Perturb,Error)

[y_pred, B] = ErrorModel_TanSS(Z(1),Z(2),Z(3),Z(4),NbTrials,StartIdx,ModelBeforeSwitch,Perturb,Error);
f = mean((y_pred - Error).^2);

end
