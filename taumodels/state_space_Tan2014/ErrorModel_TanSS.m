function [y_pred, B] = ErrorModel_TanSS(A,c,Bstart,X0,NbPreviousTrials,StartIdx,ModelBeforeSwitch,Perturb,Error)

% c is the coefficient in front of the variation
% last argument is long array
% X0 is init state
% A is retention
% NbPreviousTrials is number of trials taken to comput variablity (in Tan 20)
% Perturb, Error -- true experimental observations
% ModelBeforeSwitch either "NoAdapt" or something else (in this case learning rate IC 
% taken to be Bstart

y_pred = zeros(size(Error));
state = zeros(size(Error));
B = zeros(size(Error)); % dynamic learning rate

state(1) = X0;

% Calculate the first prediction error as the difference between Perturb(1) and X0
y_pred(1) = Perturb(1) - X0;
            
% Loop over the rest of the elements in Error, t is integer index
for t = 2:length(Error)
    if t < StartIdx
        if(ModelBeforeSwitch == "NoAdapt")
            B(t) = 0;
        else
            B(t) = Bstart;
        end
    else
        ErrorMean = mean(Error(max(1,t-NbPreviousTrials):t-1));
        ErrorVar = std(Error(max(1,t-NbPreviousTrials):t-1));
%         ErrorMean = mean(y_pred(max(1,t-NbPreviousTrials):t-1));
%         ErrorVar = std(y_pred(max(1,t-NbPreviousTrials):t-1));
        
        % this is key assumption of Tan model
        B(t) = c*(ErrorMean^2)/(ErrorVar^2);
        % Clip B(t) to the range [-0.5, 0.5]
        if(B(t) > 0.5), B(t) = 0.5; end
        if(B(t) < -0.5), B(t) = -0.5; end    
    end
    
    % Update the state as a function of A, state(t-1), B(t) and y_pred(t-1)
    state(t) = A*state(t-1) + B(t)*y_pred(t-1);
    % Calculate the prediction error as the difference between Perturb(t) and state(t)
    y_pred(t) = Perturb(t) - state(t);
end
end
