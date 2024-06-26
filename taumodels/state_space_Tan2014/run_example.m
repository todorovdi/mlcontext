%% State-space modeling of human reach adaptation (EEG_Sign)
% Lucas Struber - 2021

% Loading files and constant initialization
clear all, clc;
load('example_behavior') % 80 trials x 3 subjects

% Parameters initialization
minNbTrials = 3; % minimum number of "retained" trials
maxNbTrials = 40; % maxumum number of "retained" trials
minStartIdx = 6; % minimum index of first trial where previous trials are taken into account
maxStartIdx = 6; % maximum index of first trial where previous trials are taken into account
a0 = 0.95; % initial retention factor
b0 = 0.01; % initial adaptation factor (for trials before startIdx)
x00 = 0; % initial initial state
c0 = 0.001; % initial c coefficient (that is multiplied by the ratio of mean error and error variance)
amin = 0.5; % minimum retention factor
amax = 1; % maximum retention factor
cmin = 0; % minimum c
cmax = 0.03; % maximum c
bmin = 0; % minimum adaptation rate
bmax = 0.5; % maximum adaptation rate
x0min = -30; % minimum initial state
x0max = 30; % maximum initial state

% optimization options
opts = optimoptions('fmincon','Algorithm','sqp','ConstraintTolerance',1e-10,'MaxFunctionEvaluations',3000,'MaxIterations',3000,'StepTolerance',1e-10);
nbOptParams = 6; % A, c, b_start, nbT, x0 and error (eventually add startIdx)

% init fit outputs
nbSub = size(measured_error,2);
nbTrials = size(measured_error,1);

output_Tan = zeros(nbTrials,nbSub);
params_Tan = zeros(6,nbSub); % 3 parameters, A, c, b_start, x0, nbTrials, startIdx
adaptationRate_Tan = zeros(nbTrials,nbSub);
perf_Tan = zeros(2,nbSub); % 2 performances indicators, AICc & MSE


% loop on the number of subjects
for s = 1:nbSub
    error = smooth(measured_error(:,s),2); % interpolate NaNs
    
    % with least square minimization
    minfval = Inf;
    idx = 0;
    fval_all = zeros(1,length(minNbTrials:maxNbTrials));
    for nbT = minNbTrials:maxNbTrials
        idx = idx +1;
        for startIdx = minStartIdx:maxStartIdx 

            f = @(Z)MSE_TanSS(Z,nbT,startIdx,'NoAdapt',perturb,error); 
            [Z_est, fval] = fmincon(f,[a0 c0 b0 x00],[],[],[],[],[amin bmin cmin x0min],[amax cmax bmax x0max],[],opts);

            fval_all(idx) = fval;

            if fval < minfval
                minfval = fval;
                Z = Z_est;
                nbT_opt = nbT;
                startIdx_opt = startIdx;
            end
        end
    end
%         FigName = strcat('NbTrialsOpt - Condition :',CondNames{c},' - Sub : ', num2str(s));
%         figure('Name',FigName);
%         plot(MinNbTrials:MaxNbTrials,fval_all);

    params_Tan(:,s) = [Z, nbT_opt startIdx_opt];
    
    [output_Tan(:,s), adaptationRate_Tan(:,s)] = ErrorModel_TanSS(params_Tan(1,s),params_Tan(2,s),params_Tan(3,s),params_Tan(4,s),params_Tan(5,s),params_Tan(6,s),'NoAdapt',perturb,error);   
    
    sigma2 = var(output_Tan(:,s) - error);
    likelihood = -1/(2*sigma2)*fval-(length(error)/2)*log(sigma2)-(length(error)/2)*log(2*pi);
    AIC = 2*nbOptParams - 2*likelihood + (2*nbOptParams*(nbOptParams+1))/(length(error)-nbOptParams-1);
    
    perf_Tan(:,s) = [fval, AIC];
    
    figure();
    hold on;
    title(['Subject ' num2str(s)])
    yyaxis left
    plot(error, 'LineWidth',1);
    plot(output_Tan(:,s), 'LineWidth',2);
    ylabel('Angle (°)');
    yyaxis right
    plot(adaptationRate_Tan(:,s), 'LineWidth',2);
    legend('Observed error','Model output', 'Adaptation rate');
    hold off;
    xlabel('Trials');
end