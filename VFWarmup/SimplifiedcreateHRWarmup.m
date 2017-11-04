function warmupPts= SimplifiedcreateHRWarmup(model,nPoints,verbFlag,bias,nPointsCheck)
% createHRWarmup Create a warmup point set for hit-and-run sampling by
% combining orthogonal and random points
%
% warmupPts= createHRWarmup(model,nPoints,verbFlag)
%
%INPUTS
% model     Model structure
%
%OPTIONAL INPUTS
% nPoints   Number of warmup points (Default = 5000);
% verbFlag  Verbose flag (Default = false)
% bias
%   method          Biasing distribution: 'uniform', 'normal'
%   index           The reaction indexes which to bias (nBias total)
%   param           nBias x 2 matrix of parameters (for uniform it's min
%   max, for normal it's mu, sigma).
%
%OUTPUT
% warmupPts Set of warmup points
%
% Markus Herrgard 4/21/06
%
% Richard Que (11/23/09) Integrated subfunctions into script.

if (nargin < 2)||isempty(nPoints), nPoints = 5000; end
if (nargin < 3)||isempty(verbFlag), verbFlag = false; end
if (nargin < 4), bias = []; end
if (nargin < 5)||isempty(nPointsCheck), nPointsCheck = true; end

if isfield(model,'A')
    [nMets,nRxns] = size(model.A);
else
    [nMets,nRxns] = size(model.S);
    model.A=model.S;
end
if ~isfield(model,'csense')
    model.csense(1:size(model.S,1)) = 'E';
end

if nPointsCheck && (nPoints < nRxns*2) 
    warning(['Need a minimum of ' num2str(nRxns*2) ' warmup points']);
    nPoints = nRxns*2;
end
warmupPts = sparse(nRxns,nPoints);


i = 1;
h = waitbar(0,'Creating warmup points ...');
%Generate the points
while i <= nPoints/2
    if mod(i,10) == 0
        waitbar(2*i/nPoints,h);
    end

    % Create random objective function
    model.c = rand(nRxns,1)-0.5;
    
    for maxMin = [1, -1]
        % Set the objective function
        if i <= nRxns%FVA unbiased
            model.c = zeros(nRxns,1);
            model.c(i) = 1;
        end
        model.osense = maxMin;
        
        % Determine the max or min for the rxn
        sol = solveCobraLP(model);
        x = sol.full;
        status = sol.stat;
        if status == 1
            validFlag = true;
        else
            continue
%             display ('invalid solution')
%             validFlag = false;
%             display(status)
%             pause;
        end
        
        % Continue if optimal solution is found
        
        % Move points to within bounds
        x(x > model.ub) = model.ub(x > model.ub);
        x(x < model.lb) = model.lb(x < model.lb);
        
        % Store point
        if (maxMin == 1)
            warmupPts(:,2*i-1) = x;
        else
            warmupPts(:,2*i) = x;
        end
        
        if (verbFlag)
            if mod(i,100)==0
                fprintf('%4.1f\n',i/nPoints*100);
            end
        end
        
        
    end
    if validFlag
        i = i+1;
    end 
end

centerPoint = mean(warmupPts,2);

.67*centerPoint*ones(1,nPoints);

% Move points in
if isempty(bias)
    warmupPts = warmupPts*.33 + .67*centerPoint*ones(1,nPoints);
else
    warmupPts = warmupPts*.99 + .01*centerPoint*ones(1,nPoints);
end

close(h);


