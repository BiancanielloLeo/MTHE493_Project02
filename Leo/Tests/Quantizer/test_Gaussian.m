%Leo Biancaniello

% Design a Scalar Quantizer for a Gaussian Source 
% (Not yet channel optimized)


% ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
%Parameters
N = 8;  % Number of quantization levels
mu = 0; % Mean of the Gaussian Source
sigma = 1; % Standard Deviation of the Gaussian Source

pdf = @(x) (1/(sigma*sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma).^2); % Gaussian PDF
cdf = @(x) 0.5 * (1 + erf((x - mu)/(sigma*sqrt(2)))); % Gaussian CDF

normalPDF = @(x) (1/sqrt(2*pi)) * exp(-0.5*x.^2); % Standard Normal PDF
% Update equations to take pdf instead of normalPDF***

% ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
% Initialize decision boundaries and reconstruction levels uniformly
% Use range of 3 standard deviations to start
thresholds = [-Inf, linspace(-3*sigma, 3*sigma, N-1), Inf]; % N-1 thresholds within 3 standard deviations aswell as +-Infinity

codebook = zeros(1,N);  % Initialize codebook with N zeros

% Calculate initial reconstruction levels as Gaussian centroid
% Mean of the Gaussian within the Nth region
for k = 1:N % 
    
    P = cdf(thresholds(k+1)) - cdf(thresholds(k));  % Denominator

    if P > 0    % Prevent division by 0
        codebook(k) = mu + sigma * ...
            (normalPDF((thresholds(k)-mu)/sigma) ...
            - normalPDF((thresholds(k+1)-mu)/sigma)) / P;
        % From "Quantizing for Noisy Channels" by Kurtenbach
    end
end


% ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
% Lloyd-Max Algorithm
% Decide on best N reconstruction levels and decision boundaries for the quantizer

maxIterations = 1000;
tolerance = 1e-4; %0.0001
MSE = zeros(1, maxIterations); % Initialize array with zeros to store each iterations MSE


for iteration = 1:maxIterations
    oldCodebook = codebook;

    % STEP 1: Update decision boundaries ---------- ---------- ---------- ---------- ----------
    thresholds(2:N) = (codebook(1:N-1) + codebook(2:N)) / 2; % Midpoint of adjacent reconstruction levels

    % STEP 2: Update reconstruction levels as Gaussian centroid---------- ---------- ---------- ----------
    for k = 1:N

        P = cdf(thresholds(k+1)) - cdf(thresholds(k));  % Denominator

        if P > 0    % Prevent division by 0
            codebook(k) = mu + sigma * ...
                (normalPDF((thresholds(k)-mu)/sigma) ...
                - normalPDF((thresholds(k+1)-mu)/sigma)) / P; 
            % From "Quantizing for Noisy Channels" by Kurtenbach
        end

    end

    % STEP 3: Calculate MSE ---------- ---------- ---------- ----------
    MSE(iteration) = 0;

    for k = 1:N
        % Calculate total expected squared quantization error
        % by integrating the probability weighted error over all N quantization regions
        integrand = @(x) (x-codebook(k)).^2 .* pdf(x);

        MSE(iteration) = MSE(iteration) + ...
            integral(integrand,thresholds(k),thresholds(k+1));

    end

    % STEP 4: Check convergence ---------- ---------- ---------- ----------
    if iteration > 1
        if(abs(MSE(iteration-1) - MSE(iteration)) / MSE(iteration-1) < tolerance)
            break;  % Stop if distortion converges
        end
    end


end


% ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
% Results
disp('---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------');
disp([num2str(N) '-level Quantizer for Gaussian Source']);
disp(' ');

disp('Optimal Reconstruction Levels:');
disp(codebook);

disp('Optimal Decision Boundaries:');
disp(thresholds);

disp('Final MSE:');
disp(MSE(iteration));

disp('Number of Iterations:');
disp(iteration);