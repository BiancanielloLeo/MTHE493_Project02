%   Leo Biancaniello



% ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
% Constants 
% 0 <= BIT_ERROR_RATE + BIT_ERASURE_RATE <= 1
BIT_ERROR_RATE = 0.1;   
BIT_ERASURE_RATE = 0.1;                      
BIT_CORRECT_RATE = 1 - BIT_ERROR_RATE - BIT_ERASURE_RATE;     
DELTA = 0.0001; % <=0

% Input
X = [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0];
N = length(X);

% Noise
Z = zeros(1,N);

% Output
Y = zeros(1,N); 



% ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
% Initialize Noise Process
r = rand; % Random number in[0,1] 

if r < BIT_ERROR_RATE
    Z(1) = 1;
elseif r < BIT_ERROR_RATE + BIT_ERASURE_RATE
    Z(1) = 2;
else
    Z(1) = 0;
end

% Markov Chain
for i = 2:N
        P = [BIT_CORRECT_RATE, BIT_ERROR_RATE, BIT_ERASURE_RATE];
        lastState = Z(i-1);
        P(lastState + 1) = P(lastState + 1) + DELTA;
        P = P / (1 + DELTA);
        
        r = rand;

        if r < P(1)
                Z(i) = 0;
        elseif r < P(1) + P(2)
                Z(i) = 1;
        else
                Z(i) = 2;
        end
end



% Alter Output
for i = 1:N
    if Z(i) == 0
        Y(i) = X(i);       
    elseif Z(i) == 1
        Y(i) = 1 - X(i);   
    elseif Z(i) == 2
        Y(i) = 2;          
    end
end



% ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
% Display Results
disp('Input X:');       disp(X);
disp('Noise Z:');       disp(Z);
disp('Output Y:');      disp(Y);



% ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
% Calculate PSNR (Peak Signal-to-Noise Ratio)
% Assume Erasures are corrupted bits
errors = (Y ~= X);
num_errors = sum(errors);
MSE = num_errors / N;

% Calculatations
MAX_VALUE = 1;

if MSE == 0
    PSNR = Inf;
else
    PSNR = 10 * log10(MAX_VALUE^2 / MSE);
end

% Display results
disp(' ');
disp('--- PSNR Results ---');
disp(['Total bits: ', num2str(N)]);
disp(['Bit errors: ', num2str(sum((Y ~= X) & (Y ~= 2)))]);
disp(['Erased bits: ', num2str(sum(Y == 2))]);
disp(['Total corrupted bits: ', num2str(num_errors)]);
disp(['MSE: ', num2str(MSE)]);
disp(['PSNR: ', num2str(PSNR), ' dB']);