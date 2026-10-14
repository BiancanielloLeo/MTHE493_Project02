%   Leo Biancaniello


%Constants 
%0 <= BIT_ERROR_RATE + BIT_ERASURE_RATE <= 1
BIT_ERROR_RATE = 0.1;   
BIT_ERASURE_RATE = 0.1;                      
BIT_CORRECT_RATE = 1 - BIT_ERROR_RATE - BIT_ERASURE_RATE;     
DELTA = 0.0001; 

%Input
X = [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0];
N = length(X);

%Noise
Z = zeros(1,N);

%Output
Y = zeros(1,N); 



%Initialize Noise Process
r = rand; %Random number in[0,1] 

if r < BIT_ERROR_RATE
    Z(1) = 1; %Bit Error
elseif r < BIT_ERROR_RATE + BIT_ERASURE_RATE
    Z(1) = 2; %Bit Erasure
else
    Z(1) = 0; %Bit Correct
end

%Markov Chain
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



%Alter Output
for i = 1:N
    if Z(i) == 0
        Y(i) = X(i);       
    elseif Z(i) == 1
        Y(i) = 1 - X(i);   
    elseif Z(i) == 2
        Y(i) = 2;          
    end
end


% Display Results
disp('Input X:');       disp(X);
disp('Noise Z:');       disp(Z);
disp('Output Y:');      disp(Y);