%% Exercise 4: ARTIFICIAL NEURAL NETWORKS (Regression)
% Creating an ANN to predict the pendulum period based on 
% length (L), initial angle (theta0), and gravity (g).
clear; clc; close all;

%% A. Data Preparation 
rng(0);                % Reproducibility
N = 1000;              % Number of samples
% Larger N -> richer dataset -> better generalization & more stable training

% --- Dataset Creation: Generate inputs and calculate output with Gaussian noise ---

% 1. INPUT GENERATION
L = 0.2 + (2.0 - 0.2) * rand(N, 1);         % L belongs to [0.2, 2.0] m
theta_deg = 5 + (80 - 5) * rand(N, 1);      % Theta0 belongs to [5, 80] degrees. 
theta_rad = deg2rad(theta_deg);             % Angle conversion to radians! Must be in rad for the formula
g = 9.6 + (9.9 - 9.6) * rand(N, 1);         % g belongs to [9.6, 9.9] m/s^2

% 2. OUTPUT CALCULATION WITH GAUSSIAN NOISE
% Theoretical Period Calculation
% Using the formula for large angles (general formula)
% T ≈ 2*pi*sqrt(L/g) * [1 + (1/16)*theta^2 + (11/3072)*theta^4]
T_theory = 2 * pi * sqrt(L ./ g) .* (1 + (1/16)*(theta_rad.^2) ...
    + (11/3072)*(theta_rad.^4));     % Using ./,.*,.^ because L, theta, g are vectors
    
% Adding Gaussian Noise 
% T is on the order of 1–3 s. Assuming a small measurement error (std deviation 0.02 sec).
sigma = 0.02; 
noise = sigma * randn(N, 1);
T_noisy = T_theory + noise; % Target (Y) for the network to learn.

% 3. DATASET CREATION 
% Input X and Target Y matrices
% ! fitnet requires [features x samples] dimensions
X = [L, theta_rad, g]';      % 3 x N
Y = T_noisy';                % 1 x N

% 4. SPLIT train/validation/test (BEFORE normalization)
% (to calculate normalization parameters only from the train set and prevent data leakage)
idx = randperm(N);      % ensures randomness in sample partitioning for train, validation, and test
sampleTrain = floor(0.70*N);
sampleVal   = floor(0.15*N);
% The rest goes to test so it sums exactly to N
trainInd = idx(1:sampleTrain);
valInd   = idx(sampleTrain+1 : sampleTrain+sampleVal);
testInd  = idx(sampleTrain+sampleVal+1 : end);

% 5. NORMALIZATION in [-1,1] with mapminmax (based ONLY on the train set)
% !!! mapminmax works with features × samples dimensions, so I take X, Y as defined 
% Calculating psX, psY from train data, then applying the same scaling everywhere.
[~, psX] = mapminmax(X(:, trainInd), -1, 1);
[~, psY] = mapminmax(Y(:, trainInd), -1, 1);        % Xn_train, Yn_train are not used elsewhere, hence ~
Xn = mapminmax('apply', X, psX);
Yn = mapminmax('apply', Y, psY);

% Normalization parameters psX, psY will be used for:
% psY: reverse scaling predictions back to original units
% psX: normalizing new X samples with the same transformation as the train set

%% B. NEURAL NETWORK
% 1. Creating a feedforward neural network.
hiddenNeurons = 20;           % 1 hidden layer with 20 neurons 
net = fitnet(hiddenNeurons);  % regression feedforward network
net.trainFcn = 'trainlm';     % Levenberg–Marquardt: fast convergence for small/medium regression problems
net.trainParam.max_fail = 10; % early stopping: stops if validation doesn't improve for 10 consecutive checks

% Training parameters
net.trainParam.epochs = 200;        % epochs = number of training cycles (passes) performed
net.trainParam.showWindow = true;   % Open the training window

% 2. Separating the net into train/validation/test
net.divideFcn = 'divideind';        % Locking the split (so the network is trained/evaluated on exact indices) 
net.divideParam.trainInd = trainInd;
net.divideParam.valInd   = valInd;
net.divideParam.testInd  = testInd;

% 3. Training the network with normalized data
[net, tr] = train(net, Xn, Yn);
% net = trained NN (with updated weights/biases after training)
% tr  = train/val/test indices, performance per epoch, termination reason

% 4. Prediction on the test set
Xn_test = Xn(:, tr.testInd);    % inputs of test samples
Yn_test = Yn(:, tr.testInd);    % actual outputs of test samples
% tr.testInd = indices of the samples assigned to the test set
Yn_pred = net(Xn_test);

% 5. Reverse mapping from [-1,1] back to the original scale (seconds)
Y_test = mapminmax('reverse', Yn_test, psY)';   % Nx1
Y_pred = mapminmax('reverse', Yn_pred, psY)';   % Nx1
% Converting to column vectors for easier arithmetic, scatter plots, and graphing

%% C. EVALUATION
% 1. Calculating metrics on the test set
error = Y_test - Y_pred;
MSE  = mean(error.^2);
RMSE = sqrt(MSE);
MAE  = mean(abs(error));
MAPE = mean(abs(error ./ Y_test)) * 100;  % In %

fprintf('\n=================== EXERCISE 4: ANN Regression ===================\n');
fprintf('\n--- Sample counts in train-validation-test sets ---\n');
fprintf('Train: %d | Val: %d | Test: %d\n', ...
    numel(tr.trainInd), numel(tr.valInd), numel(tr.testInd));
fprintf('\n--- MSE, MAE, RMSE, MAPE calculation on test set ---\n');
fprintf('MSE  : %.4f\n', MSE);
fprintf('RMSE : %.4f\n', RMSE);
fprintf('MAE  : %.4f\n', MAE);
fprintf('MAPE : %.4f %%\n', MAPE);

% 2. Plot of Actual vs. Predicted Period
figure;
% One point for each sample (Treal, Tpred)
scatter(Y_test, Y_pred, 25, 'filled');      
grid on; 
hold on;
minT = min([Y_test; Y_pred]); 
maxT = max([Y_test; Y_pred]);

% Selecting min/max values between real and pred so the diagonal
% extends across the full range, even if predictions exceed actual range.
% Drawing y=x diagonal from (minT, minT) to (maxT, maxT)
plot([minT maxT], [minT maxT], 'r--', 'LineWidth', 2);
xlabel('Actual Period T (s)');
ylabel('Predicted Period Tpred (s)');
title(sprintf('Actual vs Predicted (RMSE=%.4f s, MAPE=%.2f%%)', RMSE, MAPE));
axis equal;
legend('Test samples','Ideal y=x','Location','best');