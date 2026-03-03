clear; close all; clc;
% EXERCISE 3: DECISION TREES & SVM 

%% 1A
% DATA LOADING AND LABEL CONVERSION TO BINARY VARIABLE
varNames = {'fLength','fWidth','fSize','fConc','fConc1','fAsym',...     % Defining Variable names for each data column
    'fM3Long','fM3Trans','fAlpha','fDist','class'};
data = readtable('magic04.data', 'FileType','text', 'ReadVariableNames', false, 'Delimiter',','); % Loading data from .data file as table 
data.Properties.VariableNames = varNames;       % Assigning names to table columns
%head(data);
data_num = data{:,1:10};    % Isolating numeric features (columns 1:10) as Nx10 array for ML commands
%head(data_num);
data_labels01 = strcmp(data.class, 'g');        % Converting to binary: 1 if 'g', otherwise 0
%disp(data_labels01(12330:12340));

% SPLITTING INTO TRAIN AND TEST SETS (BINARY CLASSIFICATION) 
rng(1);                     % For reproducibility
DATA = data_num;
labels = data_labels01;
nan_rows = any(isnan(DATA), 2) | isnan(labels); % Check for NaN 
                                                % (if NaN exists in DATA or label, the whole row 
                                                % must be removed - using ...,2 in any() - 
                                                % so DATA and labels have same row count) 
DATA(nan_rows, :) = [];     % Removing samples with NaN 
                            % (none exist here, but since samples are many, 
                            % it should be checked if it wasn't already known)
labels(nan_rows)  = [];
cv = cvpartition(labels, 'HoldOut', 0.30);      % Partitioning based on labels (class info 0/1)
DATA_train = DATA(training(cv), :);
labels_train = labels(training(cv));
DATA_test  = DATA(test(cv), :);
labels_test  = labels(test(cv));
%mean(labels)        % total gamma percentage  |
%mean(labels_train)  % gamma percentage in train | -> check if split was correct
%mean(labels_test)   % gamma percentage in test  |

% Z-SCORE NORMALIZATION (STANDARD SCALER)
% Train set z-score !Fit only on train to avoid data leakage
[DATA_train_scaled, mu, sigma] = zscore(DATA_train);
% Scaling the test set with mu and std calculated from train set
DATA_test_scaled = (DATA_test - mu) ./ sigma;
% Labels remain the same (binary classification)
labels_train_scaled = labels_train;  
labels_test_scaled  = labels_test;
%mean(DATA_train_scaled)     % check correct zscore application (should be mean~=0, std~=1)
%std(DATA_train_scaled)

%% 1B
fprintf('---------------------------1B---------------------------\n');
% CONSTRUCTION OF DECISION TREE CLASSIFIER MODEL
% Definition of hyperparameters
max_depth = [5, 10, 15, 20, 50];
min_samples_leaf  = [1, 5, 10, 15, 20];
% Variable names (excluding class)
varNames_DATA = {'fLength','fWidth','fSize','fConc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist'};
% Creating cell array to store confusion matrices for every combination of maxdepth & minsamplesleaf
cm = cell(length(max_depth), length(min_samples_leaf));
% Array for accuracy storage
accuracy = zeros(length(max_depth), length(min_samples_leaf));
% Cell array for tree storage
tree_all = cell(length(max_depth), length(min_samples_leaf));

% EXPERIMENTING WITH DIFFERENT MAX_DEPTH & MIN_SAMPLES_LEAF 
for i = 1:length(max_depth)
    for j = 1:length(min_samples_leaf)
        % Decision Tree Training
        tree = fitctree(DATA_train_scaled, labels_train, ...
                        'MaxNumSplits', max_depth(i), ...
                        'MinLeafSize', min_samples_leaf(j), ...
                        'PredictorNames', varNames_DATA); % Adding variable names to tree for splits instead of x1,x2,...
        
        % Predictions on test set
        labels_pred = predict(tree, DATA_test_scaled);
        
        % Accuracy calculation ([TP+TN]/[TP+TN+FP+FN])
        accuracy(i, j) = sum(labels_pred == labels_test) / length(labels_test);
        
        % Confusion matrix calculation
        cm{i, j} = confusionmat(labels_test, labels_pred);
        % Tree storage
        tree_all{i,j} = tree;
        % Print confusion matrix for each combination of maxdepth & minsamplesleaf
        fprintf('max_depth=%d, min_samples_leaf=%d --> Accuracy: %.2f%%\n', ...
                max_depth(i), min_samples_leaf(j), accuracy(i, j)*100);
        disp('Confusion Matrix:');
        disp(cm{i, j});
        fprintf('-----------------------------------------------------------------------------\n');
    end
end
% TOTAL ACCURACY TABLE
disp('Accuracy matrix (rows=max_depth, cols=min_samples_leaf):');
accuracyTable = array2table(accuracy, ...
                             'RowNames', string(max_depth), ...
                             'VariableNames', string(min_samples_leaf));
disp(accuracyTable);
% Finding max accuracy and its positions i,j in the table
[maxVal, idxLinear] = max(accuracy(:));
[iBest, jBest] = ind2sub(size(accuracy), idxLinear);    % Mapping row/column to linear index 
                                                        % (MATLAB matrices are stored linearly)
fprintf('Best accuracy = %.5f%% at maxDepth=%d, minSamples=%d\n\n', ...
    maxVal*100, max_depth(iBest), min_samples_leaf(jBest));
% PLOTTING BEST MODEL
bestTree = tree_all{iBest, jBest};
view(bestTree, 'Mode', 'graph');

%% 1C
fprintf('1C\n');
% CONSTRUCTION OF SVM CLASSIFIER MODEL
% Parameter definition
C = [0.01, 10];                 % Different BoxConstraint values
kernel = {'linear', 'rbf'};     % Linear and RBF kernels
% Initializing metric storage
accuracy_svm = zeros(length(C), length(kernel));
precision_svm = zeros(length(C), length(kernel));
recall_svm = zeros(length(C), length(kernel));

% EXPERIMENTING WITH DIFFERENT C & KERNELS
for j = 1:length(kernel)
    for i = 1:length(C)
        % SVM Training
        svm = fitcsvm(DATA_train_scaled, labels_train, ...
            'KernelFunction', kernel{j}, ...
            'BoxConstraint', C(i), 'KernelScale', 'auto');
        
        % Predictions on test set
        labels_pred = predict(svm, DATA_test_scaled);
        
        % METRICS CALCULATION
        % Accuracy calculation ([TP+TN]/[TP+TN+FP+FN])
        accuracy_svm(i, j) = sum(labels_pred == labels_test) / length(labels_test);
        
        % Precision & Recall for binary classification (labels: 0/1)
        TP = sum((labels_pred == 1) & (labels_test == 1));
        FP = sum((labels_pred == 1) & (labels_test == 0));
        FN = sum((labels_pred == 0) & (labels_test == 1));
        precision_svm(i,j) = TP / (TP + FP);
        recall_svm(i,j) = TP / (TP + FN);
        
        % Print results
        fprintf('Kernel=%s, C=%.2f --> Accuracy=%.5f%%, Precision=%.5f, Recall=%.5f\n', ...
            kernel{j}, C(i), accuracy_svm(i, j)*100, precision_svm(i,j), recall_svm(i,j));
    end
end
% DISPLAYING SUMMARY TABLES (ACCURACY, PRECISION & RECALL)
fprintf('\n');
disp('Accuracy matrix (rows=C values, cols=kernels):');
disp(array2table(accuracy_svm, 'VariableNames', kernel, 'RowNames', string(C)));
disp('Precision matrix:');
disp(array2table(precision_svm, 'VariableNames', kernel, 'RowNames', string(C)));
disp('Recall matrix:');
disp(array2table(recall_svm, 'VariableNames', kernel, 'RowNames', string(C)));

%% 2A
fprintf('2A\n');
% CONSTRUCTION OF DECISION TREE REGRESSOR MODEL
% Data Loading
data2 = readtable('paper.xlsx');
% Last column -> target (path loss: continuous numeric values)
DATA2 = data2(:, 1:end-1);      % Features
target = data2(:, end);         % Path loss
% Train/Test split
rng(1);                         % For reproducibility
cv = cvpartition(height(data2), 'HoldOut', 0.3); % 70% train, 30% test split
DATA2_train = DATA2(training(cv), :);
target_train = target{training(cv), 1}; % Using { } because target must be vector/array
DATA2_test  = DATA2(test(cv), :);
target_test  = target{test(cv), 1};

% Hyperparameter definition and metric storage initialization
max_depth2 = [2, 5, 10, 15, 20, 50];    % Experimental MaxDepth2 values
MAE = zeros(length(max_depth2), 1);     % Mean Absolute Error
RMSE = zeros(length(max_depth2), 1);    % Root Mean Squared Error
MAPE = zeros(length(max_depth2), 1);    % Mean Absolute Percentage Error

% EXPERIMENTING WITH DIFFERENT MAX_DEPTH2
for i = 1:length(max_depth2)
    % Decision Tree Regressor Training
    tree = fitrtree(DATA2_train, target_train, 'MaxNumSplits', max_depth2(i));
    
    % Predictions on test set
    target_pred = predict(tree, DATA2_test);
    
    % Metric calculation
    errors = target_test - target_pred;
    
    MAE(i) = mean(abs(errors));
    RMSE(i) = sqrt(mean(errors.^2));
    MAPE(i) = mean(abs(errors ./ target_test)) * 100;
    
    % Printing
    fprintf('MaxDepth2=%d --> MAE=%.3f, RMSE=%.3f, MAPE=%.3f%%\n', ...
        max_depth2(i), MAE(i), RMSE(i), MAPE(i));
end
fprintf('\n');

%% 2B, 2Γ
fprintf('2Β & 2Γ\n');
% CONSTRUCTION OF SVR MODEL WITH LINEAR AND RBF KERNEL
% !Using the same train/test split as in Decision Tree
% Hyperparameter definition
C2 = [0.01, 10];                % Different BoxConstraint values
kernel2  = {'linear','rbf'};
% Metric initialization
MAE_svr  = zeros(length(C2), length(kernel2));
RMSE_svr = zeros(length(C2), length(kernel2));
MAPE_svr = zeros(length(C2), length(kernel2));

% EXPERIMENTING WITH DIFFERENT C2 & KERNELS
for j = 1:length(kernel2)
    for i = 1:length(C2)
        % SVR training (regression)
        svr = fitrsvm(DATA2_train, target_train, ...
                      'KernelFunction', kernel2{j}, ...
                      'BoxConstraint', C2(i), 'KernelScale', 'auto');
        
        % Predictions on test set
        target_pred = predict(svr, DATA2_test);
        
        % Metric calculation
        errors = target_test - target_pred;
        MAE_svr(i,j)  = mean(abs(errors));
        RMSE_svr(i,j) = sqrt(mean(errors.^2));
        MAPE_svr(i,j) = mean(abs(errors ./ target_test)) * 100;
        
        % Print results
        fprintf('Kernel=%s, C=%.2f --> MAE=%.3f, RMSE=%.3f, MAPE=%.3f%%\n', ...
                kernel2{j}, C2(i), MAE_svr(i,j), RMSE_svr(i,j), MAPE_svr(i,j));
    end
end
% SUMMARY METRIC TABLES
fprintf('\n');
disp('MAE matrix (rows=C values, cols=kernels):');
disp(array2table(MAE_svr, 'VariableNames', kernel2, 'RowNames', string(C2)));
disp('RMSE matrix:');
disp(array2table(RMSE_svr, 'VariableNames', kernel2, 'RowNames', string(C2)));
disp('MAPE matrix:');
disp(array2table(MAPE_svr, 'VariableNames', kernel2, 'RowNames', string(C2)));