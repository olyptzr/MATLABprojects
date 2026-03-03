clear; close all; clc;
%% EXERCISE 2: KNN and Performance Metrics
fprintf('------------------------------------------2.1------------------------------------------\n')
%% 2.1 -------------------------------------------------------------------------------------------------------------
% ACCURACY IMPROVEMENT AND CALCULATION OF CM, RECALL, PRECISION, F1
%% Example: kNN Classification on Physics Data
% We create a simple dataset based on orbital mechanics:
% If total energy E = (1/2)mv^2 + U is negative → bound orbit (class 0)
% If E is positive → unbound orbit (class 1)
rng(1); % for reproducibility
% --- Generate synthetic physics data ---
N = 300;
mass = 1;                 % assume mass = 1 kg
velocity = 10*rand(N,1);  % velocities from 0–10 m/s
potential = -32 + 100*rand(N,1);  % potential energy from -32 to 78 joules
% Compute total energy E
E = 0.5*mass.*velocity.^2 + potential;
% Class labels: 0 = bound, 1 = unbound
labels = double(E > 0);     % Positive class=1 (Unbound orbit), Negative class=0 (Bound orbit)
% Combine features
X = [velocity potential];   % 2D numeric array -> matrix N=300 x 2 

%% --- Split into training and testing sets ---
cv = cvpartition(labels,'HoldOut',0.3);
Xtrain = X(training(cv),:);
Ytrain = labels(training(cv));
Xtest  = X(test(cv),:);
Ytest  = labels(test(cv));

%% ======================= ORIGINAL KNN (without feature selection / scaling) =======================
k_orig = 1;
distance_orig = "euclidean";
mdl_orig = fitcknn(Xtrain, Ytrain, NumNeighbors=k_orig, Distance=distance_orig);
Ypred_orig = predict(mdl_orig, Xtest);
classes = [1 0]; % positive class first
confMat_orig = confusionmat(Ytest,Ypred_orig, Order=classes);       % I define the order, otherwise MATLAB sorts classes in ascending order (0 before 1) 
TP_orig = confMat_orig(1,1); FN_orig = confMat_orig(1,2);
FP_orig = confMat_orig(2,1); TN_orig = confMat_orig(2,2); 
e = eps;    % Small constant to avoid division by 0 in Precision/Recall/F1
Acc_orig = (TP_orig + TN_orig) / numel(Ytest);
Prec_orig = TP_orig / (TP_orig + FP_orig + e);
Rec_orig = TP_orig / (TP_orig + FN_orig + e);
F1_orig = 2*Prec_orig*Rec_orig / (Prec_orig + Rec_orig + e);
metrics_orig = [Acc_orig, Prec_orig, Rec_orig, F1_orig];

%% ================= IMPROVED KNN (scaling, grid search & cv) =================
% The accuracy of KNN is strongly affected by the scale of features because it relies on distances.
% It decreases when many or irrelevant features exist, as they degrade the contribution of important weighted distances.
% For improvement, features will be normalized to have comparable contributions to the distance metric.
% Normalization is done in a way that avoids leakage within k-fold cross-validation (Standardize=true),
% meaning in each fold, mean/std are calculated only from the training part of the fold.
% For larger datasets, features can be removed via feature selection, but here I only have 2 features, so irrelevant features are not an issue.
% Additionally, a grid search on different k values and distance metrics with k-fold cross-validation will be performed to select the optimal hyperparameter combination.

%% --- Cross validation & Grid search: k and distance ---
k = 1:2:15;     % vector of numbers from 1 to 15 with step 2
distance_types = {"euclidean","cityblock","cosine"};
bestAcc = 0;        % variable initialization
bestK = 1;
bestDist = "euclidean";
for i = 1:length(k)
    for j = 1:length(distance_types)
        mdl = fitcknn(Xtrain, Ytrain, NumNeighbors=k(i), ...
            Distance=distance_types{j}, Standardize=true);     % Implements knn with the corresponding distance type each time
        cvmdl = crossval(mdl, 'KFold', 5);     % Performs cross-validation using folds and returns loss / predictions
        acc = 1 - kfoldLoss(cvmdl);     % This directly gives me the accuracy 
                                        % I use built-in k-fold instead of manual for simplicity
        if acc > bestAcc                % Calculates the best accuracy, distance, and number of neighbors
            bestAcc = acc;
            bestK = k(i);
            bestDist = distance_types{j};
        end
    end
end
fprintf("~RESULTS OF GRID SEARCH & CROSS VALIDATION\n\n")
fprintf("Best k: %d\n", bestK);
fprintf("Best distance: %s\n", bestDist);
fprintf("CV Accuracy: %.2f%%\n", bestAcc*100);

%% --- Train final model ---
best_mdl = fitcknn(Xtrain, Ytrain, NumNeighbors=bestK, Distance=bestDist, Standardize=true);

%% --- Prediction ---
Ypred = predict(best_mdl, Xtest);       % Predict only for best_mdl 

%% --- Confusion Matrix ---
fprintf("\n~COMPARISON OF CONFUSION MATRICES\n\n");
% Confusion Matrix with labels for original model
classes = [1 0]; % positive class first
confMat_orig_labeled = array2table(confMat_orig, ...
    RowNames={'Actual 1','Actual 0'}, ...
    VariableNames={'Pred 1','Pred 0'});
disp('Confusion Matrix of original model:');
disp(confMat_orig_labeled);
% Confusion Matrix with labels for best model
confMat_best = confusionmat(Ytest, Ypred, Order=classes);
confMat_best_labeled = array2table(confMat_best, ...
    RowNames={'Actual 1','Actual 0'}, ...
    VariableNames={'Pred 1','Pred 0'});
disp('Confusion Matrix of best model:');
disp(confMat_best_labeled);
TP_best = confMat_best(1,1);
FN_best = confMat_best(1,2);
FP_best = confMat_best(2,1);
TN_best = confMat_best(2,2);

%% --- Performance Metrics ---
Accuracy  = (TP_best + TN_best) / sum(confMat_best(:));
Precision = TP_best / (TP_best + FP_best + e);
Recall = TP_best / (TP_best + FN_best + e);
F1 = 2 * Precision * Recall / (Precision + Recall + e);
metrics_best = [Accuracy, Precision, Recall, F1];
fprintf("~METRICS OF BEST MODEL:\n\n");
fprintf("Test Accuracy : %.3f%%\n", Accuracy*100);
fprintf("Precision     : %.3f\n", Precision);
fprintf("Recall        : %.3f\n", Recall);
fprintf("F1-score      : %.3f\n", F1);

%% ================= Comparison of original and improved performance metrics in a table =================
MetricNames = {'Accuracy','Precision','Recall','F1'};
T = table(metrics_orig', metrics_best', ...     % I use ' to get the transpose and thus a row vector instead of column
    RowNames=MetricNames, VariableNames= {'Original','Improved'});
fprintf("\n\n");
fprintf("~COMPARISON BETWEEN THE ORIGINAL AND IMPROVED MODEL'S METRICS\n\n");
disp(T);

%% 2.2--------------------------------------------------------------------------------------------------------------
fprintf('------------------------------------------2.2------------------------------------------')
fprintf('\n\n');
% ACCURACY IMPROVEMENT
%% kNN Classification Example Using Real Physics Data (Ionosphere Radar Dataset)
rng(0);
%% --- Load data ---
data = readtable('ionosphere.data','FileType','text','Delimiter',',','ReadVariableNames',false);
X = data{:,1:34};               % features
Y = categorical(data{:,35});    % convert last column = Good/Bad to categorical labels

%% --- Train/test split ---
cv = cvpartition(Y, 'HoldOut', 0.3);
Xtrain = X(training(cv), :);
Ytrain = Y(training(cv));
Xtest  = X(test(cv), :);
Ytest  = Y(test(cv));
classes = categories(Y);
e = eps;   % Small constant for avoiding division by 0

%% ========================= Original KNN ===========================
Mdl_orig = fitcknn(Xtrain, Ytrain, NumNeighbors=5, Distance="euclidean" ,Standardize=true);     % Automatically performs z-score scaling
[Ypred_orig, score_orig] = predict(Mdl_orig, Xtest);    % score: n x 2, columns correspond to Mdl.ClassNames
% Confusion matrix 
cm_orig = confusionmat(Ytest, Ypred_orig, 'Order', classes);
% Positive class = classes(1), Negative = classes(2)
posIdx = 1; negIdx = 2;
% Basic counts & metrics per class
TP = cm_orig(1,1); 
FP = sum(cm_orig(2,1));
FN = sum(cm_orig(1,2)); 
TN = sum(cm_orig(2,2));
Precision_pos = TP/(TP+FP+e); 
Recall_pos = TP/(TP+FN+e); 
F1_pos = 2*(Precision_pos*Recall_pos)/(Precision_pos+Recall_pos+e);

TP = cm_orig(negIdx,negIdx); 
FP = sum(cm_orig(:,negIdx))-TP;
FN = sum(cm_orig(negIdx,:))-TP; 
TN = sum(cm_orig(:))-TP-FP-FN;
Precision_neg_orig = TP/(TP+FP+e); 
Recall_neg_orig    = TP/(TP+FN+e); 
F1_neg_orig        = 2*(Precision_neg_orig*Recall_neg_orig)/(Precision_neg_orig+Recall_neg_orig+e);

% Macro averages ORIGINAL
Precision_orig = mean([Precision_pos, Precision_neg_orig]); 
Recall_orig = mean([Recall_pos, Recall_neg_orig]); 
F1_orig = mean([F1_pos, F1_neg_orig]); 
Accuracy_orig = sum(diag(cm_orig))/sum(cm_orig(:));

%% ============================ Improved KNN ===========================
Mdl_opt = fitcknn(Xtrain, Ytrain, ...
    OptimizeHyperparameters="auto", ...
    HyperparameterOptimizationOptions=struct(AcquisitionFunctionName="expected-improvement-plus"));     
    % "OptimizeHyperparameters='auto'" uses Bayesian optimization to find 
    % the best hyperparameters for KNN (NumNeighbors, Distance, Standardize). 
    % The final model is trained with the Best Observed Feasible Point.

[Ypred_opt, score_opt] = predict(Mdl_opt, Xtest);

%% Confusion matrix & metrics per class
cm_opt = confusionmat(Ytest, Ypred_opt, 'Order', classes);
posIdx = 1; negIdx = 2;

TP = cm_opt(posIdx,posIdx); 
FP = sum(cm_opt(:,posIdx))-TP;
FN = sum(cm_opt(posIdx,:))-TP; 
TN = sum(cm_opt(:))-TP-FP-FN;
Precision_pos = TP/(TP+FP+e); 
Recall_pos    = TP/(TP+FN+e); 
F1_pos        = 2*(Precision_pos*Recall_pos)/(Precision_pos+Recall_pos+e);

TP = cm_opt(negIdx,negIdx); 
FP = sum(cm_opt(:,negIdx))-TP;
FN = sum(cm_opt(negIdx,:))-TP; 
TN = sum(cm_opt(:))-TP-FP-FN;
Precision_neg = TP/(TP+FP+e); 
Recall_neg    = TP/(TP+FN+e); 
F1_neg        = 2*(Precision_neg*Recall_neg)/(Precision_neg+Recall_neg+e);

% Macro averages OPTIMIZED
Precision_opt = mean([Precision_pos, Precision_neg]);
Recall_opt    = mean([Recall_pos, Recall_neg]);
F1_opt        = mean([F1_pos, F1_neg]);
Accuracy_opt  = sum(diag(cm_opt))/sum(cm_opt(:));

%% ==================== COMPARISON ORIGINAL VS OPTIMIZED ====================
fprintf("Positive class used for metrics/ROC-AUC: %s\n", string(classes(posIdx)));

%% --- ROC / AUC for positive class ---
% ORIGINAL
[~,colPos] = ismember(classes(posIdx), Mdl_orig.ClassNames);
scoresPos_orig = score_orig(:, colPos);
[Xroc_orig, Yroc_orig, ~, AUC_orig] = perfcurve(Ytest, scoresPos_orig, classes(posIdx));

% OPTIMIZED
[~,colPos] = ismember(classes(posIdx), Mdl_opt.ClassNames);
scoresPos_opt = score_opt(:, colPos);
[Xroc_opt, Yroc_opt, ~, AUC_opt] = perfcurve(Ytest, scoresPos_opt, classes(posIdx));

% Plot ROC
figure;
plot(Xroc_orig, Yroc_orig, 'b-', 'LineWidth', 1.5); hold on;
plot(Xroc_opt,  Yroc_opt,  'r-', 'LineWidth', 1.5);
grid on;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend(sprintf('Original ROC (AUC = %.3f)', AUC_orig), ...
       sprintf('Optimized ROC (AUC = %.3f)', AUC_opt), ...
       'Location','best');
title('ROC Curve Comparison – kNN Ionosphere');

%% --- Display results ---
fprintf("\n~COMPARISON OF kNN METRICS~\n\n")
fprintf('Original KNN metrics:\n'); fprintf('Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f\n', Accuracy_orig, Precision_orig, Recall_orig, F1_orig);
fprintf('Improved KNN metrics:\n'); fprintf('Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f\n', Accuracy_opt, Precision_opt, Recall_opt, F1_opt);
fprintf('ROC AUC (positive class):\n');
fprintf('  Original  = %.4f\n', AUC_orig);
fprintf('  Optimized = %.4f\n', AUC_opt);

% Original vs Improved Metrics Table
MetricNames = {'Accuracy','Precision','Recall','F1'};
metrics_orig = [Accuracy_orig, Precision_orig, Recall_orig, F1_orig];
metrics_opt  = [Accuracy_opt, Precision_opt, Recall_opt, F1_opt];
T = table(metrics_orig', metrics_opt', 'RowNames', MetricNames, ...
          'VariableNames', {'Original','Improved'});
fprintf("\n~ORIGINAL VS IMPROVED MODEL\n\n");
disp(T);

%% ================= Confusion matrix plot =================
figure;
subplot(1,2,1);
confusionchart(Ytest, Ypred_orig);
title('Original kNN – Confusion Matrix');
subplot(1,2,2);
confusionchart(Ytest, Ypred_opt);
title('Optimized kNN – Confusion Matrix');
% !!! OBSERVATION
% Although optimized kNN shows higher accuracy and F1-score on the test set, the original model 
% shows a slightly higher AUC. This means that, in terms of scoring, the original model has better 
% overall class separation/ranking ability, while the optimized model performs better on the 
% specific final prediction given by the kNN decision rule.

%% 2.4--------------------------------------------------------------------------------------------------------------
fprintf('------------------------------------------2.4------------------------------------------\n\n');
% After Q3: Improving MAPE in knn_regression_random
% I keep original MAPE and then perform grid search with k-fold CV on train set.
% In the improved version, I standardize inside CV to avoid leakage.
rng(1,'twister');
%% --- Example data ---
n = 200;
x = linspace(0,10,n)';
Y = sin(x) + 0.3*randn(n,1);
X = [x, x.^2];

%% --- Holdout split (Train/Test) ---
cv = cvpartition(n,'HoldOut',0.30);
trainIdx = training(cv); testIdx = test(cv);
Xtrain = X(trainIdx,:); Ytrain = Y(trainIdx);
Xtest  = X(testIdx,:);  Ytest  = Y(testIdx);
tol = 1e-2;   % For MAPE (since Y passes through 0)

%% ======================= ORIGINAL (Initial file logic) =======================
% Standardize using train stats
mu = mean(Xtrain,1); sigma = std(Xtrain,0,1);
sigma(sigma==0) = 1;
Xtr = (Xtrain - mu) ./ sigma;
Xte = (Xtest  - mu) ./ sigma;
% KNN params (original)
k0 = 5;
dist0 = 'euclidean';
Ypred0 = knn_regress(Xtr, Ytrain, Xte, k0, dist0);

% MAPE calculation (omit zero actuals) - Exactly as original
den0 = max(abs(Ytest), tol);        
mape_orig = mean(abs((Ytest - Ypred0) ./ den0)) * 100;      
fprintf('Original File: k=%d  dist=%s  -> MAPE=%.2f%%\n\n', k0, dist0, mape_orig);

%% ======================= GRID SEARCH & K-FOLD CV (Target: min MAPE) =======================
% Improve MAPE using knn_regress(Q3) and standardize INSIDE CV.
k_list = 1:1:40;
distance_types = {'euclidean','cityblock','cosine','chebyshev'};
bestMAPE = inf;         
bestK = 5;
bestDist = 'euclidean';
K = 5;
cv_train = cvpartition(numel(Ytrain), 'KFold', K);
for i = 1:length(k_list)
    for j = 1:length(distance_types)
        mape_folds = zeros(K,1);
        for f = 1:K
            trF = training(cv_train, f);
            vaF = test(cv_train, f);
            XtrF = Xtrain(trF,:);       
            YtrF = Ytrain(trF);         
            XvaF = Xtrain(vaF,:);       
            YvaF = Ytrain(vaF);         
            % --- Standardize inside fold (no leakage) ---
            muF = mean(XtrF,1);
            sigmaF = std(XtrF,0,1);
            sigmaF(sigmaF==0) = 1;
            XtrFs = (XtrF - muF) ./ sigmaF;
            XvaFs = (XvaF - muF) ./ sigmaF;
            % --- Predict in fold-val using knn_regress (Q3) ---
            YvaPred = knn_regress(XtrFs, YtrF, XvaFs, k_list(i), distance_types{j});
            % --- MAPE fold ---
            denF = max(abs(YvaF), tol);
            mape_folds(f) = mean(abs((YvaF - YvaPred) ./ denF)) * 100;
        end
        MAPE_cv = mean(mape_folds);
        if MAPE_cv < bestMAPE
            bestMAPE = MAPE_cv;
            bestK = k_list(i);
            bestDist = distance_types{j};
        end
    end
end
fprintf("~RESULTS OF GRID SEARCH & CROSS VALIDATION (MAPE)\n\n")
fprintf("Best k: %d\n", bestK);
fprintf("Best distance: %s\n", bestDist);
fprintf("CV MAPE: %.2f%%\n\n", bestMAPE);

%% ======================= FINAL MODEL (train full -> test) =======================
% Standardize with full train set stats
mu = mean(Xtrain,1);
sigma = std(Xtrain,0,1);
sigma(sigma==0) = 1;
Xtr = (Xtrain - mu) ./ sigma;
Xte = (Xtest  - mu) ./ sigma;
% Prediction on test with best params
Ypred_best = knn_regress(Xtr, Ytrain, Xte, bestK, bestDist);

% MAPE on test
den_best = max(abs(Ytest), tol);
mape_improved = mean(abs((Ytest - Ypred_best) ./ den_best)) * 100;
fprintf("~MAPE COMPARISON\n\n");
fprintf("Original MAPE : %.2f%%\n", mape_orig);
fprintf("Improved MAPE : %.2f%%  (k=%d, dist=%s)\n\n", mape_improved, bestK, bestDist);

%% --- Plot (optional) ---
figure;
scatter(Xtest(:,1), Ytest, 30, 'b', 'filled'); hold on;
scatter(Xtest(:,1), Ypred_best, 30, 'r');
legend('True','Predicted'); xlabel('x'); ylabel('y');
title('KNN regression (MAPE improved)'); grid on;

%% 2.3--------------------------------------------------------------------------------------------------------------
% Processing the function knn_regress and adding more distance options
function ypred = knn_regress(Xtrain, Ytrain, Xtest, k, distance)
% Professor's clarification:
% ypred = zeros(size(Xtest)); 
% for i= 1:length(Xtest) 
% d = abs(Xtrain- Xtest(i)); %distance to all points 
% [~,idx]= sort(d); % nearest neighbors 
% ypred(i)= mean(Ytrain(idx(1:k))); %mean value of neighbors 
% end 

% Adding distance options with optional argument.
% distance (optional): "cityblock"(default), "euclidean", "cosine", "chebyshev"
    if nargin < 5 || isempty(distance)      
        distance = 'cityblock';             
    end
    distance = lower(char(distance));       
    Xtrain = double(Xtrain);
    Xtest  = double(Xtest);
    Ytrain = double(Ytrain(:));             % Nx1 column
    N = size(Xtrain,1);     
    M = size(Xtest,1);      
    k = min(k, N);          
    ypred = zeros(M,1);     
    for i = 1:M
        p = Xtest(i,:);     
        % --- distance from test point p to all training points (N x 1) ---
        switch distance
            case 'cityblock'
                d = sum(abs(Xtrain - p), 2);
            case 'euclidean'
                d = sqrt(sum((Xtrain - p).^2, 2));
            case 'cosine'
                dotprod = Xtrain * p.';                
                norms_X = sqrt(sum(Xtrain.^2, 2));     
                norm_p  = norm(p);                     
                cos_sim  = dotprod ./ (norms_X * (norm_p + eps) + eps);   
                d = 1 - cos_sim;      
            case 'chebyshev'
                d = max(abs(Xtrain - p), [], 2);
            otherwise
                error('Unknown distance type: %s', distance);
        end
        % --- nearest neighbors & prediction ---
        [~, idx] = sort(d);                  
        ypred(i) = mean(Ytrain(idx(1:k)));   
    end
end

%{
%% NOTE ON COMMENTED CODE BELOW:
% The following section is an "advanced" version of the knn_regress function. It includes the 'weighting' parameter.
% (Distance-based weighting -> closer neighbors have a higher influence on the prediction).
% This version is kept for future reference and extensibility, but was 
% commented out to maintain consistency with the simpler uniform-average logic provided in the assignment's instructions.

function ypred = knn_regress_advanced(Xtrain, Ytrain, Xtest, k, distance, weighting)
% kNN regression: ypred(i)=mean of k neighbors (or weighted mean if weighting='distance')
% distance: 'cityblock', 'euclidean', 'cosine', 'chebyshev'
% weighting (optional): 'uniform'(default) ή 'distance'

    if nargin < 5 || isempty(distance)
        distance = 'cityblock';
    end
    if nargin < 6 || isempty(weighting)
        weighting = 'uniform';
    end

    distance  = lower(char(distance));
    weighting = char(weighting);

    Xtrain = double(Xtrain);
    Xtest  = double(Xtest);
    Ytrain = double(Ytrain(:));

    N = size(Xtrain,1);          % training points
    M = size(Xtest,1);           % test points
    k = min(k, N);               % δεν γίνεται k>N

    ypred = zeros(M,1);

    % (μόνο για cosine: προ-υπολογισμός norms train)
    if strcmp(distance,'cosine')
        normX = sqrt(sum(Xtrain.^2, 2));    % N x 1
    end

    for i = 1:M
        p = Xtest(i,:);

        switch distance
            case 'cityblock'
                d = sum(abs(Xtrain - p), 2);

            case 'euclidean'
                d = sqrt(sum((Xtrain - p).^2, 2));

            case 'chebyshev'
                d = max(abs(Xtrain - p), [], 2);

            case 'cosine'
                dotprod = Xtrain * p.';                 % N x 1
                normp = sqrt(sum(p.^2));                % scalar
                den = normX * (normp + eps);            % N x 1
                cos_sim = dotprod ./ (den + eps);       % N x 1
                d = 1 - cos_sim;

            otherwise
                error('Unknown distance type: %s', distance);
        end

        [~, idx] = sort(d, 'ascend');
        nn = idx(1:k);

        if strcmpi(weighting,'distance')
            w = 1 ./ (d(nn) + eps);
            ypred(i) = sum(w .* Ytrain(nn)) / sum(w);
        else
            ypred(i) = mean(Ytrain(nn));
        end
    end
end
%}