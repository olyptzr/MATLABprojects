%% EXERCISE 1: DATA PREPROCESSING 
%% 1A: Data loading and storage in a table
data = load('Sensor_data.mat');
T = table(data.t(:), data.x(:), 'VariableNames', {'t','x'});
% I use data.t(:) and data.x(:) to store t,x vectors as column vectors
% so that table T has dimensions 1000x2 and each column is 1000x1.
% Display first 10 values for t and x
disp('First 10 values of t:');
disp(T.t(1:10));
disp('First 10 values of x:');
disp(T.x(1:10));

%% 1B: Identification of NaN values and replacement with linear interpolation
NaN_t = sum(ismissing(T.t));
NaN_x = sum(ismissing(T.x));
fprintf('Total NaN: t = %d, x = %d\n', NaN_t, NaN_x);
T.t = fillmissing(T.t, 'linear', 'EndValues', 'extrap');  % Use 'extrap' to replace NaNs at the boundaries
T.x = fillmissing(T.x, 'linear', 'EndValues', 'extrap');

%% 1C: Outlier handling
% Outlier detection
outliers = zeros(1, width(T));   % Array for number of outliers
T_noOutliers = T;
for k = 1:width(T)
    r = T{:,k};                 % Get column vector for each column k
    Q1 = quantile(r, 0.25);
    Q3 = quantile(r, 0.75);
    IQR = Q3 - Q1;
    lb = Q1 - 1.5*IQR;
    ub = Q3 + 1.5*IQR;
    outliers_mask = (r < lb) | (r > ub);        % Logical vector for outliers
    outliers(k) = sum(outliers_mask);           % Number of outliers
    % Replace outliers
    r(outliers_mask) = NaN;                     % Set outliers to NaN first
    r = fillmissing(r, 'linear', 'EndValues', 'extrap');  % Then replace with linear interpolation
    T_noOutliers{:,k} = r;                       % Return to new table T_noOutliers
end
% Display number of outliers per column
fprintf('Outliers in t: %d\n', outliers(1));
fprintf('Outliers in x: %d\n', outliers(2));

%% 1D: Data Normalization
% Normalization to [0,1] (min-max scaling)
T_minmax = T_noOutliers;    % Min-max and z-score scaling are performed on "cleaned" data
for k = 1:width(T)
    T_minmax{:,k} = rescale(T_noOutliers{:,k});
end
% Normalization as z-score (standard scaling)
T_zscore = T_noOutliers;  
for k = 1:width(T)
    T_zscore{:,k} = zscore(T_noOutliers{:,k});  
end
% Comparison of min-max and standard scaling
fprintf('Min-Max on t: min_t=%.2f, max_t=%.2f\n', min(T_minmax.t), max(T_minmax.t));
fprintf('Min-Max on x: min_x=%.2f, max_x=%.2f\n', min(T_minmax.x), max(T_minmax.x));
% In min-max:
% All data is mapped within [0,1]
% Relative order of values is preserved
% Distances between values are proportional to original data
disp('Min-Max normalized first 10 values:');
disp(T_minmax(1:10,:));
fprintf('Z-score on t: mean_t=%.2f, std_t=%.2f\n', mean(T_zscore.t), std(T_zscore.t));
fprintf('Z-score on x: mean_x=%.2f, std_x=%.2f\n', mean(T_zscore.x), std(T_zscore.x));
% Data is transformed so mean = 0 and standard deviation = 1
% Values can be negative or greater than 1
% Focuses on the position of values relative to the mean rather than absolute range
disp('Z-score normalized first 10 values:');
disp(T_zscore(1:10,:));

%% 1E: Plots
figure; hold on;        % Open figure window
% 1. Original data
plot(data.t, data.x, 'k.', 'DisplayName', 'Original Data');        
% 2. After NaN replacement
plot(T.t, T.x, 'b-', 'DisplayName', 'After NaN Removal');
% 3. After outlier handling
plot(T_noOutliers.t, T_noOutliers.x, 'r-', 'DisplayName', 'Without Outliers');
% 4. Normalization / Min-Max
plot(T_minmax.t, T_minmax.x, 'g-', 'DisplayName', 'Normalized/Min-Max');
% 5. Normalization / Z-Score
plot(T_zscore.t, T_zscore.x, 'c-', 'DisplayName', 'Normalized/Z-Score');
xlabel('t [s]');
ylabel('x [Sensor Measurement]');
title('Data Visualization across Preprocessing Stages');
legend;
grid on;

%% 1F: Save to CSV and MAT files
FinalT = table();       % Organize original and processed data in a table for saving
% Original data
FinalT.t_original = data.t(:);          
FinalT.x_original = data.x(:);
% After NaN
FinalT.t_noNaN = T.t;
FinalT.x_noNaN = T.x;
% After outliers
FinalT.t_noOutliers = T_noOutliers.t;
FinalT.x_noOutliers = T_noOutliers.x;
% Min-Max normalized
FinalT.t_minmax = T_minmax.t;
FinalT.x_minmax = T_minmax.x;
% Z-score normalized
FinalT.t_zscore = T_zscore.t;
FinalT.x_zscore = T_zscore.x;
% Save to CSV
writetable(FinalT, 'FinalData.csv');
% Save to MAT
save('FinalData.mat', 'FinalT');

%% 2: Application of Preprocessing Rules to physics_data_sample.csv 
% Loading data from physics_data_sample.csv
data2 = readtable('physics_data_sample.csv', 'Delimiter', ',');       
% Identify NaN values
data2_noNaN = data2;
for k = 1:width(data2)
    numNaN = sum(ismissing(data2{:,k}));
    fprintf('NaN in %s: %d\n', data2.Properties.VariableNames{k}, numNaN);         
end
% Replace NaN with spline interpolation 
data2_noNaN{:,:} = fillmissing(data2{:,:}, 'spline', 'EndValues', 'extrap');

% Outlier detection 
data2_noOutliers = data2_noNaN;
outliers2 = zeros(1, width(data2_noNaN));
for k = 1:width(data2_noNaN)
    v = data2{:,k};
    Q1_2 = quantile(v, 0.25);
    Q3_2 = quantile(v, 0.75);
    IQR_2 = Q3_2 - Q1_2;
    lb2 = Q1_2 - 1.5*IQR_2;
    ub2 = Q3_2 + 1.5*IQR_2;
    
    outliers2_mask = (v < lb2) | (v > ub2);
    outliers2(k) = sum(outliers2_mask);
    
    % Replace with NaN and spline interpolation 
    v(outliers2_mask) = NaN;
    data2_noOutliers{:,k} = fillmissing(v, 'spline', 'EndValues', 'extrap');
end
% Print number of outliers per column
for k = 1:width(data2_noNaN)
    fprintf('Outliers in %s: %d\n', data2_noOutliers.Properties.VariableNames{k}, outliers2(k));
end

% Robust Scaling 
% (Robust scaling uses median and IQR, making it naturally resistant to outliers)
data2_robust = data2_noNaN;   
for k = 1:width(data2_noNaN)
    v = data2{:,k};
    med = median(v);
    IQR_v = iqr(v);
    data2_robust{:,k} = (v - med) / IQR_v;
end

% Statistics before scaling
for k = 1:width(data2)
    fprintf('\nColumn: %s before scaling\n', data2.Properties.VariableNames{k});
    fprintf('mean=%.2f, std=%.2f, min=%.2f, max=%.2f\n', ...
        mean(data2{:,k}), std(data2{:,k}), min(data2{:,k}), max(data2{:,k}));
end

% Standard Scaling (Z-Score)
data2_zscore = data2_noOutliers;  
for k = 1:width(data2_noOutliers)
    data2_zscore{:,k} = zscore(data2_noOutliers{:,k});
end

% Statistics after scaling
for k = 1:width(data2_zscore)
    fprintf('\nColumn: %s after standard scaling\n', data2_zscore.Properties.VariableNames{k});
    fprintf('mean=%.2f, std=%.2f, min=%.2f, max=%.2f\n', ...
        mean(data2_zscore{:,k}), std(data2_zscore{:,k}), min(data2_zscore{:,k}), max(data2_zscore{:,k}));
end

% Comparison via Plots
figure;         
for k = 1:width(data2)
    subplot(width(data2), 1, k);
    plot(data2{:,k}, 'b', 'DisplayName', 'Original/noNaN'); hold on;
    plot(data2_zscore{:,k}, 'm', 'DisplayName', 'Standard Scaled (Z-Score)'); 
    title(['Column: ', data2.Properties.VariableNames{k}]);
    legend;
end

% Correlation & Covariance coefficients
R = corr(data2{:,:});  % Pearson correlation for linear relationship [-1,1]
C = cov(data2{:,:});   % Covariance shows joint variability
disp('Pearson correlation matrix:');
disp(R);
disp('Covariance matrix:');
disp(C);

% Heatmaps for correlation and covariance 
figure('Color', [0.6 0.4 0.2]);      % Open figure window with brown background
subplot(1,2,1);
h1 = heatmap(data2.Properties.VariableNames, ...
            data2.Properties.VariableNames, ...
            R, 'ColorLimits', [-1 1], 'Colormap', parula);          
h1.Title = 'Correlation matrix';
h1.CellLabelFormat = '%.2f';

subplot(1,2,2);
h2 = heatmap(data2.Properties.VariableNames, ...
            data2.Properties.VariableNames, C, 'Colormap', jet(512));
h2.Title = 'Covariance matrix';
h2.CellLabelFormat = '%.2f';

%% Save to CSV and MAT files
FinalTable2 = table();           % Create table with all data stages
% Names of each stage
stages = {'Original', 'noNaN', 'noOutliers', 'Robust', 'Zscore'};             
dataStages = {data2, data2_noNaN, data2_noOutliers, data2_robust, data2_zscore};  
for s = 1:length(stages)                
    stageName = stages{s};              
    TableStage = dataStages{s};         
    for k = 1:width(TableStage)         
        columnName = TableStage.Properties.VariableNames{k};            
        FinalTable2.([stageName '_' columnName]) = TableStage{:,k};     
    end
end
% Save to CSV
writetable(FinalTable2, 'FinalData2.csv');
% Save to MAT
save('FinalData2.mat', 'FinalTable2', 'R', 'C');

% Save R and C matrices to separate CSVs for supplementary analysis
R_table = array2table(R, 'VariableNames', data2.Properties.VariableNames, ...   
                            'RowNames', data2.Properties.VariableNames);     
C_table = array2table(C, 'VariableNames', data2.Properties.VariableNames, ...
                            'RowNames', data2.Properties.VariableNames);
writetable(R_table, 'Correlation_table.csv', 'WriteRowNames', true);           
writetable(C_table, 'Covariance_table.csv', 'WriteRowNames', true);