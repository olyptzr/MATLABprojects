%% EXERCISE 2
clear; clc; close all; 
%% --- PART A: BASIC SIMULATION ---
fprintf('PART A: BASIC SIMULATION');
fprintf('\n--------------------------------------------------\n');
% Processor Information
fprintf('Processor Model: AMD Ryzen 5 5500U with Radeon Graphics (2.10 GHz)\n');

%% Simulation of random nanoparticle movement
% Parameter definition: Student ID (AEM) and simulation time [s].
AEM = 16126; 
T_max = 100000; 
% Definition of 3 cases for the number of particles N (numeric array)
% Using round() to ensure N is an integer.
N_cases = [round(AEM/10), round(AEM), round(AEM*10)];
% Main loop for each case of N
for i = 1:length(N_cases)
    N = N_cases(i); % Current N for this iteration
    
    fprintf('For N = %d --> ', N);
    
    % Start Timing
    tic; 
    
    % Initialization
    nA_tot = zeros(1, T_max);       % store the count in part A for each time point in a numeric array
    
    nA_t = N;               % Initial condition: All particles are in part A (nB_t=0)
    
    % Simulation Process 
    for j = 1:T_max
        
        % At most 1 particle passes per unit of time.
        % The probability of a particle leaving A to B is proportional to the count in A relative to total N
        % P(A->B) = nA / N
        % P(B->A) = nB / N = (N - nA) / N
        Dprob_AtoB = nA_t / N;
        
        % Generate a random number r in [0, 1]
        r = rand();
        
        if r <= Dprob_AtoB
            % Particle leaves A and goes to B
            nA_t = nA_t - 1;        % nB_current = nB_current + 1
        else
            % Particle leaves B and goes to A
            nA_t = nA_t + 1;        % nB_current = nB_current - 1
        end
        
        % Storing the value for the specific time point
        nA_tot(j) = nA_t;
    end
    
    % End Timing
    elapsed_time = toc;
    fprintf('Execution time = %.4f seconds\n', elapsed_time);
    
    % Theoretical Curve Calculation
    % Relation (1): nA(t) = (N/2) * (1 + exp(-2*t/N))
    t = 1:T_max;
    DnA_theory = (N/2) * (1 + exp(-2 * t / N));
    
    %% Plotting
    figure(i);       % New window for each case of N
    
    plot(t, DnA_theory, 'r', 'LineWidth', 1.5);        % Plot theoretical curve
    hold on;
    plot(t, nA_tot, 'b', 'LineWidth', 1);         % Plot simulation results
    grid on;
    title(sprintf('Theoretical relationship VS simulation for nA(t) (N = %d)', N));   % sprintf: convert N double -> string
    xlabel('Time (t) [s]');
    ylabel('Number of particles in A (n_A)');
    legend('Simulation', 'Theoretical Prediction');
    ylim([0, N]); 
    
    hold off;
    
end

%% PART B: STATISTICAL CONVERGENCE - MULTIPLE RUNS
fprintf('\n\nPART B: STATISTICAL CONVERGENCE - MULTIPLE RUNS');
fprintf('\n--------------------------------------------------\n');
% Parameters remain the same: AEM = 16126, T_max = 100000, N_cases = [round(AEM/10), round(AEM), round(AEM*10)]
M = 5; % Number of runs per N_case
% Outer loop for different N
for i = 1:length(N_cases)
    N = N_cases(i);
    all_runs_array = zeros(M, T_max);        % Table for all runs for the specific N
                                             % 5 x 10000 (runs x T_max), where each row --> separate simulation
    
    % Inner loop: Execution M times
    for j = 1:M
        nA_tot = zeros(1, T_max);
        nA_t = N; % Initial condition
        
        for k = 1:T_max
            Dprob_AtoB = nA_t / N;
            r = rand();
            
            if r <= Dprob_AtoB
                nA_t = nA_t - 1;
            else
                nA_t = nA_t + 1;
            end
            nA_tot(k) = nA_t;
        end
        
        all_runs_array(j, :) = nA_tot;       % Saving the result of this run in the correct row
    end
    
    %% Mean and Std Calculation
    % Calculate the mean value vertically (across the 5 runs for each t)
    nA_mean = mean(all_runs_array, 1);      % average across columns, 1 -> indicates dimension (columns)
    
    % Calculate standard deviation 
    nA_std = std(all_runs_array, 0, 1);     % Using 0 for sample standard deviation (division by N-1), not population
    %% Plots for each N
    figure(i+3);
    clf; % Clear figure
    t = 1:T_max;
    
    % 1. Plot Theoretical (Red line)
    % Theoretical Curve Calculation
    DnA_theory = (N/2) * (1 + exp(-2 * t / N));
    plot(t, DnA_theory, 'r', LineWidth=1.5);
    hold on;
    
    % 2. Plot Simulation Mean Value (Blue line)
    plot(t, nA_mean, 'b', LineWidth=1);
    
    % 3. Plot error bars (Mean + Std and Mean - Std) (Green dashed)
    % Adding error bars per step (sparse along t-axis to avoid continuous line)
    step = 10000; 
    t_errbars = 1:step:T_max; 
    errorbar(t(t_errbars), nA_mean(t_errbars), nA_std(t_errbars), 'g.', CapSize=10, LineWidth=1);
   
    grid on;
    title(sprintf('Statistical convergence (runs=%d) for N=%d', M, N));
    xlabel('Time (t) [s]');
    ylabel('Number of particles n_A');
    legend('Theory', 'Simulation Mean Value', 'Error bars', Location='best');
    ylim([0, N]);
    hold off;
    
end

%% PART C: COMPUTATIONAL COMPLEXITY ANALYSIS
fprintf('\n\nPART C: COMPUTATIONAL COMPLEXITY ANALYSIS');
fprintf('\n--------------------------------------------------\n');
% Parameters AEM = 16126 T_max = 100000 remain the same
% Defining 5 values for N (multiples of AEM)
multipliers = [1, 3, 6, 10, 15]; 
N_values = round(AEM * multipliers);
cpu_times = zeros(1, length(N_values));     % array 1 x 5
fprintf('Processor Model: AMD Ryzen 5 5500U\n');
%% Measurements Loop
for i = 1:length(N_values)
    N = N_values(i);
    
    fprintf('For N = %d ... ', N);
    
    % Start Timing
    tic; 
    
    % Simulation
    nA_t = N; 
    for t = 1:T_max
        Dprob_AtoB = nA_t / N;
        r = rand();
        
        if r <= Dprob_AtoB
            nA_t = nA_t - 1;
        else
            nA_t = nA_t + 1;
        end
    end
    
    % End Timing
    elapsed_time = toc;
    cpu_times(i) = elapsed_time;
    
    fprintf('Time = %.4f s\n', elapsed_time);
end

%% CPU time relationship with N^p - Least squares fit calculation (Fitting)
% Time = C * N^p
% Logarithmic form: log(Time) = log(C) + p * log(N)
% This is a line y = b + ax, where a = p (slope).
% polyfit(x, y, 1) returns [slope, intercept] --> Least squares line
EET = polyfit(log10(N_values), log10(cpu_times), 1);
slope = EET(1); % exponent p
intercept = EET(2);
fprintf('\nslope p=%.4f, intercept=%.4f\n\n', slope, intercept);
% Time = C * N^p
C = 10^intercept;      % Finding constant C
time_fitting = C * N_values .^ slope; 

%% Log-Log plot design
figure(7); 
% 1. Experimental data
loglog(N_values, cpu_times, 'bo', LineWidth=2);
hold on;
% 2. Fitting line
loglog(N_values, time_fitting, 'r--', LineWidth=2);
equationEET = sprintf('Fit: log(t) = %.4f + (%.4f) * log(N)', intercept, slope);
fprintf('Estimated relationship: CPU Time = %.4f * N^(%.4f) \n', C, slope);
grid on;
legend('CPU Measurements', equationEET, Location='northeast');
title('Log-log plot t-N');
xlabel('N'); 
ylabel('Time (s)');
hold off;



%% PART D: GENERALIZED INITIAL CONDITION
fprintf('\nPART D: Generalized Initial Condition');
fprintf('\n--------------------------------------------------\n');

%% Conversion of name and surname to ASCII codes
my_name = 'Olympia';
my_surname = 'Pantzartzi';
ascii_name = double(my_name);
ascii_surname = double(my_surname);
ON = sum(ascii_name);       % Sum of ASCII Name
EP = sum(ascii_surname);    % Sum of ASCII Surname
fprintf('Student: %s %s\n', my_name, my_surname);
fprintf('ON (Sum ASCII Name): %d\n', ON);
fprintf('EP (Sum ASCII Surname): %d\n', EP);
% Parameter definition
N = 1000;        % total number of particles in the box
T_max = 10*N;     % simulation time

%% Calculation of Ratio r and Initial Populations N1, N2
r = min(ON, EP) / max(ON, EP);
% Solving the system:
% 1) N1 / N2 = r  => N1 = r * N2
% 2) N1 + N2 = N  => r*N2 + N2 = N => N2 = N / (1+r) and N1 = N - N2
N2 = round(N / (1 + r));  % Particles in Part B
N1 = N - N2;              % Particles in Part A (N1 doesn't need rounding since N2 is already rounded)
fprintf('Ratio r: %.4f\n', r);
fprintf('Total N: %d\n', N);
fprintf('Initial Condition: N1 (A) = %d, N2 (B) = %d\n', N1, N2);
% Calculation of Part D initial condition (N1)
N2_partD = round(N / (1 + r));  
N1_partD = N - N2_partD;
fprintf('Initial Condition Comparison:\n');
fprintf('1. Part A: nA(0) = %d\n', N);
fprintf('2. Part D: nA(0) = %d (Due to name)\n', N1_partD);

%% SIMULATION 1: PART A (Starts from N)
AnA_array = zeros(1, T_max);
AnA_current = N; 
for t = 1:T_max
    AnA_array(t) = AnA_current;
    Aprob_AtoB = AnA_current / N;
    
    if rand() <= Aprob_AtoB 
        AnA_current = AnA_current - 1; 
    else
        AnA_current = AnA_current + 1; 
    end
end

%% SIMULATION 2: PART D (Starts from N1)
DnA_array = zeros(1, T_max); 
DnA_current = N1;
for i = 1:T_max
    DnA_array(i) = DnA_current;
    Dprob_AtoB = DnA_current / N;
    
    if rand() <= Dprob_AtoB
        DnA_current = DnA_current - 1;
    else
        DnA_current = DnA_current + 1;
    end
end

%% Plotting
figure(8); clf;
% Theoretical curves calculation
% General solution with initial value nA(0) = N1
% nA(t) = N/2 + (nA(0) - N/2) * exp(-2t/N) (previously nA(0)=N)
t_theory = 1:T_max;
AnA_theory = (N/2) + (N - N/2) * exp(-2 * t_theory / N);      % Theoretical curve for part A
DnA_theory = (N/2) + (N1 - N/2) * exp(-2 * t_theory / N);    % Theoretical curve for part D
% Simulation curves (Blue, Cyan)
plot(t_theory, AnA_array, 'b', LineWidth=1);
hold on;
plot(t_theory, DnA_array, 'c', LineWidth=1);
% Theoretical curves (Red, Magenta)
plot(t_theory, AnA_theory, 'r', LineWidth=1.5);
plot(t_theory, DnA_theory, 'm', LineWidth=1.5);
grid on;
legend('Simulation part A', 'Simulation part D', 'Theory part A', 'Theory part D', Location='best');
title(sprintf('Comparison for different initial count nA(0): nA(0)=%d for part A & nA(0)=%d for part D', N, N1));
xlabel('Time (t) [s]');
ylabel('Number of particles in A (n_A)');
ylim([0, N]);
hold off;