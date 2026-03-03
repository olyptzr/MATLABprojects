%% EXERCISE 1: QUESTIONS 2-5
%% Initialization
clear; clc; close all;

%% 1. Heat_transfer Function
fprintf('\n--- Question 1: General heat flow calculation function ---\n');
fprintf('The general heat flow calculation function Heat_transfer was written in a separate file Heat_transfer.m ---\n');

% General parameters for all questions
A = 10;             % Wall surface area (m^2)
T_in = 25 + 273.15; % Internal temperature (K) - e.g. 25 C
T_out = 0 + 273.15; % External temperature (K) - e.g. 0 C
dT_total = T_in - T_out;

%% 2. Composite multi-layer wall 
fprintf('\n--- Question 2: Composite Wall ---\n');
% For the multi-layer composite wall, I create a Structure Array, where:
% layers: array variable name, (1/2/3): layer index,
% .name, .thick, .k : fields (properties) of the specific layer.

% Layer definition (Concrete, Insulation, Gypsum)
% thick: thickness [m], k: conductivity [W/(mK)]
layers(1).name = 'Concrete';   layers(1).thick = 0.15; layers(1).k = 1.4;
layers(2).name = 'Insulation'; layers(2).thick = 0.10; layers(2).k = 0.04;
layers(3).name = 'Gypsum';     layers(3).thick = 0.05; layers(3).k = 0.5;

% Calculation of thermal resistances (R = thick / (k*A)) for conduction
R_total_cond = 0;
for i = 1:length(layers)
    layers(i).R = layers(i).thick / (layers(i).k * A);     % calculate thermal resistance of each layer
    R_total_cond = R_total_cond + layers(i).R;             % calculate total thermal resistance
end

% Calculation of total flow (conduction only)
Q_cond = dT_total / R_total_cond;
fprintf('Total Thermal Resistance (Conduction): %.4f K/W\n', R_total_cond);
fprintf('Heat Flow Q (Conduction): %.2f W\n', Q_cond);

% Calculation of interface temperatures
% T1: Internal surface, T2: Between 1-2, T3: Between 2-3, T4: External surface
T_current = T_in;   % Temperature initialization
fprintf('Interface temperatures (assuming T_in and T_out at the boundaries):\n');
fprintf('  T_current: %.2f C\n', T_current - 273.15);
for i = 1:length(layers)
    dT_layer = Q_cond * layers(i).R; % Temperature drop per layer
    T_current = T_current - dT_layer;
    fprintf('  Interface after layer %s: %.2f C\n', layers(i).name, T_current - 273.15); % print temperature in Celsius
end

%% 3. Combination of conduction + convection 
fprintf('\n--- Question 3: Conduction + Convection ---\n');
% Formula connecting conduction and convection:
% R_all = R_cond + R_conv where R_cond = thick / (k*A) and R_conv = 1/(h*A)
% h_i/h_o: convection coefficients internal and external

h_i = 10; % Internal convection coefficient [W/(m^2K)]
% Studying h_o (external) from 4 to 40 W/(m^2K) (air at rest vs increased wind)
h_o_range = 4:1:40; 
Qcond_conv = zeros(size(h_o_range));

% Constant wall resistance from before
R_wall = R_total_cond; 
R_conv_in = 1 / (h_i * A); % Transfer resistance inside
for i = 1:length(h_o_range)
    h_current = h_o_range(i);
    R_conv_out = 1 / (h_current * A);
    
    % Total resistance of conduction + convection: R_in + R_wall + R_out
    Rcond_conv = R_conv_in + R_wall + R_conv_out;
    
    % Using general formula Q = dT / R_total
    Qcond_conv(i) = dT_total / Rcond_conv;
end
fprintf('Total Thermal Resistance (Conduction + Convection): %.4f K/W\n', R_total_cond);
fprintf('Heat Flow Q (Conduction): %.2f W\n', Q_cond);

% Q - h_o Diagram
figure(1);
plot(h_o_range, Qcond_conv, 'LineWidth', 2);
grid on;
xlabel('Coefficient h_{out} [W/(m^2K)]');
ylabel('Heat Flow Q [W] (conduction + convection)');
title('Effect of External Wind (h_o) on Heat Flow');

%% 4. Adding radiation
fprintf('\n--- Question 4: Adding Radiation ---\n');
% Radiation calculation formula:
% Q_rad = sigma * epsilon * A * (T_surf^4 - T_env^4);

% Set a fixed average value for h_o
h_o_fixed = 20; 
epsilon = 0.91; % Emissivity coefficient for concrete
T_env = T_out;  % Set T_environment = T_out

% 1. Calculate Q with conduction + convection (without radiation)
R_conv_out_fixed = 1 / (h_o_fixed * A);
R_tot_no_rad = R_conv_in + R_wall + R_conv_out_fixed;
Q_no_rad = dT_total / R_tot_no_rad;

% 2. Calculate Radiation
% Finding external surface temperature (T_surf_out)
% Q = (T_surf_out - T_out) / R_conv_out => T_surf_out = Q * R_conv_out + T_out
% ! Heat flow moves from inside to outside, so T_surf > T_out
% Temperature drop in the external air layer:
dT_air_layer = Q_no_rad * R_conv_out_fixed; 
T_surf_out = T_out + dT_air_layer; 

% Using the Heat_transfer function from Question 1
params_rad.epsilon = epsilon;
params_rad.A = A;
params_rad.T_surf = T_surf_out;
params_rad.T_env = T_env;
Q_rad = Heat_transfer_en('radiation', params_rad);

% Total heat flow from the external surface (Conduction + Convection + Radiation)
Q_total_flow = Q_no_rad + Q_rad;    % Total loss from the external surface
fprintf('Q (Conduction + Convection): %.2f W\n', Q_no_rad);
fprintf('Q (Due to Radiation): %.2f W\n', Q_rad);
fprintf('Total Q (Conduction + Convection + Radiation): %.2f W\n', Q_total_flow);

% NOTE: This calculation is approximate. 
% In reality, because radiation removes energy, the surface will cool further (T_surf will drop). 
% This would change the conduction/convection flows again. 
% For absolute accuracy, the system should be solved iteratively until the temperature balances.
% Energy conservation at the surface:
% Q_from_inside = Q_to_air + Q_radiated
% (T_in - T_surf) / R_wall = h * (T_surf - T_out) + sigma * epsilon * (T_surf^4 - T_out^4)
% And then solve for T_surf.

%% 5. Parametric study and plots
fprintf('\n--- Question 5: Parametric Study ---\n');

% Case A: Q as a function of Insulation Thickness
Thick_ins_range = 0.01:0.01:0.20; % 1cm to 20cm (row vector)
Q_Thick_ins = zeros(size(Thick_ins_range));

% Constant resistances (internal/external air, concrete, and gypsum) 
R_air_l1_l3 = R_conv_in + (layers(1).thick/(layers(1).k*A)) + (layers(3).thick/(layers(3).k*A)) + R_conv_out_fixed;

% Calculation of variable resistance due to insulation and total heat flow
for i = 1:length(Thick_ins_range)
    R_ins = Thick_ins_range(i) / (layers(2).k * A);
    Q_Thick_ins(i) = dT_total / (R_air_l1_l3 + R_ins);
end

figure(2);
subplot(2,1,1);
plot(Thick_ins_range * 100, Q_Thick_ins, 'b-o', 'LineWidth', 1.5);
grid on;
xlabel('Insulation Thickness (cm)');
ylabel('Heat Flow Q (W)');
title('Q as a function of Insulation Thickness');

% Case B: Q vs. Insulation thermal conductivity (k)
k_ins_range = 0.02:0.005:0.1; % From high-quality to poor insulation
Q_k_ins = zeros(size(k_ins_range));
Thick_ins_fixed = 0.1; % Constant thickness at 10cm
for i = 1:length(k_ins_range)
    R_ins_var = Thick_ins_fixed / (k_ins_range(i) * A);
    Q_k_ins(i) = dT_total / (R_air_l1_l3 + R_ins_var);
end

subplot(2,1,2);
plot(k_ins_range, Q_k_ins, 'r-o', 'LineWidth', 1.5);
grid on;
xlabel('Insulation k Coefficient (W/mK)');
ylabel('Heat Flow Q (W)');
title('Q as a function of Insulation k Coefficient');
fprintf('The corresponding plots have been generated.\n');