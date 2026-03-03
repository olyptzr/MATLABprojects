%% EXERCISE 1
% 1.1
function Q = Heat_transfer(mode, params)
% Calculation of heat flow Q depending on the mode of transfer.
% Q = Heat_transfer(mode, params)
% mode: Char vector defining the mode ('conduction', 'convection', 'radiation')
% params: Structure containing the necessary variables (A, k, dT, L, h, epsilon, T_surf, T_env)
%
% Note: Omni Calculator provides an Energy [J] formula by including time. 
% Here, I calculate power [Watts] as the exercise does not provide time and studies steady-state.
%
% If energy [J] is required, multiply the result by time t [s].

    % Stefan-Boltzmann constant [J/(s·m^2·k^4) = W/(m^2K^4)]
    sigma = 5.67e-8; 
    
    switch mode
        case 'conduction'
            % Conduction: Q = (k * A * dT) / L
            % Parameters: k, A, dT, L
            % Check if the params structure has all required parameters
            if isfield(params, 'k') && isfield(params, 'A') && isfield(params, 'dT') ...
                    && isfield(params, 'L')       
                Q = (params.k * params.A * params.dT) / params.L;
            else
                % Error (terminates execution)
                error('Missing parameters for conduction (k, A, dT, L).');   
            end
            
        case 'convection'
            % Convection: Q = h * A * dT
            % Parameters: h, A, dT
            if isfield(params, 'h') && isfield(params, 'A') && isfield(params, 'dT')
                Q = params.h * params.A * params.dT;
            else
                error('Missing parameters for convection (h, A, dT).');
            end
            
        case 'radiation'
            % Radiation: Q = sigma * epsilon * A * (T_surf^4 - T_env^4)
            % ! Temperatures must be in Kelvin.
            % Parameters: epsilon, A, T_surf, T_env
            if isfield(params, 'epsilon') && isfield(params, 'A') && isfield(params, 'T_surf') && isfield(params, 'T_env')
                Q = sigma * params.epsilon * params.A * (params.T_surf^4 - params.T_env^4);
            else
                error('Missing parameters for radiation (epsilon, A, T_surf, T_env).');
            end
            
        otherwise
            error('Invalid mode choice. Choose conduction, convection, or radiation.');
    end
end