%% ================================================================
% FILE: environment_engine.m
% REALISTIC ATMOSPHERIC + RF ENVIRONMENT MODEL
%
% Atmosphere : ISA standard + Dryden wind turbulence model
% RF field   : free-space + Gaussian jamming zone
% GPS field  : ionospheric scintillation proxy
%% ================================================================

function env = environment_engine(env)

%% ============================================================
% MEAN WIND  (slow Ornstein–Uhlenbeck process)
%% ============================================================
% Wind speed: mean-reverting around 3 m/s with σ ≈ 0.5 m/s
TAU_WIND = 120;   % relaxation time (steps)
MU_WIND  = 3.0;   % mean wind speed (m/s)

env.wind_speed = env.wind_speed ...
    + (MU_WIND - env.wind_speed)/TAU_WIND ...
    + 0.06 * randn;
env.wind_speed = max(0, env.wind_speed);

% Wind direction: slow random walk
env.wind_dir = mod(env.wind_dir + 1.2*randn, 360);

%% ============================================================
% GUST EVENTS  (rare but sharp, Beaufort-scale spikes)
%% ============================================================
if rand < 0.015
    gust_mag     = 2.5 + 2*rand;   % 2.5–4.5 m/s extra
    env.wind_speed = env.wind_speed + gust_mag;
end

%% ============================================================
% VISIBILITY  (0.5 = thick haze, 1.0 = clear)
%% ============================================================
env.visibility = min(1.0, max(0.5, env.visibility + 0.008*randn));

%% ============================================================
% TEMPERATURE & PRESSURE  (slow diurnal drift)
%% ============================================================
env.temperature = env.temperature + 0.005*randn;
env.pressure    = env.pressure    + 0.20 *randn;

%% ============================================================
% LOCAL WIND FIELD  (Dryden turbulence proxy)
% Returns a turbulence velocity vector for position p (m)
%% ============================================================
% Dryden spatial scale lengths at low altitude:
%   Lu = 200 m (longitudinal), Lw = 50 m (vertical)
Lu = 200; Lw = 50;

env.wind_field = @(p) env.wind_speed * [ ...
    0.6*sin(p(2)/Lu) + 0.4*randn*0.1 , ...
    0.6*cos(p(1)/Lu) + 0.4*randn*0.1 , ...
    0.3*sin(p(1)/Lw + p(2)/Lw) * 0.5 ];

%% ============================================================
% RF / COMMS INTERFERENCE FIELD
% Jamming zone — Gaussian spatial model (dBm degradation)
%
% FIX 1: Only initialise rf_field on the FIRST call.
% The anomaly engine stacks jammer closures on top of rf_field;
% overwriting it here every step would erase those overlays.
%% ============================================================
if ~isfield(env, 'rf_field') || ~isfield(env, 'rf_field_base')
    jam_centre = [20, -30, 10];   % (m) in local frame
    jam_radius = 25;              % (m) 1σ
    env.rf_field_base = @(p) ...
        -28 * exp(-norm(p - jam_centre)^2 / (2*jam_radius^2));
    env.rf_field = env.rf_field_base;
end
% NOTE: Do NOT reassign env.rf_field here after first call.
% The anomaly engine (comms-jammer case 7) owns rf_field during attacks.
% On attack expiry it resets rf_field to env.rf_field_base.

%% ============================================================
% GPS DISTORTION FIELD  (ionospheric scintillation proxy)
% Returns extra pseudorange error standard deviation (m)
%% ============================================================
gps_centre = [-25,  15, 10];
gps_radius  = 30;

env.gps_field = @(p) ...
    2.0 * exp(-norm(p - gps_centre)^2 / (2*gps_radius^2));

%% ============================================================
% AMBIENT ANOMALY RATE  (affects natural comm bursts in feature engine)
%% ============================================================
if ~isfield(env, 'anomaly_rate')
    env.anomaly_rate = 0.03;
end

env.anomaly_rate = env.anomaly_rate + 0.004*randn;

if rand < 0.01
    env.anomaly_rate = env.anomaly_rate + 0.04;
end

env.anomaly_rate = max(0.01, min(0.07, env.anomaly_rate));

end
