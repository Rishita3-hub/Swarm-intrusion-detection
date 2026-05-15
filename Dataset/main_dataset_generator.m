%% ================================================================
% FILE: main_dataset_generator.m
% UAV Swarm Dataset Generator — Real Drone Sensor Model
% Inspired by ArduPilot/PX4 log fields (ATT, IMU, GPS, BARO, CURR, RCOU)
%
% Enhancements over baseline:
%   - Leader follows a lawnmower surveillance path (swarm_engine)
%   - Battery-low RTB: drones auto-return to base (swarm_engine)
%   - Leader failure triggers immediate re-election (swarm_intelligence_engine)
%   - New label column: rtb_flag (1 = drone is returning to base)
%% ================================================================
clc; clear; close all;
clear swarm_intelligence_engine   % flush persistent leader / timer state
rng(42);   % reproducible dataset

%% ────────────────────────────────────────────────────────────────
% CONFIG
%% ────────────────────────────────────────────────────────────────
NUM_DRONES   = 10;
TIME_STEPS   = 5000;   % 10 drones × 5000 steps = 50 000 rows
NUM_FEATURES = 60;

% Columns: time + drone_id + features + labels
%   anomaly_flag, attack_class, severity, anomaly_onset,
%   drone_state,  response_type, rtb_flag   ← NEW
TOTAL_COLS = 2 + NUM_FEATURES + 7;
DATA       = zeros(NUM_DRONES * TIME_STEPS, TOTAL_COLS);
row_idx    = 1;

VISUAL_STEP = 5;
SIM_SPEED   = 0.01;
leader_id   = 1;

%% ────────────────────────────────────────────────────────────────
% INITIAL POSITIONS  (NED-like local frame, metres)
%% ────────────────────────────────────────────────────────────────
CRUISE_ALT = 15;
BASE_POS   = [0, 0, 0];   % home / RTB landing point

leader_pos = [-80, -80, CRUISE_ALT];   % start at first waypoint
pos = zeros(NUM_DRONES, 3);
vel = zeros(NUM_DRONES, 3);

for i = 1:NUM_DRONES
    angle    = 2*pi*(i-2)/(NUM_DRONES-1);
    pos(i,:) = leader_pos + [14*cos(angle), 14*sin(angle), rand*2];
    vel(i,:) = [cos(angle), sin(angle), 0] * 0.5;
end

%% ────────────────────────────────────────────────────────────────
% BATTERY STATE  (4S LiPo: 13.2 V – 16.8 V)
%% ────────────────────────────────────────────────────────────────
batt.volt     = 16.0 + 0.4*rand(NUM_DRONES,1);
batt.curr     = 19   + 2  *randn(NUM_DRONES,1);
batt.mah      = zeros(NUM_DRONES,1);
batt.capacity = 5000 * ones(NUM_DRONES,1);
batt.pct      = 95   + 3  *rand(NUM_DRONES,1);

%% ────────────────────────────────────────────────────────────────
% MOTOR STATE  (920 KV, 10-inch props)
%% ────────────────────────────────────────────────────────────────
motor.rpm    = 6800*ones(NUM_DRONES,4) + 150*randn(NUM_DRONES,4);
motor.pwm    = 1480*ones(NUM_DRONES,4) +  15*randn(NUM_DRONES,4);
motor.curr   =  5.0*ones(NUM_DRONES,4) + 0.4*randn(NUM_DRONES,4);
motor.temp   =   38*ones(NUM_DRONES,4) +   3*randn(NUM_DRONES,4);
motor.failed = zeros(NUM_DRONES,4);
motor.volt_sag  = zeros(NUM_DRONES,1);
motor.rtb_flag  = zeros(NUM_DRONES,1);   % will be updated by swarm_engine

%% ────────────────────────────────────────────────────────────────
% IMU STATE
%% ────────────────────────────────────────────────────────────────
imu.bias_acc  = 0.02*randn(NUM_DRONES,3);
imu.bias_gyr  = 0.005*randn(NUM_DRONES,3);
imu.temp      = 45*ones(NUM_DRONES,1) + 3*randn(NUM_DRONES,1);

%% ────────────────────────────────────────────────────────────────
% GPS STATE
%% ────────────────────────────────────────────────────────────────
gps.fix    = 3 * ones(NUM_DRONES,1);
gps.hdop   = 0.9 + 0.2*rand(NUM_DRONES,1);
gps.nsats  = round(11 + 2*rand(NUM_DRONES,1));
gps.offset = zeros(NUM_DRONES,3);

%% ────────────────────────────────────────────────────────────────
% ENVIRONMENT
%% ────────────────────────────────────────────────────────────────
env.wind_speed  = 2.5;
env.wind_dir    = rand*360;
env.visibility  = 0.92;
env.temperature = 25;
env.pressure    = 101325;

%% ────────────────────────────────────────────────────────────────
% SWARM & ATTACK STATES
%% ────────────────────────────────────────────────────────────────
drone_state   = zeros(NUM_DRONES,1);
response_type = zeros(NUM_DRONES,1);
attack_state  = [];

%% ────────────────────────────────────────────────────────────────
% VISUALIZATION
%% ────────────────────────────────────────────────────────────────
figure('Color','white','Name','UAV Swarm — Surveillance Mission');

prev_pos_obs = pos;

%% ════════════════════════════════════════════════════════════════
% MAIN SIMULATION LOOP
%% ════════════════════════════════════════════════════════════════
for t = 1:TIME_STEPS

    %% Environment update
    env = environment_engine(env);

    %% Anomaly / fault injection
    motor.volt_sag = zeros(NUM_DRONES,1);
    [pos_obs, vel_inj, motor_inj, gps_inj, imu_inj, attack_state, env] = ...
        anomaly_injection_engine(pos, vel, motor, gps, imu, t, attack_state, env);

    %% Feature extraction
    features = feature_engine(pos_obs, vel_inj, batt, motor_inj, imu_inj, ...
                               gps_inj, env, leader_id, prev_pos_obs);

    %% Swarm intelligence (detection + leader election)
    [drone_state, response_type, leader_id, vel] = swarm_intelligence_engine( ...
        pos_obs, vel_inj, features, drone_state);

    %% Blend injected dynamics into real physics
    ALPHA_BLEND = 0.75;
    vel_real = (1 - ALPHA_BLEND)*vel + ALPHA_BLEND*vel_inj;

    motor_real          = motor;
    motor_real.rpm      = motor_inj.rpm;
    motor_real.pwm      = motor_inj.pwm;
    motor_real.curr     = motor_inj.curr;
    motor_real.temp     = motor_inj.temp;
    motor_real.volt_sag = motor_inj.volt_sag;

    %% Real physics update (positions, motor, battery, RTB)
    [pos, vel, motor, batt] = swarm_engine(pos, vel_real, motor_real, batt, env, leader_id);

    %% Persist injected GPS state
    gps.offset = 0.9*gps.offset + 0.1*gps_inj.offset;
    gps.hdop   = 0.9*gps.hdop   + 0.1*gps_inj.hdop;
    gps.nsats  = gps_inj.nsats;
    gps.fix    = gps_inj.fix;

    %% True labels
    anomaly_flag  = attack_state.active;
    attack_class  = attack_state.type;
    severity      = attack_state.severity;
    anomaly_onset = attack_state.onset;

    %% RTB flag from swarm_engine (written into motor struct)
    rtb_label = motor.rtb_flag;   % NUM_DRONES × 1

    %% Store rows
    for d = 1:NUM_DRONES
        DATA(row_idx,:) = [t d ...
            features(d,:) ...
            anomaly_flag(d) attack_class(d) severity(d) anomaly_onset(d) ...
            drone_state(d) response_type(d) rtb_label(d)];
        row_idx = row_idx + 1;
    end

    %% Slow IMU bias drift
    imu.bias_acc = imu.bias_acc + 0.0001*randn(NUM_DRONES,3);
    imu.bias_gyr = imu.bias_gyr + 0.00002*randn(NUM_DRONES,3);
    imu.temp     = imu.temp + 0.01*randn(NUM_DRONES,1);

    %% Visualization
    if mod(t, VISUAL_STEP) == 0
        visualize_swarm(pos, vel, anomaly_flag, attack_class, severity, t, leader_id);
        pause(SIM_SPEED);
    end

    %% Update previous observed position
    if all(size(pos_obs) == size(prev_pos_obs)) && all(isfinite(pos_obs(:)))
        prev_pos_obs = pos_obs;
    else
        prev_pos_obs = pos;
    end

end

%% ════════════════════════════════════════════════════════════════
% COLUMN NAMES
%% ════════════════════════════════════════════════════════════════
cols = [ ...
    "time", "drone_id", ...
    "acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","imu_temp", ...
    "roll","pitch","yaw","des_roll","des_pitch","heading_err", ...
    "gps_lat","gps_lon","gps_alt","gps_speed","gps_course","gps_fix","gps_hdop","gps_nsats", ...
    "baro_alt","baro_press","baro_temp", ...
    "volt","curr","mah_consumed","batt_pct", ...
    "pwm1","pwm2","pwm3","pwm4","rpm1","rpm2","rpm3","rpm4","mcurr1","mcurr2","mcurr3","mcurr4", ...
    "vibe_x","vibe_y","vibe_z", ...
    "groundspeed","airspeed","vert_speed","turn_rate","curvature", ...
    "neighbor_count","sep_min","formation_score","centroid_dist","leader_dist", ...
    "rssi","snr","latency","packet_loss","link_quality", ...
    "wind_speed","wind_dir", ...
    "anomaly_flag","attack_class","severity","anomaly_onset", ...
    "drone_state","response_type","rtb_flag" ...
];

%% ════════════════════════════════════════════════════════════════
% SAVE
%% ════════════════════════════════════════════════════════════════
T = array2table(DATA, 'VariableNames', cols);
writetable(T, 'uav_swarm_dataset.csv');

disp("======================================");
disp("DATASET GENERATED SUCCESSFULLY");
disp("Rows    : " + size(T,1));
disp("Columns : " + size(T,2));
disp("RTB events: " + sum(T.rtb_flag));
disp("======================================");

verify_dataset(T);
