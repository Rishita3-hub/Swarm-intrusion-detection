%% ================================================================
% FILE: anomaly_injection_engine.m
% REALISTIC DRONE FAULT / ATTACK INJECTION
%
% Fault types (mapped to real failure modes):
%   1 — Motor failure       RPM collapse, ESC overcurrent then cutback
%   2 — GPS spoofing        Gradual position offset, fix looks valid
%   3 — GPS jamming         HDOP spike, satellite drop, position noise
%   4 — Battery failure     Voltage sag, overcurrent, cell imbalance
%   5 — IMU drift           Accel/gyro bias ramp (thermal / aging)
%   6 — Propeller damage    RPM oscillation, vibration spike
%   7 — Comms loss          RSSI collapse, packet loss burst
%% ================================================================

function [pos_obs, vel_inj, motor_inj, gps_inj, imu_inj, attack_state, env] = ...
    anomaly_injection_engine(pos, vel, motor, gps, imu, t, attack_state, env)

WARMUP       = 70;
NUM_DRONES   = size(pos,1);
ATTACK_TYPES = 7;

%% ============================
% INITIALISE STATE ON FIRST CALL
%% ============================
if isempty(attack_state)
    attack_state.active    = zeros(NUM_DRONES,1);
    attack_state.type      = zeros(NUM_DRONES,1);
    attack_state.duration  = zeros(NUM_DRONES,1);
    attack_state.time_left = zeros(NUM_DRONES,1);
    attack_state.severity  = zeros(NUM_DRONES,1);
    attack_state.cooldown  = zeros(NUM_DRONES,1);
    attack_state.motor_idx = ones(NUM_DRONES,1);   % which motor is affected

    % GPS spoofing internal accumulator (magnitude in metres).
    % Kept separate from gps.offset so GPS smoothing in main loop
    % does not corrupt the accumulator (avoids circular dependency).
    attack_state.spoof_mag = zeros(NUM_DRONES,1);

    % Temporal label: 1 only on the FIRST step of a new attack (onset marker)
    attack_state.onset     = zeros(NUM_DRONES,1);
end

%% Pass-through defaults
pos_obs   = pos;
vel_inj   = vel;
motor_inj = motor;
gps_inj   = gps;
imu_inj   = imu;

% Clear per-step overlays each timestep
% Note: batt_inj/vibe_inj/rssi_inj/loss_inj removed — effects now flow
% through physical channels (motor_inj.volt_sag, motor RPM asymmetry,
% env.rf_field) to eliminate label leakage into feature_engine.
attack_state.onset    = zeros(NUM_DRONES,1);   % reset every step; set to 1 at activation

if t < WARMUP
    return;
end

for i = 1:NUM_DRONES

    %% Cooldown counter
    if attack_state.cooldown(i) > 0
        attack_state.cooldown(i) = attack_state.cooldown(i) - 1;
    end

    %% ---- ACTIVATION ----
    % FIX 2: Cap (≤ 30% of swarm) only gates NEW activations.
    % Drones already mid-attack always continue — the old 'continue'
    % skipped the entire loop body, silently truncating active faults.
    if attack_state.active(i) == 0 && attack_state.cooldown(i) == 0
        if sum(attack_state.active) < ceil(0.3 * NUM_DRONES)
            if rand < 0.004
                attack_state.active(i)    = 1;
                attack_state.type(i)      = randi(ATTACK_TYPES);
                attack_state.duration(i)  = randi([30 80]);
                attack_state.time_left(i) = attack_state.duration(i);
                attack_state.severity(i)  = 0.4 + 0.5*rand;
                attack_state.motor_idx(i) = randi(4);
                % Temporal onset marker — fires ONLY on this first step
                attack_state.onset(i)     = 1;
            end
        end
    end

    %% ---- APPLY FAULT ----
    if attack_state.active(i) == 1

        phase = 1 - attack_state.time_left(i) / attack_state.duration(i);
        % Floored bell curve: I starts at 20% of severity at onset (phase=0)
        % instead of 0, giving a slight initial sensor signature that static
        % ML models can detect. Peaks at severity, fades back to ~20% at end.
        I     = attack_state.severity(i) * (0.2 + 0.8*sin(pi * phase));

        m = attack_state.motor_idx(i);   % affected motor index

        switch attack_state.type(i)

            %%----------------------------------------------------
            case 1  % MOTOR FAILURE
            %%----------------------------------------------------
            % RPM falls progressively (up to 85 % loss at peak)
            fail_frac = I * 0.85;
            motor_inj.rpm(i,m) = motor.rpm(i,m) * (1 - fail_frac);
            motor_inj.pwm(i,m) = motor.pwm(i,m) * (1 - fail_frac*0.5);

            % ESC current: spikes early (overcurrent) then drops (motor stall)
            if phase < 0.3
                motor_inj.curr(i,m) = motor.curr(i,m) * (1 + I*2.2);
            else
                motor_inj.curr(i,m) = motor.curr(i,m) * max(0, 1 - fail_frac*0.8);
            end

            % Remaining motors compensate → slightly higher RPM
            for mm = 1:4
                if mm ~= m
                    motor_inj.rpm(i,mm) = motor.rpm(i,mm) * (1 + I*0.12);
                end
            end

            % FIX 6: Realistic thrust-loss effect.
            % A dead motor removes ~25% thrust → drone descends and yaws.
            % Previous 0.20*randn was far too weak to model this.
            vel_inj(i,3)   = vel(i,3)   - fail_frac * 2.5;            % altitude loss (m/s)
            vel_inj(i,1:2) = vel(i,1:2) + fail_frac * 1.5 * randn(1,2); % lateral yaw disturbance

            %%----------------------------------------------------
            case 2  % GPS SPOOFING
            %%----------------------------------------------------
            % Gradual lateral offset; HDOP and satellite count unchanged
            % (attacker maintains plausible-looking fix).
            % pos (true ground-truth) is NEVER modified — the real
            % drone is physically at pos, but thinks it is at pos_obs.
            %
            % spoof_mag is the internal accumulator for the offset magnitude.
            % Using attack_state.spoof_mag (not gps.offset) as accumulator
            % decouples this from the exponential smoothing applied to
            % gps.offset in main_dataset_generator, preventing a regression
            % where smoothed-down values fed back as a smaller starting point.
            MAX_SPOOF_OFFSET = 8.0;   % metres
            spoof_dir = randn(1,3);  spoof_dir(3) = spoof_dir(3)*0.2;
            spoof_dir = spoof_dir / (norm(spoof_dir) + 1e-6);
            attack_state.spoof_mag(i) = min(MAX_SPOOF_OFFSET, ...
                                            attack_state.spoof_mag(i) + I * 1.2);
            gps_inj.offset(i,:) = attack_state.spoof_mag(i) * spoof_dir;

            % Fix quality looks normal — that what makes spoofing dangerous
            gps_inj.hdop(i)  = gps.hdop(i) * (1 - 0.1*I);
            gps_inj.nsats(i) = gps.nsats(i);

            % Observed position reflects spoofed GPS
            pos_obs(i,:) = pos(i,:) + gps_inj.offset(i,:);

            %%----------------------------------------------------
            case 3  % GPS JAMMING
            %%----------------------------------------------------
            % Signal quality collapses; position becomes noisy
            gps_inj.fix(i)   = max(0, gps.fix(i) - round(I * 2));
            gps_inj.hdop(i)  = gps.hdop(i) + I * 8.0;
            gps_inj.nsats(i) = max(2, gps.nsats(i) - round(I * 8));

            % Increased position noise proportional to HDOP
            pos_noise_scale = gps_inj.hdop(i) * 2.5;
            pos_obs(i,:) = pos(i,:) + pos_noise_scale * randn(1,3);

            %%----------------------------------------------------
            case 4  % BATTERY FAILURE (cell degradation / short)
            %%----------------------------------------------------
            % Internal resistance rises → voltage sag under load.
            % Sag is pre-computed here and written into motor_inj.volt_sag
            % so feature_engine reads a physical quantity, not a fault scalar.
            % This eliminates the former label-leakage path via attack_state.batt_inj.
            total_draw = sum(motor.curr(i,:)) + 1.5;   % A (motors + avionics)
            v_sag_val  = I * total_draw * 0.25;        % Ohmic sag (V)
            motor_inj.volt_sag(i) = v_sag_val;         % read by feature_engine

            % Motor temperatures rise from unbalanced cell loads
            motor_inj.temp(i,:) = motor.temp(i,:) + I * 18;

            %%----------------------------------------------------
            case 5  % IMU BIAS DRIFT (thermal soak or aging)
            %%----------------------------------------------------
            % FIX 5: Realistic MEMS-grade bias injection.
            % Previous values (2.8/2.0/1.4 m/s²) were 5–8× too large
            % (~25% of gravity at peak), making the fault trivially obvious.
            % Real thermal/aging drift: 0.01–0.5 m/s² acc, ~0.01–0.05 rad/s gyr.

            % Accelerometer bias injection (m/s²)
            imu_inj.bias_acc(i,:) = imu.bias_acc(i,:) + I * [0.35, 0.25, 0.18];

            % Gyroscope drift injection (rad/s)
            imu_inj.bias_gyr(i,:) = imu.bias_gyr(i,:) + I * [0.04, 0.03, 0.02];

            % Wrong attitude estimate causes position drift
            vel_inj(i,:) = vel(i,:) + I * 0.15 * randn(1,3);

            %%----------------------------------------------------
            case 6  % PROPELLER DAMAGE (chip, crack, imbalance)
            %%----------------------------------------------------
            % RPM oscillates at 1× rotation frequency
            osc_freq = motor.rpm(i,m) / 60;   % Hz
            osc = sin(2*pi * osc_freq * phase * 0.3);

            motor_inj.rpm(i,m)  = motor.rpm(i,m)  * (1 - 0.08*I + 0.18*I*osc);
            motor_inj.curr(i,m) = motor.curr(i,m) * (1 + 0.35*I*abs(osc));

            % Vibration signature flows through motor RPM asymmetry.
            % feature_engine computes rpm_asym = std(rpm1..rpm4) which
            % already captures the imbalance — no separate vibe_inj needed.
            % The large RPM oscillation above is sufficient for detection.

            %%----------------------------------------------------
            case 7  % COMMUNICATION LOSS (jamming / range / obstruction)
            %%----------------------------------------------------
            % All comms effects flow through env.rf_field exclusively.
            % rf_field is rebuilt in a post-loop block below from all
            % currently active Type-7 drones — this avoids monotonically
            % growing stacked closures and correctly handles multi-jammer
            % expiry (expired jammers are simply absent from the rebuild).
            % No rf_field modification here.

        end

        attack_state.time_left(i) = attack_state.time_left(i) - 1;

        %% ---- END OF FAULT ----
        if attack_state.time_left(i) <= 0
            attack_state.active(i)    = 0;
            attack_state.type(i)      = 0;
            attack_state.severity(i)  = 0;

            % Reset GPS spoofing offset and internal magnitude accumulator
            gps_inj.offset(i,:)       = zeros(1,3);
            attack_state.spoof_mag(i) = 0;   % reset accumulator
            gps_inj.fix(i)            = 3;
            gps_inj.hdop(i)           = gps.hdop(i);
            gps_inj.nsats(i)          = gps.nsats(i);
            % Note: rf_field is NOT reset here — it is rebuilt post-loop
            % from remaining active jammers, so expiry is handled correctly.

            % Recovery cooldown (120–250 steps ≈ 12–25 s at 10 Hz)
            attack_state.cooldown(i) = randi([120 250]);
        end
    end
end

%% ================================================================
% POST-LOOP: Rebuild env.rf_field from all currently active Type-7 drones
%
% This replaces per-drone closure stacking (old approach), which had two bugs:
%   1. Closures accumulated across steps → interference grew monotonically
%      with attack duration, independent of the bell-shaped I.
%   2. Expiry of ANY Type-7 drone reset rf_field_base, erasing all still-
%      active jammers.
%
% New approach: start from rf_field_base every timestep, then add one
% Gaussian term per active Type-7 drone using that drone current I.
% Expired jammers are absent from the rebuild — no special reset needed.
%% ================================================================
if isfield(env, 'rf_field_base')
    % W1 FIX: Pre-collect all active jammer positions and powers into arrays,
    % then build a SINGLE vectorised closure over those arrays.
    % This eliminates the N-deep closure chain (each prev_rf invoking the
    % previous) that grew with the number of active jammers.
    % With ≤3 simultaneous jammers the chain was harmless in practice, but
    % the array approach is O(1) closure depth for any swarm size.
    jam_sigma = 50;   % m  1σ spatial spread (same for all jammers)
    jpos = zeros(0, 3);   % N_active × 3
    jpow = zeros(0, 1);   % N_active × 1

    for i = 1:NUM_DRONES
        if attack_state.active(i) == 1 && attack_state.type(i) == 7
            phase_i = 1 - attack_state.time_left(i) / attack_state.duration(i);
            I_i     = attack_state.severity(i) * (0.2 + 0.8*sin(pi * phase_i));
            jpos(end+1, :) = pos(i,:);       %#ok<AGROW>
            jpow(end+1)    = I_i * 10;       %#ok<AGROW>
        end
    end

    if isempty(jpos)
        % No active jammers — restore clean baseline
        env.rf_field = env.rf_field_base;
    else
        % Single closure over captured arrays; depth = 1 regardless of N
        base_fn   = env.rf_field_base;
        js        = jam_sigma;
        env.rf_field = @(p) base_fn(p) ...
            + sum(-jpow .* exp(-sum((jpos - p).^2, 2) / (2*js^2)));
    end
end

end
