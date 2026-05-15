%% ================================================================
% FILE: feature_engine.m
% REALISTIC DRONE SENSOR FEATURE EXTRACTION
%
% Groups (60 total features):
%   1  IMU          acc_x/y/z, gyr_x/y/z, imu_temp          (7)
%   2  Attitude     roll, pitch, yaw, des_roll, des_pitch,
%                   heading_err                               (6)
%   3  GPS          lat, lon, alt, speed, course, fix,
%                   hdop, nsats                               (8)
%   4  Barometer    baro_alt, baro_press, baro_temp           (3)
%   5  Battery      volt, curr, curr_total, batt_pct          (4)
%   6  Motors       pwm1-4, rpm1-4, mcurr1-4                 (12)
%   7  Vibration    vibe_x, vibe_y, vibe_z                   (3)
%   8  Kinematics   groundspeed, airspeed, vert_speed,
%                   turn_rate, curvature                      (5)
%   9  Swarm        neighbor_count, sep_min, formation_score,
%                   centroid_dist, leader_dist                (5)
%  10  Comms        rssi, snr, latency, packet_loss,
%                   link_quality                              (5)
%  11  Environment  wind_speed, wind_dir                      (2)
%% ================================================================

function features = feature_engine(pos, vel, batt, motor, imu, gps, env, leader_id, prev_pos)

NUM_DRONES = size(pos,1);
features   = zeros(NUM_DRONES, 60);

GRAVITY    = 9.81;
RHO_AIR    = 1.225;   % kg/m^3

% Reference geodetic origin (Delhi region)
REF_LAT    = 28.7041;
REF_LON    = 77.1025;
M_PER_DEG  = 111320;  % metres per degree latitude

wind_vec = env.wind_speed * [cosd(env.wind_dir), sind(env.wind_dir), 0];

for i = 1:NUM_DRONES

    %% ==============================================================
    % GROUP 1 - IMU  (MPU-6000 noise model)
    % Accelerometer: ~0.35 m/s2/sqrt(Hz) white noise at 10 Hz sim step
    % Gyroscope:     ~0.017 rad/s/sqrt(Hz)
    %% ==============================================================
    % Specific force = measured_acc - gravity (body frame proxy)
    sf_x = vel(i,1)*0.1 - GRAVITY*sind(0) + imu.bias_acc(i,1) + 0.12*randn;
    sf_y = vel(i,2)*0.1                    + imu.bias_acc(i,2) + 0.12*randn;
    sf_z = -GRAVITY + norm(vel(i,:))*0.05  + imu.bias_acc(i,3) + 0.12*randn;

    acc_x = sf_x;
    acc_y = sf_y;
    acc_z = sf_z;

    gyr_x = imu.bias_gyr(i,1) + 0.008*randn;
    gyr_y = imu.bias_gyr(i,2) + 0.008*randn;
    gyr_z = imu.bias_gyr(i,3) + norm(vel(i,1:2))*0.008 + 0.008*randn;

    imu_temp = imu.temp(i) + 0.08*randn;

    %% ==============================================================
    % GROUP 2 - ATTITUDE  (estimated from EKF proxy)
    %% ==============================================================
    % Simple kinematic attitude proxy (no full EKF, but realistic range)
    vxy = norm(vel(i,1:2));
    roll  = atand(vel(i,2)*0.07 / (GRAVITY + abs(acc_z - (-GRAVITY))*0.01 + 1e-6)) + 0.4*randn;
    pitch = atand(-vel(i,1)*0.07 / GRAVITY) + 0.4*randn;
    yaw   = atan2d(vel(i,2)+1e-6, vel(i,1)+1e-6) + 0.3*randn;

    des_roll  = 0 + 0.3*randn;
    des_pitch = 0 + 0.3*randn;

    % FIX 10: heading_err now measures deviation from the desired formation
    % heading (pointing toward the swarm centroid from this drone's position).
    % Previously it was yaw minus itself -> always ~0 -> zero ML value.
    centroid_xy  = mean(pos(:,1:2), 1);
    des_heading  = atan2d(centroid_xy(2) - pos(i,2) + 1e-6, ...
                          centroid_xy(1) - pos(i,1) + 1e-6);
    heading_err  = abs(yaw - des_heading);
    heading_err  = min(heading_err, 360 - heading_err);  % shortest angle

    %% ==============================================================
    % GROUP 3 - GPS  (u-blox M8N: CEP ~ 1.5 m, HDOP-scaled)
    %% ==============================================================
    pos_noise_h = gps.hdop(i) * 1.5;   % 1sigma horizontal position noise (m)
    pos_noise_v = gps.hdop(i) * 3.0;   % vertical worse

    gps_lat = REF_LAT + (pos(i,2) + gps.offset(i,2)) / M_PER_DEG ...
              + (pos_noise_h / M_PER_DEG) * randn;
    gps_lon = REF_LON + (pos(i,1) + gps.offset(i,1)) / (M_PER_DEG * cosd(REF_LAT)) ...
              + (pos_noise_h / (M_PER_DEG*cosd(REF_LAT))) * randn;
    gps_alt    = pos(i,3) + gps.offset(i,3) + pos_noise_v * randn;
    gps_speed  = norm(vel(i,:)) + 0.08*randn;

    % gps_course from delta-position (not velocity vector).
    % Using atan2d(vel_y,vel_x) gave correlation ~0.999 with yaw - redundant.
    % Delta-position introduces realistic quantisation lag and larger noise.
    if all(isfinite(prev_pos(i,1:2)))
        dp = pos(i,1:2) - prev_pos(i,1:2) + 1e-6;
    else
        dp = vel(i,1:2);   % fallback
    end
    gps_course = atan2d(dp(2), dp(1)) + 0.8*randn;   % coarser than yaw

    gps_fix    = gps.fix(i);
    gps_hdop   = gps.hdop(i);
    gps_nsats  = gps.nsats(i);

    %% ==============================================================
    % GROUP 4 - BAROMETER  (MS5611: ~0.10 m altitude noise 1sigma)
    %% ==============================================================
    h = max(0, pos(i,3));
    baro_press = env.pressure * (1 - 2.2577e-5*h)^5.2559 + 0.6*randn;
    baro_alt   = pos(i,3) + 0.12*randn;
    baro_temp  = env.temperature - 0.0065*h + 0.10*randn;

    %% ==============================================================
    % GROUP 5 - BATTERY MONITOR  (4S LiPo)
    %% ==============================================================
    % Voltage sag from battery failure (Case 4) is pre-computed in
    % anomaly_injection_engine and written into motor.volt_sag.
    % Reading it here is safe - it is a derived physical quantity
    % (Ohmic sag = I*R), NOT the raw fault intensity scalar.
    v_sag = 0;
    if isfield(motor, 'volt_sag')
        v_sag = motor.volt_sag(i);
    end

    volt       = batt.volt(i) - v_sag + 0.04*randn;
    % Allow sag below nominal 13.2 V floor so battery failure is visible
    volt       = max(10.0, min(16.8, volt));
    curr       = batt.curr(i) + 0.2*randn;
    curr_total = batt.mah(i);
    batt_pct   = batt.pct(i);

    %% ==============================================================
    % GROUP 6 - MOTORS / RCOU  (ESC telemetry)
    %% ==============================================================
    pwm1 = max(1100, min(1900, motor.pwm(i,1) + 2*randn));
    pwm2 = max(1100, min(1900, motor.pwm(i,2) + 2*randn));
    pwm3 = max(1100, min(1900, motor.pwm(i,3) + 2*randn));
    pwm4 = max(1100, min(1900, motor.pwm(i,4) + 2*randn));

    rpm1 = max(0, motor.rpm(i,1) + 18*randn);
    rpm2 = max(0, motor.rpm(i,2) + 18*randn);
    rpm3 = max(0, motor.rpm(i,3) + 18*randn);
    rpm4 = max(0, motor.rpm(i,4) + 18*randn);

    mcurr1 = max(0, motor.curr(i,1) + 0.08*randn);
    mcurr2 = max(0, motor.curr(i,2) + 0.08*randn);
    mcurr3 = max(0, motor.curr(i,3) + 0.08*randn);
    mcurr4 = max(0, motor.curr(i,4) + 0.08*randn);

    %% ==============================================================
    % GROUP 7 - VIBRATION  (VIBE log: ~m/s2 clipping metric)
    % Normal level: 5-15 m/s2 from motors; propeller damage spikes it.
    % Prop damage (Case 6) injects RPM oscillation in motor_inj, which
    % shows up naturally in rpm_asym below - no separate overlay needed.
    %% ==============================================================
    avg_rpm   = mean([rpm1 rpm2 rpm3 rpm4]);
    rpm_asym  = std([rpm1 rpm2 rpm3 rpm4]);   % imbalance -> vibration

    vibe_base = 6.0 + (avg_rpm/10000)^2 * 4;
    % CRITICAL FIX: divisor /300 -> /15.
    % With /300, prop-damage rpm_asym ~500 RPM only added ~1.5 m/s2 of
    % vibe - far below VIBE_THRESH=25. With /15, peak vibe_asym ~ 33 m/s2
    % (clearly above threshold). Normal noise floor: 18 RPM / 15 ~ 1.2 m/s2.
    vibe_asym = rpm_asym / 15;

    vibe_x = max(0, vibe_base + vibe_asym + 0.8*abs(randn));
    vibe_y = max(0, vibe_base + vibe_asym + 0.8*abs(randn));
    vibe_z = max(0, vibe_base*0.7 + vibe_asym*0.6 + 0.6*abs(randn));

    %% ==============================================================
    % GROUP 8 — KINEMATICS
    %% ==============================================================
    groundspeed = norm(vel(i,1:2)) + 0.05*randn;
    airspeed    = norm(vel(i,:) - wind_vec) + 0.08*randn;
    vert_speed  = vel(i,3) + 0.05*randn;
    turn_rate   = abs(gyr_z);
    curvature   = turn_rate / (groundspeed + 0.1);

    %% ==============================================================
    % GROUP 9 — SWARM TOPOLOGY
    %% ==============================================================
    dists = sqrt(sum((pos - pos(i,:)).^2, 2));
    dists(i) = inf;

    neighbor_count  = sum(dists < 12);
    sep_min         = min(dists);
    d_finite        = dists(~isinf(dists));
    formation_score = exp(-std(d_finite));
    centroid_dist   = norm(pos(i,:) - mean(pos));
    leader_dist     = norm(pos(i,:) - pos(leader_id,:));

    %% ==============================================================
    % GROUP 10 — COMMUNICATION  (2.4 GHz MAVLink link)
    % RSSI model: free-space path loss + RF interference
    %% ==============================================================
    rf_eff = env.rf_field(pos(i,:));

    % Base RSSI: -35 dBm at 1 m, -6 dB per doubling of distance
    % Comms attack (Case 7) is encoded entirely in env.rf_field via the
    % Gaussian jammer closure — no separate rssi_inj/loss_inj overlay.
    rssi = -35 - 20*log10(max(1, leader_dist)) + rf_eff + 1.5*randn;
    rssi = max(-120, min(-20, rssi));

    noise_floor = -95;   % dBm typical
    snr = max(0, rssi - noise_floor + randn);

    latency     = max(1, 10 + 0.3*leader_dist + 0.4*abs(rf_eff) + 1.2*randn);
    packet_loss = max(0, 0.4 + abs(rf_eff)*0.25 + 0.6*randn);
    packet_loss = min(100, packet_loss);

    % Burst noise events (natural)
    if rand < 0.02, packet_loss = packet_loss + 8 + 5*rand; end
    if rand < 0.03, latency     = latency + 35 + 20*rand;   end

    link_quality = max(0, min(100, 100 - packet_loss*1.8 - max(0,-rssi-70)*0.3));

    %% ==============================================================
    % GROUP 11 — ENVIRONMENT (from met sensors on GCS)
    %% ==============================================================
    wind_speed_s = env.wind_speed + 0.1*randn;
    wind_dir_s   = mod(env.wind_dir + 0.5*randn, 360);

    %% ==============================================================
    % ASSEMBLE
    %% ==============================================================
    vec = [ ...
        acc_x acc_y acc_z gyr_x gyr_y gyr_z imu_temp ...                  % 1-7
        roll pitch yaw des_roll des_pitch heading_err ...                  % 8-13
        gps_lat gps_lon gps_alt gps_speed gps_course gps_fix gps_hdop gps_nsats ... % 14-21
        baro_alt baro_press baro_temp ...                                  % 22-24
        volt curr curr_total batt_pct ...                                  % 25-28
        pwm1 pwm2 pwm3 pwm4 rpm1 rpm2 rpm3 rpm4 mcurr1 mcurr2 mcurr3 mcurr4 ... % 29-40
        vibe_x vibe_y vibe_z ...                                          % 41-43
        groundspeed airspeed vert_speed turn_rate curvature ...            % 44-48
        neighbor_count sep_min formation_score centroid_dist leader_dist ... % 49-53
        rssi snr latency packet_loss link_quality ...                      % 54-58
        wind_speed_s wind_dir_s ...                                        % 59-60
    ];

    % SAFETY CHECK
    if length(vec) ~= 60 || any(~isfinite(vec))
        warning("Feature corruption at drone %d", i);
        vec = zeros(1,60);   % fallback safe row
    end

    features(i,:) = vec;
end

end
