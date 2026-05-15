%% ================================================================
% FILE: swarm_engine.m
% QUADROTOR SWARM PHYSICS  (returns updated motor + battery state)
%
% Enhancements:
%   - Leader follows a multi-waypoint surveillance lawnmower path
%   - Followers maintain formation behind leader (offset by angle)
%   - Battery-low RTB: any drone below BATT_RTB_THRESH flies home
%   - Leader death / RTB → instant new leader election (handled by
%     swarm_intelligence_engine; swarm_engine respects leader_id)
%
% Physical model:
%   Airframe  : 450-class quadrotor, mass = 1.5 kg
%   Motors    : 920 KV brushless, 10-inch props
%   Battery   : 4S LiPo 5000 mAh, R_int = 0.05 Ω
%   Thrust    : T = Ct * ρ * (n/60)² * D⁴   (n in RPM)
%   Drag      : F_d = ½ ρ Cd A v²
%% ================================================================

function [pos, vel, motor, batt] = swarm_engine(pos, vel, motor, batt, env, leader_id)

NUM_DRONES = size(pos,1);
dt         = 0.1;    % simulation time step (s) → 10 Hz

%% ── Physical constants ──────────────────────────────────────────
MASS       = 1.5;     % kg
GRAVITY    = 9.81;    % m/s²
RHO        = 1.225;   % kg/m³ (sea level)
CT         = 9.5e-6;  % thrust coefficient (fitted to 10-inch prop)
KV         = 920;     % motor KV rating
PROP_D     = 0.254;   % prop diameter (m) — 10 inch
CD_BODY    = 0.50;    % body drag coefficient
A_FRONTAL  = 0.04;    % frontal area (m²)

RPM_HOVER  = 60 * sqrt(MASS*GRAVITY / (4 * CT * RHO * PROP_D^4));
PWM_HOVER  = 1000 + (RPM_HOVER / (KV*16.8)) * 1000;

%% ── Mission & RTB constants ─────────────────────────────────────
CRUISE_ALT   = 15;      % m — nominal patrol altitude
BASE_POS     = [0, 0, 0];   % RTB landing target (ground)
RTB_ALT      = 12;      % m — approach altitude during RTB glide
BATT_RTB_THRESH = 25;   % % SoC — trigger return-to-base
BATT_LAND_THRESH = 10;  % % SoC — force land wherever you are

%% ── Surveillance waypoints (lawnmower grid across base) ─────────
% 5-column lawnmower: X sweeps ±80 m, Y steps 40 m each pass
% Altitude varies slightly between legs for 3-D realism
WP = [ ...
   -80, -80, CRUISE_ALT;
    80, -80, CRUISE_ALT+3;
    80, -40, CRUISE_ALT;
   -80, -40, CRUISE_ALT+3;
   -80,   0, CRUISE_ALT;
    80,   0, CRUISE_ALT+3;
    80,  40, CRUISE_ALT;
   -80,  40, CRUISE_ALT+3;
   -80,  80, CRUISE_ALT;
    80,  80, CRUISE_ALT+3;
];
NUM_WP = size(WP,1);
WP_ACCEPT_RAD = 6;   % m — waypoint capture radius

%% ── Persistent state ────────────────────────────────────────────
persistent wp_idx rtb_flag
if isempty(wp_idx),  wp_idx  = 1;                      end
if isempty(rtb_flag),rtb_flag = zeros(NUM_DRONES,1);   end

% Resize if drone count changed (shouldn't in normal use)
if length(rtb_flag) ~= NUM_DRONES
    rtb_flag = zeros(NUM_DRONES,1);
end

%% ── Wind vector ─────────────────────────────────────────────────
wind_vec = env.wind_speed * [cosd(env.wind_dir), sind(env.wind_dir), 0];

%% ── Velocity damping (aerodynamic friction proxy) ───────────────
vel = vel * 0.96;

L         = leader_id;
followers = setdiff(1:NUM_DRONES, L);

%% ═══════════════════════════════════════════════════════════════
% BATTERY-LOW RTB FLAG UPDATE
% Any drone whose SoC drops below threshold is flagged for RTB.
% Once flagged it stays flagged until it lands (handled externally)
% or battery recovers above threshold + hysteresis.
%% ═══════════════════════════════════════════════════════════════
for i = 1:NUM_DRONES
    if batt.pct(i) < BATT_RTB_THRESH
        rtb_flag(i) = 1;
    elseif batt.pct(i) > BATT_RTB_THRESH + 5   % hysteresis band
        rtb_flag(i) = 0;
    end
end

%% ═══════════════════════════════════════════════════════════════
% LEADER — follows surveillance waypoint mission
%          unless battery-low → RTB
%% ═══════════════════════════════════════════════════════════════
if rtb_flag(L)
    %% Leader RTB path: climb to RTB_ALT, fly to base, descend
    rtb_target = BASE_POS + [0, 0, RTB_ALT];
    if norm(pos(L,1:2) - BASE_POS(1:2)) < 5   % over base — descend
        rtb_target(3) = 0.5;
    end
    vel(L,:) = vel(L,:) + 0.18 * (rtb_target - pos(L,:));
else
    %% Normal mission: navigate to current waypoint
    target_wp = WP(wp_idx,:);
    dist_to_wp = norm(pos(L,:) - target_wp);

    % Advance waypoint when within capture radius
    if dist_to_wp < WP_ACCEPT_RAD
        wp_idx = mod(wp_idx, NUM_WP) + 1;
        target_wp = WP(wp_idx,:);
    end

    vel(L,:) = vel(L,:) + 0.14 * (target_wp - pos(L,:));
    vel(L,:) = vel(L,:) + 0.04 * wind_vec;
end

%% ═══════════════════════════════════════════════════════════════
% FOLLOWERS — formation + collision avoidance + RTB
%% ═══════════════════════════════════════════════════════════════
% Formation: ring of radius 22 m around the LEADER's CURRENT position
% (not a fixed world point). Each follower also keeps a slight altitude
% stagger so the ring is a 3-D helix for visual realism.
radius = 22;   % formation ring radius (m) — wider spread for surveillance coverage

for k = 1:length(followers)
    i = followers(k);

    if batt.pct(i) < BATT_LAND_THRESH
        %% Emergency land in place — cut thrust gently
        vel(i,:) = vel(i,:) * 0.85;
        vel(i,3) = vel(i,3) - 0.6;   % controlled descent
        continue
    end

    if rtb_flag(i)
        %% Low-battery RTB for follower
        rtb_target = BASE_POS + [0, 0, RTB_ALT];
        if norm(pos(i,1:2) - BASE_POS(1:2)) < 5
            rtb_target(3) = 0.5;
        end
        vel(i,:) = vel(i,:) + 0.18 * (rtb_target - pos(i,:));

    else
        %% Normal formation following
        angle    = 2*pi*(k-1) / (NUM_DRONES-1);
        alt_stagger = 1.5 * sin(angle);   % ±1.5 m altitude spread in ring
        desired  = pos(L,:) + [radius*cos(angle), radius*sin(angle), alt_stagger];

        % Formation attraction toward desired slot (softer pull for smooth motion)
        vel(i,:) = vel(i,:) + 0.10 * (desired - pos(i,:));

        % Slight randomness — prevents drones looking "locked on rails"
        vel(i,:) = vel(i,:) + 0.02 * randn(1,3);

        % Altitude hold relative to leader
        vel(i,3) = vel(i,3) + 0.10 * (pos(L,3) + alt_stagger - pos(i,3));
    end

    %% Collision avoidance (all drones, whether RTB or not)
    for j = 1:NUM_DRONES
        if j ~= i
            dvec = pos(i,:) - pos(j,:);
            d    = norm(dvec);
            if d < 10 && d > 0
                vel(i,:) = vel(i,:) + 3.0 * (dvec / (d + 1e-6));
            end
        end
    end

    %% Velocity consensus with neighbours (formation only — skip RTB drones)
    if ~rtb_flag(i)
        nbrs     = sqrt(sum((pos - pos(i,:)).^2, 2)) < 12;
        vel_avg  = mean(vel(nbrs,:), 1);
        vel(i,:) = vel(i,:) + 0.08 * (vel_avg - vel(i,:));
        vel(i,:) = vel(i,:) + 0.06 * wind_vec;
    end
end

%% ═══════════════════════════════════════════════════════════════
% GLOBAL COHESION  (only for non-RTB drones, keeps swarm together)
%% ═══════════════════════════════════════════════════════════════
active_mask = find(~rtb_flag);
if length(active_mask) > 1
    centroid = mean(pos(active_mask,:));
    for i = active_mask'
        vel(i,:) = vel(i,:) + 0.03 * (centroid - pos(i,:));
    end
end

%% ═══════════════════════════════════════════════════════════════
% AERODYNAMIC DRAG  (body drag in relative wind frame)
%% ═══════════════════════════════════════════════════════════════
for i = 1:NUM_DRONES
    v_rel = vel(i,:) - wind_vec;
    v_mag = norm(v_rel);
    if v_mag > 0.05
        F_drag = -0.5 * RHO * CD_BODY * A_FRONTAL * v_mag^2 * (v_rel/v_mag);
        vel(i,:) = vel(i,:) + (F_drag/MASS)*dt;
    end
end

%% ═══════════════════════════════════════════════════════════════
% SPEED LIMITER  (12 m/s nominal; RTB drones allowed slightly faster)
%% ═══════════════════════════════════════════════════════════════
for i = 1:NUM_DRONES
    max_spd = 12 + 3 * rtb_flag(i);   % RTB → allow 15 m/s
    s = norm(vel(i,:));
    if s > max_spd
        vel(i,:) = vel(i,:) * (max_spd/s);
    end
end

%% ═══════════════════════════════════════════════════════════════
% INTEGRATE POSITION
%% ═══════════════════════════════════════════════════════════════
pos = pos + vel * dt;

% Ground clamp (no negative altitude)
pos(:,3) = max(0, pos(:,3));

%% ═══════════════════════════════════════════════════════════════
% MOTOR STATE UPDATE  (ESC / motor dynamics)
%% ═══════════════════════════════════════════════════════════════
for i = 1:NUM_DRONES
    speed_i = norm(vel(i,:));

    alt_err       = CRUISE_ALT - pos(i,3);
    thrust_demand = MASS*GRAVITY + MASS*(0.12*alt_err + 0.04*speed_i);

    rpm_req = 60 * sqrt(max(0, thrust_demand / (4*CT*RHO*PROP_D^4)));
    rpm_req = max(1000, min(12000, rpm_req));

    pwm_req = 1000 + (rpm_req / (KV*16.8)) * 1000;
    pwm_req = max(1100, min(1900, pwm_req));

    ALPHA = 0.65;
    for m = 1:4
        target_m        = rpm_req + 40*randn;
        motor.rpm(i,m)  = (1-ALPHA)*motor.rpm(i,m) + ALPHA*target_m;
        motor.pwm(i,m)  = (1-ALPHA)*motor.pwm(i,m) + ALPHA*pwm_req;
        motor.rpm(i,m)  = max(0, motor.rpm(i,m));

        v_norm          = batt.volt(i) / 14.8;
        motor.curr(i,m) = max(0, (motor.rpm(i,m)/RPM_HOVER)^2 * 5.0 / v_norm ...
                             + 0.25*randn);

        motor.temp(i,m) = motor.temp(i,m) ...
                        + 0.01*(motor.curr(i,m) - 5)*dt ...
                        - 0.003*(motor.temp(i,m) - 25)*dt;
        motor.temp(i,m) = max(20, motor.temp(i,m));
    end
end

%% ═══════════════════════════════════════════════════════════════
% BATTERY DISCHARGE MODEL  (4S LiPo Peukert + internal-R)
%% ═══════════════════════════════════════════════════════════════
for i = 1:NUM_DRONES
    I_pack = sum(motor.curr(i,:)) + 1.5;
    batt.curr(i) = I_pack;

    mah_out      = I_pack * dt / 3.6;
    batt.mah(i)  = batt.mah(i) + mah_out;
    batt.pct(i)  = max(0, 100*(1 - batt.mah(i)/batt.capacity(i)));

    soc         = batt.pct(i)/100;
    V_oc        = 4*(3.3 + 0.9*soc);
    R_int       = 0.05;
    V_term      = V_oc - I_pack * R_int;
    batt.volt(i)= max(13.2, V_term + 0.04*randn);
end

%% ── Expose RTB flag via motor struct for labels ─────────────────
% main_dataset_generator can read motor.rtb_flag for labelling
motor.rtb_flag = rtb_flag;

end