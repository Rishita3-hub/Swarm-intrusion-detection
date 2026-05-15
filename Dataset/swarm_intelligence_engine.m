%% ================================================================
% FILE: swarm_intelligence_engine.m
% DISTRIBUTED ANOMALY DETECTION + LEADER ELECTION
%
% Enhancements:
%   - Multi-criteria leader health score (speed, HDOP, RSSI, battery,
%     vibration, packet-loss) — not just 3 metrics
%   - Immediate leader re-election when leader is RTB or severely faulty
%   - Candidate blacklist: RTB drones cannot become leader
%   - Smooth leader-handover: new leader announced to all drones
%   - Recovery respects RTB state (does not recover into normal if RTB)
%
% Feature index map (must match feature_engine.m):
%   F_GPS_HDOP  = 20   (gps_hdop)
%   F_RSSI      = 54   (rssi)
%   F_VIBE_X    = 41   (vibe_x)
%   F_BATT_PCT  = 28   (batt_pct)
%   F_PKT_LOSS  = 57   (packet_loss)
%   F_VOLT      = 25   (volt)
%% ================================================================

function [drone_state, response_type, leader_id, vel] = swarm_intelligence_engine( ...
    pos, vel, features, drone_state)

NUM_DRONES = size(pos,1);
response_type = zeros(NUM_DRONES,1);

%% Feature column indices (aligned with feature_engine.m groups)
F_GPS_HDOP  = 20;   % gps_hdop
F_RSSI      = 54;   % rssi (dBm)
F_VIBE_X    = 41;   % vibe_x (m/s²)
F_BATT_PCT  = 28;   % batt_pct (%)
F_PKT_LOSS  = 57;   % packet_loss (%)
F_VOLT      = 25;   % volt (V)

%% Detection thresholds
VEL_THRESH  = 4.0;    % m/s
HDOP_THRESH = 3.5;
RSSI_THRESH = -85;    % dBm
VIBE_THRESH = 25;     % m/s²
BATT_THRESH = 20;     % % — anomaly detection (very low)
LOSS_THRESH = 40;     % %
BATT_RTB    = 25;     % % — RTB threshold (must match swarm_engine)

%% Persistent state
persistent recovery_timer current_leader leader_lock_timer

if isempty(recovery_timer),   recovery_timer   = zeros(NUM_DRONES,1); end
if isempty(current_leader),   current_leader   = 1;                   end
if isempty(leader_lock_timer), leader_lock_timer = 0;                 end

if length(recovery_timer) ~= NUM_DRONES
    recovery_timer = zeros(NUM_DRONES,1);
end

leader = current_leader;

%% ═══════════════════════════════════════════════════════════════
% LEADER HEALTH ASSESSMENT
% A leader is "bad" when ANY critical metric is badly degraded.
% Additional check: if the leader is in RTB mode it should hand off
% (its battery is too low to keep leading the mission).
%% ═══════════════════════════════════════════════════════════════
leader_speed  = norm(vel(leader,:));
leader_hdop   = features(leader, F_GPS_HDOP);
leader_rssi   = features(leader, F_RSSI);
leader_batt   = features(leader, F_BATT_PCT);
leader_vibe   = features(leader, F_VIBE_X);
leader_loss   = features(leader, F_PKT_LOSS);

% Hard failure criteria
leader_critical = ...
    (leader_speed > VEL_THRESH*1.8) || ...
    (leader_hdop  > HDOP_THRESH*1.8) || ...
    (leader_rssi  < RSSI_THRESH - 15) || ...
    (leader_batt  < BATT_RTB) || ...           % ← RTB handoff
    (leader_vibe  > VIBE_THRESH*1.5) || ...
    (leader_loss  > LOSS_THRESH*1.5);

%% ═══════════════════════════════════════════════════════════════
% LEADER ELECTION
% Runs immediately on critical failure; locked out for 30 steps
% after a successful election to prevent thrashing.
%% ═══════════════════════════════════════════════════════════════
if leader_lock_timer > 0
    leader_lock_timer = leader_lock_timer - 1;
end

if leader_critical && leader_lock_timer == 0
    %% Composite health score (lower = better candidate)
    %  Penalise: distance from centroid, speed, HDOP, RSSI loss,
    %            vibration, packet loss. Disqualify RTB drones.
    centroid = mean(pos);
    scores   = inf(NUM_DRONES,1);

    for k = 1:NUM_DRONES
        if k == leader, continue; end   % current (bad) leader excluded

        batt_k = features(k, F_BATT_PCT);
        if batt_k < BATT_RTB, continue; end   % RTB drones not eligible

        hdop_k  = features(k, F_GPS_HDOP);
        rssi_k  = features(k, F_RSSI);
        vibe_k  = features(k, F_VIBE_X);
        loss_k  = features(k, F_PKT_LOSS);

        scores(k) = ...
            norm(pos(k,:) - centroid) * 0.5 ...      % proximity to centre
          + norm(vel(k,:)) * 1.0 ...                 % stability (low speed)
          + max(0, hdop_k - 1.0) * 4.0 ...           % GPS quality
          + max(0, -rssi_k - 60) * 0.3 ...           % comms quality
          + max(0, vibe_k - 8) * 0.5 ...             % mechanical health
          + loss_k * 0.4 ...                          % link quality
          + (100 - batt_k) * 0.2;                    % battery reserves
    end

    [best_score, new_leader] = min(scores);

    if isfinite(best_score)
        % Handover: inform all drones by updating persistent state
        current_leader    = new_leader;
        leader_lock_timer = 30;   % lock for 3 s at 10 Hz
        fprintf('[t] Leader handover: drone %d → drone %d  (score %.2f)\n', ...
                leader, new_leader, best_score);
    else
        % No healthy candidate found — keep current leader (failsafe)
        fprintf('[t] WARNING: No healthy leader candidate — keeping drone %d\n', leader);
    end
end

leader_id = current_leader;
leader    = leader_id;   % refresh for per-drone loop

%% ═══════════════════════════════════════════════════════════════
% PER-DRONE LOCAL DETECTION + CONSENSUS + RESPONSE
%% ═══════════════════════════════════════════════════════════════
for i = 1:NUM_DRONES

    speed    = norm(vel(i,:));
    hdop     = features(i, F_GPS_HDOP);
    rssi     = features(i, F_RSSI);
    vibe     = features(i, F_VIBE_X);
    batt_pct = features(i, F_BATT_PCT);
    pkt_loss = features(i, F_PKT_LOSS);

    %% ── Local anomaly detection ──────────────────────────────────
    local_flag = (speed    > VEL_THRESH)   || ...
                 (hdop     > HDOP_THRESH)  || ...
                 (rssi     < RSSI_THRESH)  || ...
                 (vibe     > VIBE_THRESH)  || ...
                 (batt_pct < BATT_THRESH)  || ...
                 (pkt_loss > LOSS_THRESH);

    %% ── Swarm consensus vote ─────────────────────────────────────
    dists     = sqrt(sum((pos - pos(i,:)).^2, 2));
    neighbors = find(dists < 15 & dists > 0);
    votes     = 0;
    for j = neighbors'
        if norm(vel(j,:)) > VEL_THRESH || ...
           features(j, F_GPS_HDOP) > HDOP_THRESH || ...
           features(j, F_RSSI) < RSSI_THRESH
            votes = votes + 1;
        end
    end

    % Trigger fault state (threshold raised to ≥ 3 to cut false positives)
    if local_flag && votes >= 3
        drone_state(i) = 1;
    end

    %% ── Graded response ─────────────────────────────────────────
    if drone_state(i) == 1

        if rssi < RSSI_THRESH
            response_type(i) = 1;   % comms degraded
            vel(i,:) = vel(i,:) * 0.80;

        elseif hdop > HDOP_THRESH
            response_type(i) = 2;   % GPS degraded
            vel(i,:) = vel(i,:) * 0.70;

        elseif vibe > VIBE_THRESH
            response_type(i) = 3;   % mechanical fault → emergency
            vel(i,:) = vel(i,:) * 0.55;

        else
            response_type(i) = 4;   % generic
            vel(i,:) = vel(i,:) * 0.65;
        end

        drone_state(i)    = 2;
        recovery_timer(i) = 0;
    end

    %% ── Recovery + rejoin ────────────────────────────────────────
    if drone_state(i) == 2

        vel(i,:) = vel(i,:) * 0.98;
        recovery_timer(i) = recovery_timer(i) + 1;

        % Curved rejoin trajectory toward current leader
        to_leader = pos(leader_id,:) - pos(i,:);
        if norm(to_leader) > 0
            dir  = to_leader / norm(to_leader);
        else
            dir  = [0 0 0];
        end
        perp = [-dir(2), dir(1), 0];

        vel(i,:) = vel(i,:) + 0.08*dir + 0.03*perp;

        % Recovery criterion — healthy sensors, minimum 15 steps
        % Skip recovery if drone is RTB (battery-low takes priority)
        if recovery_timer(i) > 15 && ...
           hdop < 2.0 && rssi > RSSI_THRESH && ...
           batt_pct > BATT_RTB
            drone_state(i)    = 0;
            recovery_timer(i) = 0;
        end
    end

end

end