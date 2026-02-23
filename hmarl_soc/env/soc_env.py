"""
Enterprise SOC Gymnasium Environment for HMARL-SOC.
Implements the Dec-POMDP formulation from Section III of the paper.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional, Tuple

from .network import EnterpriseNetwork
from .attacker import Attacker


class SOCEnv(gym.Env):
    """
    Multi-agent SOC defense environment.
    
    Agents:
        - Strategic Coordinator (SC): meta-policy, obs_dim=64
        - Threat Hunter (TH): continuous actions, obs_dim=128
        - Alert Triage (AT): discrete actions, obs_dim=64
        - Response Orchestrator (RO): hybrid actions, obs_dim=96
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: Optional[dict] = None, seed: int = 42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        config = config or {}
        
        # Environment params
        self.hosts_per_segment = config.get("hosts_per_segment", 40)
        self.max_steps = config.get("max_steps", 200)
        self.num_segments = 5
        
        # Build network
        self.network = EnterpriseNetwork(
            hosts_per_segment=self.hosts_per_segment, rng=self.rng
        )
        self.attacker = Attacker(self.network, stealth=0.7, rng=self.rng)
        
        # Agent observation/action dims (matching paper)
        self.obs_dims = {"sc": 64, "th": 128, "at": 64, "ro": 96}
        self.action_dims = {"sc": 8, "th": 16, "at": 4, "ro": 12}
        
        # Alert queue
        self.alert_queue = []
        self.max_alerts = 50
        
        # Metrics
        self.step_count = 0
        self.detection_time = None
        self.containment_time = None
        self.false_positives = 0
        self.true_positives = 0
        self.total_alerts_generated = 0
        self.total_scans = 0  # total hosts scanned (for FPR denominator)
        self.attack_contained = False
        self.sc_directive = np.zeros(8, dtype=np.float32)  # SC action cache
    
    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], dict]:
        """Reset environment for new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.network.reset()
        self.attacker.reset()
        self.alert_queue = []
        self.step_count = 0
        self.detection_time = None
        self.containment_time = None
        self.false_positives = 0
        self.true_positives = 0
        self.total_alerts_generated = 0
        self.total_scans = 0
        self.attack_contained = False
        self.sc_directive = np.zeros(8, dtype=np.float32)
        
        obs = self._get_observations()
        info = {"step": 0}
        return obs, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray], float, bool, bool, dict
    ]:
        """
        Execute one environment step.
        
        Args:
            actions: dict with keys 'sc', 'th', 'at', 'ro'
        
        Returns:
            observations, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # 1. Attacker acts
        attack_result = self.attacker.step()
        
        # 2. Generate alerts (from attacker noise + legitimate traffic)
        self._generate_alerts(attack_result)
        
        # 3. Process SC directive (modulates other agents)
        sc_action = actions.get("sc")
        if sc_action is not None:
            self.sc_directive = np.clip(np.asarray(sc_action, dtype=np.float32), -1, 1)
        
        # 4. Process agent actions (influenced by SC directive)
        th_result = self._process_threat_hunter(actions.get("th"))
        at_result = self._process_alert_triage(actions.get("at"))
        ro_result = self._process_response(actions.get("ro"))
        
        # 4. Calculate reward
        reward = self._compute_reward(attack_result, th_result, at_result, ro_result)
        
        # 5. Check termination
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_steps
        
        # 6. Get new observations
        obs = self._get_observations()
        
        info = {
            "step": self.step_count,
            "attack_phase": self.attacker.current_phase,
            "compromised": self.network.total_compromised,
            "detected": self.network.total_detected,
            "isolated": self.network.total_isolated,
            "false_positives": self.false_positives,
            "true_positives": self.true_positives,
            "detection_time": self.detection_time,
            "containment_time": self.containment_time,
            "exfiltrated": self.attacker.exfiltrated,
            "contained": self.attack_contained,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Generate per-agent observations (partial observability)."""
        net_state = self.network.get_state_vector()  # 20-dim
        
        # SC: aggregated view (64-dim)
        sc_obs = np.zeros(self.obs_dims["sc"], dtype=np.float32)
        sc_obs[:20] = net_state
        sc_obs[20] = self.step_count / self.max_steps
        sc_obs[21] = len(self.alert_queue) / self.max_alerts
        sc_obs[22] = self.network.total_compromised / self.network.num_hosts
        sc_obs[23] = self.network.total_detected / max(1, self.network.total_compromised + 1)
        # Add noise for partial observability
        sc_obs[24:] = self.rng.normal(0, 0.1, self.obs_dims["sc"] - 24)
        
        # TH: network flow + endpoint features (128-dim)
        th_obs = np.zeros(self.obs_dims["th"], dtype=np.float32)
        th_obs[:20] = net_state
        # Simulated flow statistics per segment
        for i in range(self.num_segments):
            seg = self.network.segments[i]
            base = 20 + i * 10
            th_obs[base] = seg.compromise_ratio
            th_obs[base + 1] = self.attacker.get_observation_noise() if self.attacker.active else 0
            th_obs[base + 2:base + 10] = self.rng.normal(0, 0.2, 8)  # flow features
        # Threat intel features
        th_obs[70:80] = self.rng.normal(0, 0.1, 10)
        # Endpoint behavioral features (noisy)
        for i, host in enumerate(self.network.all_hosts[:48]):
            th_obs[80 + i] = (host.compromise_stage / 5.0) * (1 - self.attacker.stealth) + \
                             self.rng.normal(0, 0.15)
        
        # AT: alert attributes (64-dim)
        at_obs = np.zeros(self.obs_dims["at"], dtype=np.float32)
        at_obs[0] = len(self.alert_queue) / self.max_alerts
        for i, alert in enumerate(self.alert_queue[:10]):
            base = 1 + i * 6
            if base + 6 <= self.obs_dims["at"]:
                at_obs[base] = alert["severity"]
                at_obs[base + 1] = alert["confidence"]
                at_obs[base + 2] = alert["segment_id"] / self.num_segments
                at_obs[base + 3] = float(alert["is_true_positive"])
                at_obs[base + 4] = alert["timestamp"] / self.max_steps
                at_obs[base + 5] = alert["noise"]
        
        # RO: network state + active incidents (96-dim)
        ro_obs = np.zeros(self.obs_dims["ro"], dtype=np.float32)
        ro_obs[:20] = net_state
        ro_obs[20] = self.network.total_isolated / self.network.num_hosts
        # Per-segment isolation and risk
        for i in range(self.num_segments):
            seg = self.network.segments[i]
            base = 21 + i * 8
            ro_obs[base] = seg.compromise_ratio
            iso = sum(1 for h in seg.hosts if h.isolated) / len(seg.hosts)
            ro_obs[base + 1] = iso
            ro_obs[base + 2] = seg.value
            ro_obs[base + 3:base + 8] = self.rng.normal(0, 0.1, 5)
        
        return {"sc": sc_obs, "th": th_obs, "at": at_obs, "ro": ro_obs}
    
    def _generate_alerts(self, attack_result: dict):
        """Generate alerts from attack activity and legitimate noise."""
        # Attack-generated alerts (true positives)
        if attack_result["phase"] >= 1:
            noise = self.attacker.get_observation_noise()
            if self.rng.random() < noise * 2:
                alert = {
                    "severity": min(1.0, attack_result["phase"] / 4.0),
                    "confidence": 0.5 + noise * 0.5,
                    "segment_id": self.attacker.entry_segment,
                    "is_true_positive": True,
                    "timestamp": self.step_count,
                    "noise": noise,
                }
                self.alert_queue.append(alert)
                self.total_alerts_generated += 1
        
        # Legitimate noise (false positives)
        if self.rng.random() < 0.3:
            fp_alert = {
                "severity": self.rng.random() * 0.5,
                "confidence": self.rng.random() * 0.4,
                "segment_id": self.rng.integers(self.num_segments),
                "is_true_positive": False,
                "timestamp": self.step_count,
                "noise": self.rng.random() * 0.3,
            }
            self.alert_queue.append(fp_alert)
            self.total_alerts_generated += 1
        
        # Trim queue
        if len(self.alert_queue) > self.max_alerts:
            self.alert_queue = self.alert_queue[-self.max_alerts:]
    
    def _process_threat_hunter(self, action: Optional[np.ndarray]) -> dict:
        """Process Threat Hunter continuous actions. SC directive modulates scan priority."""
        result = {"detected_hosts": [], "false_scans": 0}
        if action is None:
            return result
        
        action = np.clip(action, -1, 1) if isinstance(action, np.ndarray) else np.zeros(self.action_dims["th"])
        
        # SC directive[0:2] modulates scan priority per segment
        sc_scan_boost = (self.sc_directive[0] + 1) / 2 * 0.3  # 0 to 0.3 boost
        sc_scope_boost = (self.sc_directive[1] + 1) / 2 * 0.2  # 0 to 0.2 boost
        
        # Action interpretation: first 5 = investigation intensity per segment,
        # next 5 = scope (depth of scan), rest = exploration params
        for seg_id in range(self.num_segments):
            intensity = (action[seg_id] + 1) / 2 + sc_scan_boost  # SC boosts intensity
            intensity = min(1.0, intensity)
            scope = (action[seg_id + 5] + 1) / 2 + sc_scope_boost if seg_id + 5 < len(action) else 0.5
            scope = min(1.0, scope)
            
            if intensity > 0.3:  # threshold for active hunting
                segment = self.network.segments[seg_id]
                for host in segment.hosts:
                    self.total_scans += 1  # track for FPR
                    if host.compromised and not host.detected:
                        # Detection probability based on intensity, scope, and stealth
                        det_prob = intensity * scope * (1.0 - self.attacker.stealth * 0.5)
                        det_prob *= (host.compromise_stage / 5.0)  # easier to detect later stages
                        if self.rng.random() < det_prob:
                            host.detected = True
                            result["detected_hosts"].append(host.host_id)
                            self.true_positives += 1
                            if self.detection_time is None:
                                self.detection_time = self.step_count
                    elif not host.compromised and intensity > 0.7:
                        # False scan on clean host
                        if self.rng.random() < 0.05:
                            result["false_scans"] += 1
                            self.false_positives += 1
        
        return result
    
    def _process_alert_triage(self, action: Optional[np.ndarray]) -> dict:
        """Process Alert Triage discrete action. SC directive modulates aggressiveness."""
        result = {"triaged": 0, "escalated": 0, "suppressed_tp": 0}
        if action is None or not self.alert_queue:
            return result
        
        # SC directive[2:4] modulates triage aggressiveness
        sc_aggression = (self.sc_directive[2] + 1) / 2  # 0 to 1
        
        # action is index: 0=escalate, 1=suppress, 2=correlate, 3=enrich
        action_idx = int(action) if np.isscalar(action) else int(action[0]) % self.action_dims["at"]
        
        if self.alert_queue:
            alert = self.alert_queue[0]  # process oldest alert
            
            if action_idx == 0:  # Escalate
                result["triaged"] += 1
                result["escalated"] += 1
                if alert["is_true_positive"]:
                    self.true_positives += 1
                else:
                    self.false_positives += 1
                self.alert_queue.pop(0)
            
            elif action_idx == 1:  # Suppress
                result["triaged"] += 1
                if alert["is_true_positive"]:
                    result["suppressed_tp"] += 1  # missed a real alert!
                self.alert_queue.pop(0)
            
            elif action_idx == 2:  # Correlate
                # Group similar alerts (reduce noise)
                correlated = [a for a in self.alert_queue 
                             if a["segment_id"] == alert["segment_id"]]
                if len(correlated) > 1:
                    # Keep the highest severity one
                    best = max(correlated, key=lambda a: a["severity"])
                    for a in correlated:
                        if a is not best:
                            self.alert_queue.remove(a)
                result["triaged"] += len(correlated) - 1
            
            elif action_idx == 3:  # Enrich
                # Add context to alert (improves confidence)
                alert["confidence"] = min(1.0, alert["confidence"] + 0.2)
        
        return result
    
    def _process_response(self, action: Optional[np.ndarray]) -> dict:
        """Process Response Orchestrator actions. SC directive modulates urgency."""
        result = {"isolated": [], "remediated": [], "disruption_cost": 0.0}
        if action is None:
            return result
        
        action = np.clip(action, -1, 1) if isinstance(action, np.ndarray) else np.zeros(self.action_dims["ro"])
        
        # SC directive[4:6] modulates response urgency
        sc_urgency = (self.sc_directive[4] + 1) / 2 * 0.2  # 0 to 0.2 boost
        
        # Actions: first 5 = isolate intensity per segment,
        # next 5 = remediate intensity, last 2 = global params
        for seg_id in range(self.num_segments):
            isolate_intensity = (action[seg_id] + 1) / 2 + sc_urgency
            isolate_intensity = min(1.0, isolate_intensity)
            remediate_intensity = (action[seg_id + 5] + 1) / 2 if seg_id + 5 < len(action) else 0
            
            segment = self.network.segments[seg_id]
            
            # Isolate detected compromised hosts
            if isolate_intensity > 0.5:
                for host in segment.hosts:
                    if host.detected and host.compromised and not host.isolated:
                        if self.rng.random() < isolate_intensity:
                            host.isolated = True
                            result["isolated"].append(host.host_id)
                            result["disruption_cost"] += 0.1 * segment.value
                    elif not host.compromised and isolate_intensity > 0.9:
                        # Over-aggressive isolation of clean hosts
                        if self.rng.random() < 0.02:
                            host.isolated = True
                            result["disruption_cost"] += 0.2 * segment.value
            
            # Remediate isolated hosts
            if remediate_intensity > 0.5:
                for host in segment.hosts:
                    if host.isolated and host.compromised:
                        if self.rng.random() < remediate_intensity:
                            host.compromised = False
                            host.compromise_stage = 0
                            host.isolated = False
                            host.detected = False
                            result["remediated"].append(host.host_id)
        
        # Check containment
        active_compromised = [h for h in self.network.all_hosts 
                              if h.compromised and not h.isolated]
        if not active_compromised and self.network.total_compromised > 0:
            self.attack_contained = True
            if self.containment_time is None:
                self.containment_time = self.step_count
        
        return result
    
    def _compute_reward(self, attack_result, th_result, at_result, ro_result) -> float:
        """
        Composite reward: R = α*R_det + β*R_resp + δ*R_cost + λ*R_fp
        Paper: α=1.0, β=1.5, δ=-0.3, λ=-2.0
        """
        alpha, beta, delta, lam = 1.0, 1.5, -0.3, -2.0
        
        # R_det: reward for true positive detections (proportional to stage severity)
        r_det = 0.0
        for host_id in th_result.get("detected_hosts", []):
            host = self.network.all_hosts[host_id]
            r_det += host.compromise_stage / 5.0  # higher stage = higher reward
        
        # R_resp: reward for timely containment
        r_resp = 0.0
        for host_id in ro_result.get("isolated", []):
            host = self.network.all_hosts[host_id]
            # Inverse of time since detection
            if self.detection_time is not None:
                elapsed = max(1, self.step_count - self.detection_time)
                r_resp += 1.0 / elapsed
            else:
                r_resp += 0.1
        
        # R_cost: operational disruption penalty
        r_cost = ro_result.get("disruption_cost", 0.0)
        
        # R_fp: false positive penalty
        r_fp = (th_result.get("false_scans", 0) + 
                (1 if at_result.get("suppressed_tp", 0) > 0 else 0)) * 0.5
        
        # Bonus for containment
        if self.attack_contained:
            r_resp += 5.0
        
        # Penalty for exfiltration
        if self.attacker.exfiltrated:
            r_det -= 10.0
        
        reward = alpha * r_det + beta * r_resp + delta * r_cost + lam * r_fp
        return float(reward)
    
    def _check_terminated(self) -> bool:
        """Episode ends if attack contained or exfiltrated."""
        if self.attack_contained:
            return True
        if self.attacker.exfiltrated:
            return True
        return False
    
    def get_metrics(self) -> dict:
        """Return SOC performance metrics."""
        # FPR = FP / (FP + TN), capped at 1.0
        # TN approximated as total_scans - true_positives - false_positives
        fp = self.false_positives
        total_neg = max(1, self.total_scans - self.true_positives)
        fpr = min(1.0, fp / total_neg) if total_neg > 0 else 0.0
        
        return {
            "mttd": self.detection_time if self.detection_time else self.max_steps,
            "mttr": self.containment_time if self.containment_time else self.max_steps,
            "fpr": fpr,
            "csr": float(self.attack_contained),
            "compromised": self.network.total_compromised,
            "exfiltrated": self.attacker.exfiltrated,
        }
