"""MITRE ATT&CK-based attacker agent for HMARL-SOC environment."""

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .network import EnterpriseNetwork, Host


# ATT&CK phases aligned with paper
ATTACK_PHASES = {
    0: {"name": "Reconnaissance", "tactic": "T1595", "base_prob": 0.8},
    1: {"name": "InitialAccess", "tactic": "T1190", "base_prob": 0.5},
    2: {"name": "Execution", "tactic": "T1059", "base_prob": 0.6},
    3: {"name": "LateralMovement", "tactic": "T1021", "base_prob": 0.4},
    4: {"name": "Exfiltration", "tactic": "T1048", "base_prob": 0.3},
}


class Attacker:
    """
    Simulates a multi-stage APT campaign following MITRE ATT&CK framework.
    Progresses: Reconnaissance -> Initial Access -> Execution -> 
                Lateral Movement -> Exfiltration.
    """
    
    def __init__(self, network: 'EnterpriseNetwork', 
                 stealth: float = 0.7,
                 rng: Optional[np.random.Generator] = None):
        self.network = network
        self.stealth = stealth  # 0-1, higher = harder to detect
        self.rng = rng or np.random.default_rng()
        
        self.active = False
        self.current_phase: int = 0
        self.target_hosts: List['Host'] = []
        self.compromised_hosts: List['Host'] = []
        self.entry_segment: int = 0  # Start from DMZ
        self.steps_in_phase: int = 0
        self.total_steps: int = 0
        self.exfiltrated: bool = False
    
    def reset(self):
        """Reset attacker state for new episode."""
        self.active = True
        self.current_phase = 0
        self.target_hosts = []
        self.compromised_hosts = []
        self.entry_segment = 0  # Always start from DMZ
        self.steps_in_phase = 0
        self.total_steps = 0
        self.exfiltrated = False
    
    def step(self) -> dict:
        """Execute one step of the attack campaign. Returns attack info."""
        if not self.active:
            return {"action": "inactive", "phase": -1}
        
        self.total_steps += 1
        self.steps_in_phase += 1
        
        phase_info = ATTACK_PHASES[self.current_phase]
        result = {
            "action": phase_info["name"],
            "phase": self.current_phase,
            "tactic": phase_info["tactic"],
            "success": False,
            "new_compromises": [],
            "noise_level": 1.0 - self.stealth,
        }
        
        # Execute current phase
        if self.current_phase == 0:
            result = self._do_reconnaissance(result)
        elif self.current_phase == 1:
            result = self._do_initial_access(result)
        elif self.current_phase == 2:
            result = self._do_execution(result)
        elif self.current_phase == 3:
            result = self._do_lateral_movement(result)
        elif self.current_phase == 4:
            result = self._do_exfiltration(result)
        
        # Phase transition
        if result["success"] and self.steps_in_phase >= 3:
            if self.current_phase < 4:
                self.current_phase += 1
                self.steps_in_phase = 0
        
        return result
    
    def _do_reconnaissance(self, result: dict) -> dict:
        """Phase 0: Scan for vulnerable targets in entry segment."""
        segment = self.network.segments[self.entry_segment]
        vulnerable = [h for h in segment.hosts if not h.compromised and not h.isolated]
        
        if vulnerable:
            # Sort by vulnerability (attacker prefers most vulnerable)
            vulnerable.sort(key=lambda h: h.vulnerability, reverse=True)
            self.target_hosts = vulnerable[:5]  # Select top-5 targets
            result["success"] = True
        
        return result
    
    def _do_initial_access(self, result: dict) -> dict:
        """Phase 1: Exploit vulnerable targets."""
        for host in self.target_hosts:
            if host.isolated:
                continue
            prob = ATTACK_PHASES[1]["base_prob"] * host.vulnerability
            if self.rng.random() < prob:
                host.compromised = True
                host.compromise_stage = 1
                self.compromised_hosts.append(host)
                result["new_compromises"].append(host.host_id)
                result["success"] = True
        
        return result
    
    def _do_execution(self, result: dict) -> dict:
        """Phase 2: Execute payloads on compromised hosts."""
        for host in self.compromised_hosts:
            if host.isolated or host.compromise_stage >= 2:
                continue
            prob = ATTACK_PHASES[2]["base_prob"]
            if self.rng.random() < prob:
                host.compromise_stage = 2
                result["success"] = True
        
        return result
    
    def _do_lateral_movement(self, result: dict) -> dict:
        """Phase 3: Move to adjacent segments."""
        current_segments = set(h.segment_id for h in self.compromised_hosts if not h.isolated)
        
        for seg_id in list(current_segments):
            reachable = self.network.get_reachable_segments(seg_id)
            for target_seg_id in reachable:
                target_seg = self.network.segments[target_seg_id]
                candidates = [h for h in target_seg.hosts if not h.compromised and not h.isolated]
                
                if candidates:
                    target = candidates[self.rng.integers(len(candidates))]
                    prob = ATTACK_PHASES[3]["base_prob"] * target.vulnerability
                    if self.rng.random() < prob:
                        target.compromised = True
                        target.compromise_stage = 3
                        self.compromised_hosts.append(target)
                        result["new_compromises"].append(target.host_id)
                        result["success"] = True
        
        return result
    
    def _do_exfiltration(self, result: dict) -> dict:
        """Phase 4: Exfiltrate data from high-value segments."""
        high_value_compromised = [
            h for h in self.compromised_hosts
            if not h.isolated and h.compromise_stage >= 3
            and self.network.segments[h.segment_id].value >= 0.9
        ]
        
        if high_value_compromised:
            for host in high_value_compromised:
                host.compromise_stage = 4
            prob = ATTACK_PHASES[4]["base_prob"]
            if self.rng.random() < prob:
                self.exfiltrated = True
                result["success"] = True
        
        return result
    
    def get_observation_noise(self) -> float:
        """Return noise level for generating alerts (higher = more visible)."""
        phase_noise = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.4, 4: 0.6}
        return phase_noise.get(self.current_phase, 0.1) * (1.0 - self.stealth)
