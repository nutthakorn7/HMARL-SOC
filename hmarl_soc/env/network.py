"""Enterprise network simulation for HMARL-SOC."""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Host:
    """Represents a single host in the network."""
    host_id: int
    segment_id: int
    vulnerability: float = 0.0  # 0-1 vulnerability score
    compromised: bool = False
    compromise_stage: int = 0   # 0=clean, 1-5=ATT&CK phase
    detected: bool = False
    isolated: bool = False
    
    def reset(self):
        self.compromised = False
        self.compromise_stage = 0
        self.detected = False
        self.isolated = False


@dataclass
class NetworkSegment:
    """Represents a network segment (e.g., DMZ, Corporate)."""
    segment_id: int
    name: str
    hosts: List[Host] = field(default_factory=list)
    value: float = 1.0  # business criticality
    
    @property
    def num_compromised(self) -> int:
        return sum(1 for h in self.hosts if h.compromised)
    
    @property
    def compromise_ratio(self) -> float:
        if not self.hosts:
            return 0.0
        return self.num_compromised / len(self.hosts)


class EnterpriseNetwork:
    """
    Enterprise network modeled as directed graph G=(V,E).
    V = network segments, E = connectivity.
    """
    
    SEGMENT_CONFIGS = {
        0: {"name": "DMZ", "value": 0.6, "vuln_mean": 0.4},
        1: {"name": "Corporate", "value": 0.8, "vuln_mean": 0.3},
        2: {"name": "Development", "value": 0.7, "vuln_mean": 0.35},
        3: {"name": "DataCenter", "value": 1.0, "vuln_mean": 0.2},
        4: {"name": "Cloud", "value": 0.9, "vuln_mean": 0.25},
    }
    
    # Adjacency: which segments can reach which (directed)
    ADJACENCY = {
        0: [1, 2],       # DMZ -> Corporate, Development
        1: [0, 2, 3],    # Corporate -> DMZ, Dev, DC
        2: [1, 3, 4],    # Development -> Corporate, DC, Cloud
        3: [1, 2, 4],    # DataCenter -> Corporate, Dev, Cloud
        4: [2, 3],       # Cloud -> Dev, DC
    }
    
    def __init__(self, hosts_per_segment: int = 40, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.hosts_per_segment = hosts_per_segment
        self.segments: List[NetworkSegment] = []
        self.all_hosts: List[Host] = []
        self._build_network()
    
    def _build_network(self):
        """Build network topology with hosts."""
        host_id = 0
        self.segments = []
        self.all_hosts = []
        
        for seg_id, config in self.SEGMENT_CONFIGS.items():
            segment = NetworkSegment(
                segment_id=seg_id,
                name=config["name"],
                value=config["value"],
            )
            for _ in range(self.hosts_per_segment):
                vuln = np.clip(
                    self.rng.normal(config["vuln_mean"], 0.1), 0.05, 0.95
                )
                host = Host(host_id=host_id, segment_id=seg_id, vulnerability=vuln)
                segment.hosts.append(host)
                self.all_hosts.append(host)
                host_id += 1
            self.segments.append(segment)
    
    def reset(self):
        """Reset all hosts to clean state."""
        for host in self.all_hosts:
            host.reset()
    
    def get_state_vector(self) -> np.ndarray:
        """Get global state: per-segment [compromise_ratio, avg_stage, num_detected, num_isolated]."""
        state = []
        for seg in self.segments:
            comp_ratio = seg.compromise_ratio
            avg_stage = np.mean([h.compromise_stage for h in seg.hosts]) / 5.0
            det_ratio = sum(1 for h in seg.hosts if h.detected) / len(seg.hosts)
            iso_ratio = sum(1 for h in seg.hosts if h.isolated) / len(seg.hosts)
            state.extend([comp_ratio, avg_stage, det_ratio, iso_ratio])
        return np.array(state, dtype=np.float32)
    
    def get_reachable_segments(self, segment_id: int) -> List[int]:
        """Get segments reachable from given segment."""
        return self.ADJACENCY.get(segment_id, [])
    
    @property
    def num_hosts(self) -> int:
        return len(self.all_hosts)
    
    @property
    def total_compromised(self) -> int:
        return sum(1 for h in self.all_hosts if h.compromised)
    
    @property
    def total_detected(self) -> int:
        return sum(1 for h in self.all_hosts if h.detected)
    
    @property
    def total_isolated(self) -> int:
        return sum(1 for h in self.all_hosts if h.isolated)
