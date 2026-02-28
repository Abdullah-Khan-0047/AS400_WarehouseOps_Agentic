from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    msg_type: str  # "alert", "recommendation", "query", "response"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # 0=normal, 1=high, 2=critical


class BaseAgent(ABC):

    def __init__(self, name: str):
        self.name = name
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.action_log: List[Dict] = []
        self.is_initialized = False

    @abstractmethod
    def initialize(self, historical_data: Dict) -> None:
        pass

    @abstractmethod
    def analyze(self) -> List[Dict]:
        pass

    @abstractmethod
    def recommend(self) -> List[Dict]:
        pass

    def send_message(self, recipient: str, msg_type: str, payload: Dict, priority: int = 0):
        msg = AgentMessage(
            sender=self.name,
            recipient=recipient,
            msg_type=msg_type,
            payload=payload,
            priority=priority,
        )
        self.outbox.append(msg)
        return msg

    def receive_message(self, message: AgentMessage):
        self.inbox.append(message)

    def log_action(self, action: str, details: Dict = None):
        self.action_log.append({
            "agent": self.name,
            "action": action,
            "details": details or {},
            "timestamp": datetime.now(),
        })

    def __repr__(self):
        return f"<Agent: {self.name} | Messages: {len(self.inbox)} in / {len(self.outbox)} out>"