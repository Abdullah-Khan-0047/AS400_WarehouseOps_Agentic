import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import GEMINI_CONFIG

try:
    import google.generativeai as genai
    GENAI_INSTALLED = True
except ImportError:
    GENAI_INSTALLED = False


class GeminiClient:
    """Shared Gemini connection used by all agents and the orchestrator."""
    _model = None
    _available = False
    _configured = False

    @classmethod
    def configure(cls, api_key: str = None) -> bool:
        if cls._configured:
            return cls._available
        cls._configured = True

        if not GENAI_INSTALLED:
            print("  [Gemini] google-generativeai package not installed. Running stats-only mode.")
            return False

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            print("  [Gemini] No GEMINI_API_KEY found in environment. Running stats-only mode.")
            return False

        try:
            genai.configure(api_key=key)
            cls._model = genai.GenerativeModel(
                GEMINI_CONFIG.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=GEMINI_CONFIG.temperature,
                    max_output_tokens=GEMINI_CONFIG.max_output_tokens,
                ),
            )
            cls._model.generate_content("ping")  # connectivity check
            cls._available = True
            print(f"  [Gemini] Connected to {GEMINI_CONFIG.model_name}.")
            return True
        except Exception as e:
            print(f"  [Gemini] Connection failed: {e}. Running stats-only mode.")
            return False

    @classmethod
    def query(cls, prompt: str, context: str = "") -> Optional[str]:
        if not cls._available:
            return None
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        try:
            response = cls._model.generate_content(full_prompt)
            if response and response.text:
                return response.text.strip()
            return None
        except Exception:
            return None

    @classmethod
    def is_available(cls) -> bool:
        return cls._available


def truncate_for_prompt(data, max_items: int = None) -> str:
    """Serialize data for Gemini prompts, truncating large lists."""
    cap = max_items or GEMINI_CONFIG.max_prompt_items
    if isinstance(data, list):
        data = data[:cap]
    return json.dumps(data, default=str, indent=2)


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    msg_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0


class BaseAgent(ABC):

    def __init__(self, name: str):
        self.name = name
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.action_log: List[Dict] = []
        self.ai_insights: List[Dict] = []  # stores Gemini-generated analysis
        self.is_initialized = False
        self.system_prompt = ""  # overridden by each subclass

    def query_gemini(self, prompt: str) -> Optional[str]:
        return GeminiClient.query(prompt, context=self.system_prompt)

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
    
    