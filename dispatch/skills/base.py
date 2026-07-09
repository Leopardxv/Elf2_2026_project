"""Base class for all AI skills."""
from abc import ABC, abstractmethod

class Skill(ABC):
    name: str = "base"
    description: str = ""
    requires_camera: bool = False
    requires_eeg: bool = False

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the skill's model is loaded."""
        ...

    @abstractmethod
    def execute(self, question: str, **kwargs) -> str:
        """Execute the skill and return answer."""
        ...

    def cleanup(self):
        pass
