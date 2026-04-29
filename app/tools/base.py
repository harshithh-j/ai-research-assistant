from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    """
    Every tool must implement this interface.
    name        — unique identifier used by Claude
    description — tells Claude what this tool does and when to use it
    input_schema — JSON schema of the tool's parameters
    run()       — executes the tool and returns a result string
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def run(self, **kwargs) -> str:
        pass

    def to_claude_format(self) -> Dict[str, Any]:
        """
        Converts tool to the format Claude's API expects.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }