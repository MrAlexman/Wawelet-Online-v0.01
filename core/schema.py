from dataclasses import dataclass
from typing import Any, Optional, List, Dict

@dataclass
class ParamSpec:
    key: str
    label: str
    type: str  # "float","int","bool","str","enum","list_str"
    default: Any
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None

    # Backward-compatible field (can be used as short description)
    hint: str = ""

    # New fields for richer tooltips in UI
    description: str = ""
    examples: Optional[List[str]] = None

    def __post_init__(self) -> None:
        # Backward compatibility: if old "hint" is set, use it as description when description is empty
        if self.hint and not self.description:
            self.description = self.hint


Schema = List[ParamSpec]


def schema_to_dict(schema: Schema) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in schema:
        out.append({
            "key": s.key,
            "label": s.label,
            "type": s.type,
            "default": s.default,
            "min": s.min,
            "max": s.max,
            "step": s.step,
            "choices": s.choices,

            # keep for compatibility
            "hint": s.hint,

            # new fields
            "description": s.description,
            "examples": s.examples,
        })
    return out
