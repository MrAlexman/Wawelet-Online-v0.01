from .noise import NoiseComponent
from .sine import SineComponent
from .rect_pulse import RectPulseComponent
from .gauss_pulse import GaussPulseComponent
from .chirp import ChirpComponent

BUILTIN_COMPONENTS = {
    "noise": NoiseComponent,
    "sine": SineComponent,
    "rect_pulse": RectPulseComponent,
    "gauss_pulse": GaussPulseComponent,
    "chirp": ChirpComponent,
}
