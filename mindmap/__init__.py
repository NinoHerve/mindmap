"""
mindmap
=======

A toolkit for source imaging. Complements the already existing MNE-python package.

Submodules:
- io — input/output functions
- viz — visualization functions
- forward — computes forward solutions
- inverse — computes inverse solutions
- head_model — manages head models
- sensor — manages sensor data
- metrics — computes various metrics
- utils — miscelaneous functions

Example:
    import mindmap as mm
"""

from . import io, viz
from .inverse import *
from .forward import *
from .utilities import *
from .head import *
from .simulation import *
