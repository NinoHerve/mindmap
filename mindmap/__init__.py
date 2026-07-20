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

from . import core
from . import geometry
from . import anatomy
from . import tasks
from . import analysis
from . import io
from . import metric

from .viz import *

