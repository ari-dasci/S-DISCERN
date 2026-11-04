from .baseMethod import BaseMethod
from .DELETEMethod import DELETEMethod
from .EWCMethod import EWCMethod
from .SSDMethod import SSDMethod
from .NOISYMethod import NOISYMethod
from .RETRAINMethod import RETRAINMethod
from .DistributionalMethod import DistributionalMethod
from .ParametricMethod import ParametricMethod
from .BadTeacherMethod import BadTeacherMethod
from .BoundaryShrinkMethod import BoundaryShrinkMethod
from .BoundaryExpandMethod import BoundaryExpandMethod
from .SFUMethod import SFUMethod
from .SCRUBMethod import SCRUBMethod
from .RandomLabelMethod import RandomLabelMethod
from .FisherForgettingMethod import FisherForgettingMethod
from .UNSIRMethod import UNSIRMethod
from .ISPFMethod import ISPFMethod
from .ZSPAGMethod import ZSPAGMethod
from .DistributionalMethod_2 import DistributionalMethodContinual

__all__ = ['BaseMethod', 'DELETEMethod', 'EWCMethod', 'SSDMethod', 'NOISYMethod', 'RETRAINMethod', 'DistributionalMethod', 'ParametricMethod',
           'BadTeacherMethod', 'BoundaryShrinkMethod', 'BoundaryExpandMethod', 'SFUMethod', 'SCRUBMethod',
           'RandomLabelMethod', 'FisherForgettingMethod', 'UNSIRMethod', 'ISPFMethod', 'ZSPAGMethod',
           'DistributionalMethodContinual']