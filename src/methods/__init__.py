from src.methods.advdebias import AdvDebias
from src.methods.diffmcdp import DiffMCDP
from src.methods.dralign import DRAlign
from src.methods.erm import ERM
from src.methods.fairmixup import FairMixup
from src.methods.fairsplit import FairSplitMethod
from src.methods.hsic_ys import HSICYS
from src.methods.premover import PRemover

METHODS = {"erm": ERM, "fairsplit": FairSplitMethod, "advdebias": AdvDebias, "fairmixup": FairMixup,
           "dralign": DRAlign, "hsic_ys": HSICYS, "premover": PRemover, "diffmcdp": DiffMCDP}
