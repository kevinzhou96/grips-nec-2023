import numpy as np
from negmas.gb.common import ResponseType
from negmas.sao.common import SAOResponse

from scml.oneshot.common import QUANTITY
from scml.oneshot.rl.action import (
    ActionManager,
    FixedPartnerNumbersActionManager,
    LimitedPartnerNumbersActionManager,
    UnconstrainedActionManager,
)
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.factory import (
    FixedPartnerNumbersOneShotFactory,
    LimitedPartnerNumbersOneShotFactory,
)
from scml.oneshot.rl.observation import (
    FixedPartnerNumbersObservationManager,
    LimitedPartnerNumbersObservationManager,
)

def make_env(
        level : int = 0,
        n_consumers : int = 4,
        n_suppliers: int = 0,
        extra_check : bool = False,
) -> OneShotEnv:
    pass