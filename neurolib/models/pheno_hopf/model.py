from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class PhenoHopfModel(Model):
    """
    Stuart-Landau model with Hopf bifurcation.
    """

    name = "Phenomenological hopf"
    description = "Stuart-Landau model with Hopf bifurcation"

    init_vars = ["xs_init", "ys_init"]
    state_vars = ["x", "y"]
    output_vars = ["x", "y"]
    default_output = "x"
    input_vars = ["x_ext", "y_ext"]
    default_input = "x_ext"

    # because this is not a rate model, the input
    # to the bold model must be transformed
    boldInputTransform = lambda self, x: (x + 1) * 4

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None):

        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        # Initialize base class Model
        super().__init__(integration=integration, params=params)
