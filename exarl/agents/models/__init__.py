from exarl.agents.models.tf_model      import Tensorflow_Model
from exarl.agents.models.tf_mlp        import MLP
from exarl.agents.models.tf_lstm       import LSTM
from exarl.agents.models.tf_ac_softmax import ActorSoftmax
from exarl.agents.models.tf_sac        import SoftActor
from exarl.agents.models.tf_ac         import Actor
from exarl.agents.models.tf_sac        import SoftCritic
from exarl.agents.models.tf_ac         import Critic

Tensorflow_Model.register("MLP", MLP)
Tensorflow_Model.register("LSTM", LSTM)
Tensorflow_Model.register("Actor", Actor)
Tensorflow_Model.register("ActorSoftmax", ActorSoftmax)
Tensorflow_Model.register("SoftActor", SoftActor)
Tensorflow_Model.register("SoftCritic", SoftCritic)
Tensorflow_Model.register("Critic", Critic)
Tensorflow_Model.register("Actor", Actor)
