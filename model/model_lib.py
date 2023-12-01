from model.agentformer import AgentFormer
from model.dlow import DLow
from model.untrained_models import Oracle, ConstantVelocityPredictor


model_dict = {
    'agentformer': AgentFormer,
    'dlow': DLow,
    'oracle': Oracle,
    'const_velocity': ConstantVelocityPredictor
}
