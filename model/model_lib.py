from model.agentformer import AgentFormer
from model.dlow import DLow
from model.original_agentformer import OrigAgentFormerWrapper, OrigDLowWrapper
from model.untrained_models import Oracle, ConstantVelocityPredictor


model_dict = {
    'agentformer': AgentFormer,
    'dlow': DLow,
    'orig_agentformer': OrigAgentFormerWrapper,
    'orig_dlow': OrigDLowWrapper,
    'oracle': Oracle,
    'const_velocity': ConstantVelocityPredictor
}
