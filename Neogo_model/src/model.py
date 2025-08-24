import nengo
import numpy as np
import simulations as sim
import utils
import nengo_spa as spa
from nengo_spa import Network, Vocabulary, SemanticPointer
from nengo_spa import AssociativeMemory

def create_model(D=256):
    vocabulary = sim.Experiment()
    seed=0
    # Create semantic pointers in each network
    spa_voc = utils.create_spa_vocabulary(vocabulary, D=D)
    with spa.Network("AssociativeMemory", seed=seed) as model:
        sensory = spa.State(vocab=spa_voc['sensory'])

        episodic_map = {key: key for key in spa_voc['episodic'].keys()}
        episodic = spa.ThresholdingAssocMem(
            input_vocab=spa_voc['sensory'],
            output_vocab=spa_voc['episodic'],
            mapping=episodic_map,
            threshold=0.3,
        )
        nengo.Connection(sensory.output, episodic.input, transform=3)  # inputs from cloud not normalized
        # Affect: episodic -> affect

        affect_map = {key: key for key in spa_voc['affect'].keys()}
        affect = spa.ThresholdingAssocMem(
            input_vocab=spa_voc['episodic'],
            output_vocab=spa_voc['affect'],
            mapping=affect_map,
            threshold=0.4,
        )
        nengo.Connection(episodic.output, affect.input)

        # Executive: affect -> executive
        emotion_tags = spa_voc['executive'].keys
        epa_expressions = utils.get_epa_expression(emotion_tags)
        executive_map = {k: v for k, v in zip(epa_expressions, emotion_tags)}
        executive = AssociativeMemory(
            input_vocab=spa_voc['affect'],
            output_vocab=spa_voc['executive'],
            mapping=executive_map,
            threshold=0.8,
        )
        nengo.Connection(affect.output, executive.input)

        # Mood node providing constant input vector (e.g. 'E' in affect vocab)
        def input_func(t):
            return spa_voc['affect']['E'].v
        mood = nengo.Node(input_func, size_out=D)
        nengo.Connection(mood, executive.input, transform=3)

        # Action: executive -> action
        action_map = {key: key for key in spa_voc['action'].keys}
        action = AssociativeMemory(
            input_vocab=spa_voc['executive'],
            output_vocab=spa_voc['action'],
            mapping=action_map,
            wta_output=True,
            wta_inhibit_scale=0.1,
            threshold=0.8,
        )
        nengo.Connection(executive.output, action.input, transform=4)
        print(spa_voc['action']['run'])

    return model, spa_voc


model, spa_voc = create_model(D=512)
