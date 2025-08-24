import nengo

import nengo_spa as spa

import numpy as np

import simulations as sim

import utils



def create_model(D=512):

    vocabulary = sim.Experiment()

    spa_voc = utils.create_spa_vocabulary(vocabulary, D=D)



    with spa.Network("AssociativeMemory", seed=0) as model:
        # Sensory input here
        sensory = spa.State(vocab=spa_voc['sensory'], label="Sensory_Input",)
        #Create mapping for eposidic vectors

        
        episodic_map = {k: v for k, v in zip(spa_voc['episodic'].keys(), spa_voc['episodic'])}
        episodic = spa.ThresholdingAssocMem(
            input_vocab=spa_voc['sensory'],
            output_vocab=spa_voc['episodic'],
            mapping=episodic_map,
            threshold=0.3,
            label="Episodic_Memory",
        )
        nengo.Connection(sensory.output, episodic.input, transform=1.2)

        #Affect layer

        input_keys = spa_voc['episodic'].keys()
        affect_exprs = utils.get_epa_expression(input_keys)
        affect_map = {k: v for k, v in zip(input_keys, affect_exprs)}
        affect = spa.ThresholdingAssocMem(
            input_vocab=spa_voc['episodic'],
            output_vocab=spa_voc['affect'],
            mapping=affect_map,
            threshold=0.4,
            label="Affect_Processing",
        )
        nengo.Connection(episodic.output, affect.input, transform=2)
        
        #Executive layer

        emotion_tags = spa_voc['executive'].keys()
        epa_exprs = utils.get_epa_expression(emotion_tags)
        executive_map = {expr: tag for expr, tag in zip(epa_exprs, emotion_tags)}
        executive = spa.WTAAssocMem(
            input_vocab=spa_voc['affect'],
            output_vocab=spa_voc['executive'],
            mapping=executive_map,
            threshold=0.5,
            inhibit_scale=0.2,
            label="Perceived_Discrete_Emotion"
        )
        nengo.Connection(affect.output, executive.input, transform=2)

        #Action decision

        action_tags = spa_voc['action'].keys()
        act_exprs = utils.get_epa_expression(action_tags)
        action_map = {expr: tag for expr, tag in zip(act_exprs, action_tags)}
        action = spa.ThresholdingAssocMem(
            input_vocab=spa_voc['affect'],
            output_vocab=spa_voc['action'],
            mapping=action_map,
            threshold=0.1,            
            label="Action_Decision"
        )
        nengo.Connection(affect.output, action.input, transform=1.9)

        Mood_vec = spa_voc['affect']['V'].v
        slider = nengo.Node(label="Mood-slider", size_in=1)

        def scale_E(t, x):
            return x[0] * Mood_vec

        Mood_input = nengo.Node(scale_E, size_in=1, size_out=D,label="Mood")

        nengo.Connection(slider, Mood_input, synapse=None)
        nengo.Connection(Mood_input, executive.input, transform=2)
        nengo.Connection(Mood_input, action.input, transform=2)
        
    return model, spa_voc



model, _ = create_model(D=512,

                        )  ## Set true only for Sim 5 and 6 
