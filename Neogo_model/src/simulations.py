import numpy as np


class Experiment(object):

    def __init__(self):
        self.emotion_tags = ['ANGER', 'ANXIETY', 'BEING_HURT',
                             'COMPASSION', 'CONTEMPT', 'CONTENTMENT',
                             'DESPAIR', 'DISAPPOINTMENT', 'DISGUST', 'FEAR',
                             'GUILT', 'HAPPINESS', 'HATE', 'INTEREST',
                             'IRRITATION', 'JEALOUSY', 'JOY', 'LOVE',
                             'PLEASURE', 'PRIDE', 'SADNESS', 'SHAME', 'STRESS',
                             ]
        vocab = set(
            ['SNAKE', 'GLASS', 'ZOO', 'SMILED', 'FROWNED',
                'FELT_HEARTBEAT_GETTING_FASTER',
                'FELT_HEARTBEAT_SLOWING_DOWN',
                'MUSCLES_TENSING_WHOLE_BODY',
                'FELT_BREATHING_GETTING_FASTER', 'SWEATED', 'ANGRY',
                'CONSEQUENCES_NEGATIVE_FOR_SOMEBODY_ELSE',
                'FELT_GOOD',
                'FELT_NERVOUS',
                'FELT_AT_EASE',
                'EUPHORIC', 'MOTHER', 'SHOUT_AT', 'CHILD', 'SUBJECT',
                'OBJECT', 'ACTION', 'CHAIR',
                'CAKE', 'BINGE_EAT', 'PIZZA', 'PLANT', 'WALL',
                'OBESITY'])
        action_vocab=["FIGHT", 'FLIGHT','APPROACH', 'AVOIDANCE',  'EAT', 'HUG']
        epa_emo_tag=['E', 'P', 'A', 'ANGER', 'ANXIETY', 'BEING_HURT',
                             'COMPASSION', 'CONTEMPT', 'CONTENTMENT',
                             'DESPAIR', 'DISAPPOINTMENT', 'DISGUST', 'FEAR',
                             'GUILT', 'HAPPINESS', 'HATE', 'INTEREST',
                             'IRRITATION', 'JEALOUSY', 'JOY', 'LOVE',
                             'PLEASURE', 'PRIDE', 'SADNESS', 'SHAME', 'STRESS',]

        self.vocab = {}
        self.vocab['episodic'] = list(vocab)
        self.vocab['affect'] = ['V', 'A', 'D']
        self.vocab['executive'] = self.emotion_tags
        self.vocab['sensory'] = list(vocab)
        self.vocab['action'] = list(action_vocab)
        self.vocab['epa_emo_tag']=list(epa_emo_tag)
    