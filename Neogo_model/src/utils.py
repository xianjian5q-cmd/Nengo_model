import nengo_spa as spa
from epa_sentences import epa_sentences
import os
import numpy as np
import pickle
from nengo_spa import Vocabulary
import numpy as np

def get_all_words():
    """
    Load 3D-EPA values for the whole dictionary.
    """
    data_path = os.path.join(
        os.path.dirname(__file__), os.pardir, 'data', 'epa_dimensions.pkl')

    print(data_path)
    with open(data_path, 'rb') as f:
        epa_all = pickle.load(f)

    return epa_all.keys()


print(get_all_words())


def create_spa_vocabulary(experiment, D=256):
    vocab = {}
    networks = experiment.vocab.keys()

    for network in networks:
        try:
            words = experiment.vocab[network]
        except KeyError:
            raise ValueError(f"Vocabulary for network '{network}' is undefined in experiment '{experiment['name']}'.")

        use_random = False
        vocab[network] = Vocabulary(D)

        # Parse each semantic pointer
        for word in words:
            vocab[network].populate(word)
    return vocab


def get_epa_expression(words):
    """
    Returns expressions like: 0.5*E + 0.2*P -0.3*A
    """
    data_path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'epa_dimensions.pkl')
    with open(data_path, 'rb') as f:
        epa_all = pickle.load(f)

    norm_fact = 4.
    keys = []

    for word in words:
        if word not in epa_all:
            print(f"Warning: {word} not found in EPA, using (0,0,0)")
            V, A, D = 0, 0, 0
        else:
            V, A, D = (epa_all[word] -5 )/ norm_fact

        expr = f"{V:.2f}*V + {A:.2f}*A + {D:.2f}*D"
        keys.append(expr.replace("+-", "-"))
    return keys


def add_vocabularies(vocab, name1, name2):
    """
    Combine two vocabularies into one using nengo_spa's Vocabulary.
    """
    vocab1, vocab2 = vocab[name1], vocab[name2]
    assert vocab1.dimensions == vocab2.dimensions

    # Create new vocabulary with same dimensions
    new_vocab = spa.Vocabulary(vocab1.dimensions)
    
    # Add all items from first vocabulary
    for key in vocab1.keys():
        new_vocab.add(key, vocab1[key].v)
    
    # Add all items from second vocabulary
    for key in vocab2.keys():
        # Handle potential name conflicts
        new_key = key
        if key in new_vocab:
            new_key = f"{name2}_{key}"
        new_vocab.add(new_key, vocab2[key].v)

    return new_vocab


def keys_from_input(keys, factor=None):
    """
    Process input keys with optional factor multiplication.
    """
    new_keys = []
    prefix = ''

    if factor is not None:
        prefix = str(factor) + '*'

    for key in keys:
        word = key
        if '_' in key and any(w in key for w in ('Sensory', 'Episodic','Affect'
                                                 'Executive', 'Action')):
                word = key.split('_', 1)[1]
        word = prefix + word
        new_keys.append(word)
    return new_keys
