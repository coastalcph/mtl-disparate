"""
Constants shared across modules.
"""

STANCE = "semeval2016-task6-stance"
FNC = "fakenewschallenge"
NLI = "multinli"
TOPIC = "topic-based"
TOPIC_5WAY = "topic-based-5way"
LAPTOP = "absa-laptops"
RESTAURANT = "absa-restaurants"
TARGET = "target-dependent"
TASKS = [STANCE, FNC, NLI, TOPIC, TOPIC_5WAY, LAPTOP, RESTAURANT, TARGET]
RNN_CELL_TYPES = ["lstm", "phased_lstm", "layer_norm", "nas"]  # LSTM, plus the RNN cell types in Tensorflow interchangable with it

TASK_NAMES_SHORT = {"semeval2016-task6-stance": "STANCE", "fakenewschallenge": "FNC", "topic-based": "TOPIC", "multinli": "NLI",
                    "topic-based-5way": "TOPIC_5WAY", "absa-laptops": "LAPTOP", "absa-restaurants":"RESTAURANT", "target-dependent": "TARGET"}

STANCE_LABELS = ['AGAINST', 'FAVOR', 'NONE']
FNC_LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
NLI_LABELS = ['contradiction', 'entailment', 'neutral']
TOPIC_LABELS = ['negative', 'positive']
TOPIC_5WAY_LABELS = [-2.0, -1.0, 0.0, 1.0, 2.0]
ABSA_LABELS = ['negative', 'neutral', 'positive']
TARGET_LABELS = ['-1', '0', '1']

SIM = 'similarity'
DIV = 'diversity'
NONE = 'predsonly'
SIMILARITY_FEATURES = ['jensen-shannon', 'renyi', 'cosine', 'euclidean',
                       'variational', 'bhattacharyya']
DIVERSITY_FEATURES = ['num_word_types', 'type_token_ratio', 'entropy',
                      'simpsons_index', 'renyi_entropy']
# we don't use 'quadratic_entropy' at the moment, as it requires word vectors
