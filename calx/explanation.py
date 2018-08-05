class Explanation(object):

    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary
        self.local_exp = {}

    def as_list(self, label):
        # local explanations only
        exp = self.local_exp[label]

        if self.vocabulary is not None:
            exp = [(self.vocabulary[x[0]], x[1]) for x in exp]

        return exp