from generic.tf_utils.abstract_listener import EvaluatorListener
import collections


class ReferitAccuracyListener(EvaluatorListener):
    """The listener retrieves the score for each object in the image select the one with the highest score."""

    def __init__(self, require):
        super(ReferitAccuracyListener, self).__init__(require)
        self.scores = None
        self.reset()

    def after_batch(self, result, batch, is_training):

        for i, (softmax, game) in enumerate(zip(result, batch["raw"])):
            self.scores[game.dialogue_id][game.object_id] = [softmax[1], game.correct_object]

    def reset(self):
        self.scores = collections.defaultdict(dict)

    def before_epoch(self, is_training):
        self.reset()

    def evaluate(self):

        accuracy = 0.
        for game in self.scores.values():
            select_object = max(game.values(), key=lambda v: v[0])
            if select_object[1]:
                accuracy += 1.
        accuracy /= len(self.scores)

        return accuracy