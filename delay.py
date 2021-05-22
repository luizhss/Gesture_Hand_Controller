import numpy as np
from collections import Counter


class Delay():
    """
    Delay Frames Class.

    This class is responsible to provoke delays on the execution frames.
    Required Arguments:
        classes {list[string], required}: List of class names from the trained classifier.

    Keyword Arguments:
        moving_average {float, optional}: minimum percentage of pose prediction
                        in the last (frames_in_action or frames_out) frames to
                        determine the prediction.
                (Default: {0.8})
        frames_in_action {int, optional}: number of frames to be considered to
                determine a pose when already in an action state.
                (Default: {20})
        frames_out {int, optional}: number of frames to be considered to
                        determine a pose when in idle state.
                (Default: {45})

    """

    def __init__(self, classes, moving_average=.8, frames_in_action=20, frames_out=45):
        self.in_action = False
        self.counter_class = np.empty((0))
        self.classes = list(classes)
        self.counter_confidences = np.empty(
            (0, len(classes)), dtype=np.float64)
        self.moving_average = moving_average
        self.frames_in_action = frames_in_action
        self.frames_out = frames_out
        self.ignore_frames = 0

    def reset_counter(self, ignore_next_frames=0):
        """
        Clear counters arrays and can ignore the next frames
        """

        self.in_action = False
        self.counter_class = np.empty((0))
        self.counter_confidences = np.empty(
            (0, len(self.classes)), dtype=np.float64)

        if ignore_next_frames > 0:
            self.ignore_frames = ignore_next_frames

    def get_prediction(self):
        """
        Based the last frames, check if the most common prediction respect the
        moving average rule
        """

        most_common_class, most_common_rep = Counter(
            self.counter_class).most_common(1)[0]

        if (self.in_action and most_common_rep >= self.moving_average * self.frames_in_action) or\
                (not self.in_action and most_common_rep >= self.moving_average * self.frames_out):

            if most_common_class == 'Unknown':
                return ('Unknown', 1.0)

            idx_cls = self.classes.index(most_common_class)
            avg_confidence = self.counter_confidences.mean(axis=0)[idx_cls]

            return (most_common_class, avg_confidence)

        return ('Unknown', 1.0)

    def set_in_action(self, value):
        """
        Change the in_action state
        """

        if self.in_action == value:
            return
        self.in_action = value
        if value:
            self.counter_class = self.counter_class[-self.frames_in_action:]
            self.counter_confidences = self.counter_confidences[-self.frames_in_action:, :]

    def update(self, cls, conf=None):
        """
        Based on the last frames, compute the most possible prediction and
        its confidence
        """
        if conf is None:
            conf = np.zeros((1, len(self.classes)), dtype=np.float64)

        if self.ignore_frames > 0:
            self.ignore_frames -= 1
            return (None, None)

        self.counter_class = np.append(self.counter_class, cls)
        self.counter_confidences = np.vstack((self.counter_confidences, conf))

        if (self.in_action and len(self.counter_class) < self.frames_in_action) or\
           (not self.in_action and len(self.counter_class) < self.frames_out):
            return (None, None)

        self.counter_class = np.delete(self.counter_class, 0)
        self.counter_confidences = np.delete(
            self.counter_confidences, 0, axis=0)

        return self.get_prediction()
