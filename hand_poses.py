import pickle
import numpy as np


class HandPoses:
    """
            Hand Poses Class.

            This class predict hand pose from hand landmarks input.
            The prediction uses C-Support Vector Classification from Scikit-Learn

            Keyword Arguments:
                    pose_threshold {float}: SVC threshold in classification confidence.
                            (default: {0.98})
                    name_classifier {str}: path with classifier name to load model
                            (default: {handsPoseClassifier.pkl})
    """

    def __init__(self, pose_threshold=0.98, name_classifier='handsPoseClassifier_6classes.pkl'):
        self.pose_threshold = pose_threshold
        self.name_classifier = name_classifier
        self.classifier = pickle.load(open(name_classifier, 'rb'))

    def predict_pose(self, hand_detected):
        """
                This method predict hand pose from hand landmarks using C-Support Vector Classification
        """
        result = self.classifier.predict_proba(np.array([hand_detected]))
        return self.get_name_pose_predict(result)

    def get_name_pose_predict(self, result):
        """
                This method get name of predicted hand pose class, i.e. the class with the greater confidence
        """
        idx = np.argmax(result[0])
        pose = self.classifier.classes_[idx]

        if result[0, idx] < self.pose_threshold:
            pose = 'Unknown'
        return pose, result[0]
