import mediapipe as mp
import cv2


class HandDetect():
    """
    Hand Detect Class

    This class is responsible to detect the landmarks and the pose

    Keyword Arguments:
        detect_threshold {float, optional}: minimum percentage of a
        hand prediction
                (Default: {0.9})
    """

    def __init__(self, detect_threshold=0.90):
        self.detect_threshold = detect_threshold

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.landmarks = [x.name for x in mp.solutions.hands.HandLandmark]

    def image_preprocessing(self, image):
        """
        Compute the preprocessing to pass it to Mediapipe
            """

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        return image

    def detect_hand(self, hands, image, hand_pose, delay):
        """
        Detect the hand using MediaPipe and its pose using our trained SVC
        """

        image = self.image_preprocessing(image)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                if handedness.classification[0].score <= self.detect_threshold:
                    delay.update('Unknown')
                    continue

                hand_detected = []
                for lm in self.landmarks:

                    landmark_idx = self.mp_hands.HandLandmark[lm]

                    hand_detected.append(
                        hand_landmarks.landmark[landmark_idx].x)
                    hand_detected.append(
                        hand_landmarks.landmark[landmark_idx].y)
                    hand_detected.append(
                        hand_landmarks.landmark[landmark_idx].z)

                pose_now, confidences = hand_pose.predict_pose(hand_detected)
                class_in_action, confidence_in_action = delay.update(
                    pose_now, confidences)

                yield (class_in_action, confidence_in_action), (hand_detected, hand_landmarks)
        else:
            delay.update('Unknown')
