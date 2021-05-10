import numpy as np


def _get_angle(a, b, c):
    """
    Get angle between vector ba and bc
    """

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def get_points_to_zoom(lm):
    """
    Get index_finger_tip, thumb_tip and thumb_cmc coordinates
    """

    index_finger_tip = np.array([lm[24], lm[25]])
    thumb_tip = np.array([lm[12], lm[13]])
    thumb_cmc = np.array([lm[3], lm[4]])

    return index_finger_tip, thumb_tip, thumb_cmc


def get_average_points(lm):
    """
    Get average x and y cordinates of tips points
    """

    x_average = (lm[12] + lm[24] + lm[36] + lm[48] + lm[60]) / 5
    y_average = (lm[13] + lm[25] + lm[37] + lm[49] + lm[61]) / 5

    return x_average, y_average


def get_angle(hand_landmarks):
    """
    Get angle between index tip, thumb_cmc and thumb tip.
    """

    index_cd, thumb_cd, thumb_cmc_cd = get_points_to_zoom(hand_landmarks)
    return _get_angle(index_cd, thumb_cmc_cd, thumb_cd)
