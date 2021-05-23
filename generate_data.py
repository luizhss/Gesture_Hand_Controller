import cv2
import os
import mediapipe as mp
import pandas as pd
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="data file name",
                    type=str, required=True)
parser.add_argument(
    "-p", "--path", help="directory to save the data", type=str, default='.')
args = parser.parse_args()

file_name = args.file
path = args.path
if not file_name.endswith('.csv'):
    file_name += '.csv'
file_path = os.path.join(path, file_name)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
landmarks = [x.name for x in mp_hands.HandLandmark]
key2cmd = {
    's': 'left_click',
    'd': 'right_click',
    'f': 'scroll_up',
    'g': 'scroll_down',
    'h': 'zoom',
}
counts = defaultdict(lambda: 0)
data = []

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while True:
        ret, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)

        key = chr(cv2.waitKey(10) & 0xFF)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].score <= .9:
                    continue

                new_data = {}
                for lm in landmarks:
                    new_data[lm + '_x'] = hand_landmarks.landmark[mp_hands.HandLandmark[lm]].x
                    new_data[lm + '_y'] = hand_landmarks.landmark[mp_hands.HandLandmark[lm]].y
                    new_data[lm + '_z'] = hand_landmarks.landmark[mp_hands.HandLandmark[lm]].z
                new_data['hand'] = handedness.classification[0].label

                if key2cmd.get(key, 'unknown') != 'unknown':
                    counts[key2cmd[key]] += 1
                    new_data['class'] = key2cmd[key]
                    data.append(new_data)

            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('frame', image)

        # print(f'\rM: {cnt_mouse} - L: {cnt_left} - R: {cnt_right} - UP: {cnt_scrllup} - DOWN: {cnt_scrlldown} - ZOOM: {cnt_zoom} -- TOTAL = {len(data)}', end='', flush=True)
        s = f'\r'
        for k in counts:
            s += f'{k}: {counts[k]} '
        print(s, end='', flush=True)

        # Quit
        if key == 'q':
            break

        # Undo
        if key == 'w':
            last_key = data[-1]['class']
            counts[last_key] -= 1
            data.pop(-1)

        # Write what you have w/o exit
        if key == 'e':
            pd.DataFrame(data).to_csv(file_path, index=False)

print()
pd.DataFrame(data).to_csv(file_path, index=False)

cap.release()
cv2.destroyAllWindows()
