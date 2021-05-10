import cv2
import os
import mediapipe as mp
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="data file name",
                    type=str, required=True)
parser.add_argument(
    "-p", "--path", help="directory to save the data", type=str, default='.')
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

landmarks = [x.name for x in mp_hands.HandLandmark]

data = []

file_name = args.file
path = args.path

file_path = os.path.join(path, file_name + '.csv')

cnt_mouse, cnt_left, cnt_right = 0, 0, 0
cnt_scrllup, cnt_scrlldown, cnt_zoom = 0, 0, 0


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while(True):
        ret, image = cap.read()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)

        key = (cv2.waitKey(10) & 0xFF)

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

                if (key == ord('a')):
                    new_data['class'] = 'mouse'
                    data.append(new_data)
                    cnt_mouse += 1
                elif key == ord('s'):
                    new_data['class'] = 'left_click'
                    data.append(new_data)
                    cnt_left += 1
                elif key == ord('d'):
                    new_data['class'] = 'right_click'
                    data.append(new_data)
                    cnt_right += 1
                elif key == ord('f'):
                    new_data['class'] = 'scroll_up'
                    data.append(new_data)
                    cnt_scrllup += 1
                elif key == ord('g'):
                    new_data['class'] = 'scroll_down'
                    data.append(new_data)
                    cnt_scrlldown += 1
                elif key == ord('h'):
                    new_data['class'] = 'zoom'
                    data.append(new_data)
                    cnt_zoom += 1

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('frame', image)

        print(f'\rM: {cnt_mouse} - L: {cnt_left} - R: {cnt_right} - UP: {cnt_scrllup} - DOWN: {cnt_scrlldown} - ZOOM: {cnt_zoom} -- TOTAL = {len(data)}', end='', flush=True)

        if key == ord('q'):
            break

        if key == ord('w'):
            if data[-1]['class'] == 'mouse':
                cnt_mouse -= 1
            elif data[-1]['class'] == 'left_click':
                cnt_left -= 1
            elif data[-1]['class'] == 'right_click':
                cnt_right -= 1
            elif data[-1]['class'] == 'scroll_up':
                cnt_scrllup -= 1
            elif data[-1]['class'] == 'scroll_down':
                cnt_scrlldown -= 1
            elif data[-1]['class'] == 'zoom':
                cnt_zoom -= 1

            data.pop(-1)

        if key == ord('e'):
            pd.DataFrame(data).to_csv(file_path, index=False)

print()
pd.DataFrame(data).to_csv(file_path, index=False)

cap.release()
cv2.destroyAllWindows()
