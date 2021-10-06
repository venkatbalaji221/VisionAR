import cv2
import mediapipe as mp
import time
import math

font = cv2.FONT_HERSHEY_SIMPLEX
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
track_list = set()


def draw_circles(frame):
    if track_list is not None:
        for point in track_list:
            cv2.circle(frame, point, 10, (0, 255, 255), cv2.FILLED)


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = cv2.resize(image, None, None, fx=1.5, fy=1.5)
        img_ht, img_wd, img_ch = image.shape
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if hand_landmarks.landmark[2].x < hand_landmarks.landmark[17].x:  # Right Handedness

                    index_point = (int(hand_landmarks.landmark[8].x * img_wd), int(hand_landmarks.landmark[8].y * img_ht))
                    # cv2.circle(image, index_point, 10, (0, 0, 255), cv2.FILLED)
                    # finger states
                    thumbIsOpen = False
                    firstFingerIsOpen = False
                    secondFingerIsOpen = False
                    thirdFingerIsOpen = False
                    fourthFingerIsOpen = False

                    anchorPoint = hand_landmarks.landmark[2].x
                    if hand_landmarks.landmark[3].x < anchorPoint and hand_landmarks.landmark[4].x < anchorPoint:
                        thumbIsOpen = True

                    anchorPoint = hand_landmarks.landmark[6].y
                    if hand_landmarks.landmark[7].y < anchorPoint and hand_landmarks.landmark[8].y < anchorPoint:
                        firstFingerIsOpen = True

                    anchorPoint = hand_landmarks.landmark[10].y
                    if hand_landmarks.landmark[11].y < anchorPoint and hand_landmarks.landmark[12].y < anchorPoint:
                        secondFingerIsOpen = True

                    anchorPoint = hand_landmarks.landmark[14].y
                    if hand_landmarks.landmark[15].y < anchorPoint and hand_landmarks.landmark[16].y < anchorPoint:
                        thirdFingerIsOpen = True

                    anchorPoint = hand_landmarks.landmark[18].y
                    if hand_landmarks.landmark[19].y < anchorPoint and hand_landmarks.landmark[20].y < anchorPoint:
                        fourthFingerIsOpen = True

                    hand_gesture = [thumbIsOpen, firstFingerIsOpen, secondFingerIsOpen, thirdFingerIsOpen,
                                    fourthFingerIsOpen]
                    # Draw
                    if hand_gesture == [False, True, False, False, False]:
                        x = int(hand_landmarks.landmark[8].x * img_wd)
                        y = int(hand_landmarks.landmark[8].y * img_ht)
                        index_tip = (x, y)
                        track_list.add(index_tip)
                        cv2.putText(image, 'Draw', (5, 120), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
                    # Clear
                    if hand_gesture == [False, True, False, False, True]:
                        track_list.clear()
                        cv2.putText(image, 'Clear', (5, 120), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
                    # Erase
                    if hand_gesture == [True, False, False, False, False]:
                        x = int(hand_landmarks.landmark[4].x * img_wd)
                        y = int(hand_landmarks.landmark[4].y * img_ht)
                        thumb_tip = (x, y)
                        cv2.putText(image, 'Erase', (5, 120), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

                        cv2.circle(image, thumb_tip, 10, (255, 255, 255), cv2.FILLED)

                        remove_list = set()
                        for point in track_list:
                            if math.sqrt((point[0]-thumb_tip[0])**2 + (point[1]-thumb_tip[1])**2) <= 10:
                                remove_list.add(point)
                        track_list = track_list - remove_list
                        remove_list.clear()
            draw_circles(image)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(image, fps, (5, 50), font, 2, (255, 0, 0), 4, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
