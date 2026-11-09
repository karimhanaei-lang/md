import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                fingers = []

                if lm_list[3][1] > lm_list[3][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                tip_ids = [8, 12, 16, 20]
                for tip in tip_ids:
                    if lm_list[tip][2] < lm_list[tip - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total_fingers = fingers.count(1)

                cv2.putText(img, f'Fingers: {total_fingers}', (40, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if total_fingers >= 3:
                    pyautogui.press('volumeup')
                    cv2.putText(img, "Volume UP", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif total_fingers < 3: 
                    pyautogui.press('volumedown')
                    cv2.putText(img, "Volume DOWN", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Finger Count Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()