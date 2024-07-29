import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
hands = mp_hands.Hands(max_num_hands=2)
face_detection = mp_face_detection.FaceDetection()
mp_draw = mp.solutions.drawing_utils


def detect_hands(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    hand_label = "No hand detected"

    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, hand_handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)):
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            label = hand_handedness.classification[0].label
            score = hand_handedness.classification[0].score
            hand_label = f'{label} Hand ({score:.2f})'

            h, w, _ = image.shape
            bbox = [int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h), 0, 0]
            for lm in hand_landmarks.landmark:
                bbox[0] = min(bbox[0], int(lm.x * w))
                bbox[1] = min(bbox[1], int(lm.y * h))
                bbox[2] = max(bbox[2], int(lm.x * w))
                bbox[3] = max(bbox[3], int(lm.y * h))
            cv2.rectangle(image, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

            fingers_up = count_fingers(image, hand_landmarks, label)
            cv2.putText(image, f'Fingers: {fingers_up}', (bbox[0] - 20, bbox[3] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)

            if fingers_up == 1 and hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y:
                cv2.putText(image, 'Thumbs Up!', (bbox[0] - 20, bbox[3] + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                            2, cv2.LINE_AA)

    return image, hand_label


def count_fingers(image, hand_landmarks, handedness):
    finger_tips = [4, 8, 12, 16, 20]
    h, w, _ = image.shape

    fingers_up = 0

    if handedness == 'Left':
        if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
            fingers_up += 1
    else:
        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers_up += 1

    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers_up += 1

    return fingers_up


def detect_faces(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            mp_draw.draw_detection(image, detection)

    return image


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame = cv2.flip(frame, 1)

    frame, hand_label = detect_hands(frame)
    frame = detect_faces(frame)

    cv2.imshow('Hand and Face Detection', frame)
    print(hand_label)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
