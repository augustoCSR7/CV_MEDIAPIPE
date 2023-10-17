import cv2
import mediapipe as mp
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

    # REDIMENSIONANDO UMA IMAGEM

    # img_teste = Image.open('imagem_teste1.jpeg')
    # width = img_teste.width // 2
    # height = img_teste.height // 2
    # img_resized = img_teste.resize((width, height))
    # img_resized.save('imagem_redim_teste1.jpeg')

    image = cv2.imread("imagem_redim_teste1.jpeg")
    height, width, _ = image.shape
    image = cv2.flip(image, 1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    print("Handedness: ", results.multi_handedness)

    # HAND LANDMARKS
    # print("Hand Landmarks: ", results.multi_hand_landmarks)

    if results.multi_hand_landmarks is not None:

        for hand_landmarks in results.multi_hand_landmarks:

            # x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
            # y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)

            # x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
            # y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

            # x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
            # y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)

            # x4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
            # y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)

            # x5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
            # y5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)

            # cv2.circle(image, (x1,y1), 3, (255,0,0), 3)
            # cv2.circle(image, (x2,y2), 3, (255,0,0), 3)
            # cv2.circle(image, (x3,y3), 3, (255,0,0), 3)
            # cv2.circle(image, (x4,y4), 3, (255,0,0), 3)
            # cv2.circle(image, (x5,y5), 3, (255,0,0), 3)

            print(hand_landmarks)
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )

    image = cv2.flip(image, 1)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
