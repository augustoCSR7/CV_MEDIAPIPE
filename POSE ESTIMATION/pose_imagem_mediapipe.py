import cv2
import mediapipe as mp
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=False) as pose:


    image = cv2.imread("teste01.jpg")
    height, wwidth, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Teste", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()