import cv2 as cv
import mediapipe as mp
import subprocess
import sys

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cont = 0

buttons = {
    1: {"label": "Voltar", "script": './menu.py', "pos": (100, 400), "size": (200, 100)},
    2: {"label": "1 Jogador", "script": './pong_single.py', "pos": (350, 400), "size": (200, 100)},
    3: {"label": "2 Jogadores", "script": './pong_multy.py', "pos": (600, 400), "size": (200, 100)}
}


def getHandMove(hand_landmarks):
    global buttons, cont

    landmarks = hand_landmarks.landmark

    indicador_x = int(landmarks[8].x * frame_width)
    indicador_y = int(landmarks[8].y * frame_height)

    finger_in_buttons = False  # Flag to indicate if the finger is in any button

    for button_id, button_data in buttons.items():
        x, y = button_data["pos"]
        w, h = button_data["size"]
        if x <= indicador_x <= x + w and y <= indicador_y <= y + h:
            finger_in_buttons = True
            cont += 1
            if cont >= 25:  # If the finger is in place for 25 frames
                open_script(button_data['script'])
                cont = 0

    if not finger_in_buttons:
        cont = 0  # Reset the counter if the finger is not in any button

def draw_yellow_dot(image, coordinates):
    cv.circle(image, coordinates, 10, (0, 255, 255), -1)  # Yellow in BGR format: (0, 255, 255)

def open_script(script_name):
    subprocess.Popen([sys.executable, script_name])
    sys.exit()

vid = cv.VideoCapture(0)

# Define the window size
window_width = 1280
window_height = 720

# Set the webcam capture resolution
vid.set(cv.CAP_PROP_FRAME_WIDTH, window_width)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, window_height)

# Create a window and set its size
cv.namedWindow('Menu - Jogo do Pong', cv.WINDOW_NORMAL)
cv.resizeWindow('Menu - Jogo do Pong', window_width, window_height)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
) as hands:
    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break

        frame = cv.flip(frame, 1)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        frame_height, frame_width, _ = frame.shape

        results = hands.process(frame)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # Get the coordinates of the index fingertip
                indicador_x = int(hand_landmark.landmark[8].x * frame_width)
                indicador_y = int(hand_landmark.landmark[8].y * frame_height)

                draw_yellow_dot(frame, (indicador_x, indicador_y))

        hls = results.multi_hand_landmarks
        if hls and len(hls) == 1:
            getHandMove(hls[0])

        for buttons_number, button_info in buttons.items():
            x, y = button_info["pos"]
            w, h = button_info["size"]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), -1)
            cv.putText(frame, button_info["label"], (x + 10, y + 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Menu - Jogo do Pong', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
