import cv2 as cv
import mediapipe as mp
import random
import subprocess
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cont = 0

buttons = {
    1: {"label": "Voltar", "script": './menu.py', "pos": (100, 100), "size": (200, 100)},
    2: {"label": "1 Jogador", "script": './QuatroLinha/quatrolinha_single.py', "pos": (350, 100), "size": (200, 100)},
    3: {"label": "Quatro em Linha", "script": './QuatroLinha/quatrolinha_multy.py', "pos": (600, 100), "size": (200, 100)}
}


def getHandMove(hand_landmarks):
    global buttons, cont

    landmarks = hand_landmarks.landmark

    indicador_x = int(landmarks[8].x * frame_width)
    indicador_y = int(landmarks[8].y * frame_height)

    finger_in_buttons = False  # Flag para indicar se o dedo está em algum quadrado

    for button_id, button_data in buttons.items():
        x, y = button_data["pos"]
        w, h = button_data["size"]
        if x <= indicador_x <= x + w and y <= indicador_y <= y + h:
            finger_in_buttons = True
            cont += 1
            if cont >= 25:  # Se o dedo estiver no local por 50 frames
                open_script(button_data['script'])
                cont = 0  # Quando colocar o circulo ou o x

    if not finger_in_buttons:
        cont = 0  # Resetar o contador se o dedo não estiver em nenhum quadrado

def open_script(script_name):
    subprocess.Popen([sys.executable, script_name])
    sys.exit()

vid = cv.VideoCapture(0)

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
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        hls = results.multi_hand_landmarks
        if hls and len(hls) == 1:
            getHandMove(hls[0])

        for buttons_number, button_info in buttons.items():
            x, y = button_info["pos"]
            w, h = button_info["size"]
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), -1)
            cv.putText(frame, button_info["label"], (x + 10, y + 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Resize frame to increase its size
        frame = cv.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

        cv.imshow('Menu Inicial', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


vid.release()
cv.destroyAllWindows()
