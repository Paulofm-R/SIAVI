import cv2 as cv
import mediapipe as mp
import subprocess
import sys
import numpy as np
import screeninfo

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cont = 0

buttons = {
    1: {"label": "Pong", "script": './Pong/pong_menu.py', "pos": (0, 0), "size": (250, 100)},
    2: {"label": "Jogo do Galo", "script": './JogoGalo/jogoGalo_menu.py', "pos": (0, 0), "size": (250, 100)},
    3: {"label": "Quatro em Linha", "script": './QuatroLinha/quatrolinha_menu.py', "pos": (0, 0), "size": (250, 100)}
}

# Função para desenhar um ponto amarelo no indicador
def draw_yellow_dot(image, coordinates):
    cv.circle(image, coordinates, 10, (0, 255, 255), -1)  # Amarelo no formato BGR: (0, 255, 255)

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

def setButtonPos(frame_width, frame_height):
    global buttons

    button_width = 250  # Largura fixa desejada para cada botão
    total_buttons_width = len(buttons) * button_width
    total_spacing = frame_width - total_buttons_width
    spacing_between_buttons = total_spacing / (len(buttons) + 1)

    current_x = spacing_between_buttons  # Começamos com o primeiro espaçamento

    for button_number, button_info in buttons.items():
        w, h = button_info["size"]
        y = frame_height / 2 - h / 2
        button_info["pos"] = (int(current_x), int(y))
        current_x += button_width + spacing_between_buttons  # Adicionamos largura do botão e espaçamento entre botões

def open_script(script_name):
    subprocess.Popen([sys.executable, script_name])
    sys.exit()

vid = cv.VideoCapture(0)

# Obter informações sobre o monitor principal
screen = screeninfo.get_monitors()[0]
frame_width = screen.width
frame_height = screen.height

# Ajustar o tamanho do vídeo capturado
vid.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

# Criar uma janela em tela cheia
cv.namedWindow('Menu Inicial', cv.WINDOW_FULLSCREEN)

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

        results = hands.process(frame)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        setButtonPos(frame_width, frame_height)
        hls = results.multi_hand_landmarks
        if hls and len(hls) == 1:
            getHandMove(hls[0])

        for buttons_number, button_info in buttons.items():
            x, y = button_info["pos"]
            w, h = button_info["size"]

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), -1)
            cv.putText(frame, button_info["label"], (x + 10, y + 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                draw_yellow_dot(frame, (int(hand_landmark.landmark[8].x * frame_width), int(hand_landmark.landmark[8].y * frame_height)))

        cv.imshow('Menu Inicial', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


vid.release()
cv.destroyAllWindows()
