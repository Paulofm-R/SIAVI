import cv2 as cv
import mediapipe as mp
import subprocess
import sys
from screeninfo import get_monitors

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cont = 0

buttons = {
    1: {"label": "Voltar", "script": './menu.py', "pos": (0, 0), "size": (275, 100)},
    2: {"label": "1 Player", "script": './Pong/pong_single.py', "pos": (0, 0), "size": (275, 100)},
    3: {"label": "2 Players", "script": './Pong/pong_multy.py', "pos": (0, 0), "size": (275, 100)},
    4: {"label": "2 Player With PowerUps", "script": './Pong/pong_multy_powerups.py', "pos": (0, 0), "size": (275, 100)},
}

def getHandMove(hand_landmarks):
    global buttons, cont

    landmarks = hand_landmarks.landmark

    indicador_x = int(landmarks[8].x * screen_width)
    indicador_y = int(landmarks[8].y * screen_height)

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

def setButtonPos(screen_width, screen_height):
    global buttons

    button_width = 250
    total_buttons_width = len(buttons) * button_width
    total_spacing = screen_width - total_buttons_width
    spacing_between_buttons = total_spacing / (len(buttons) + 1)

    current_x = spacing_between_buttons

    for button_number, button_info in buttons.items():
        w, h = button_info["size"]
        y = screen_height / 2 - h / 2
        button_info["pos"] = (int(current_x), int(y))
        current_x += button_width + spacing_between_buttons

def draw_rounded_rectangle(image, top_left, bottom_right, color, radius=20, thickness=2, shadow=False):
    overlay = image.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right

    if shadow:
        shadow_color = (0, 0, 0)
        for i in range(5, 0, -1):
            cv.rectangle(overlay, (x1 - i, y1 - i), (x2 + i, y2 + i), shadow_color, thickness=cv.FILLED,
                         lineType=cv.LINE_AA)

    cv.rectangle(overlay, top_left, bottom_right, color, thickness=cv.FILLED, lineType=cv.LINE_AA)

    cv.addWeighted(overlay, 0.5, image, 0.5, 0, image)


def draw_text(image, text, position, font_scale, color, thickness=2):
    font = cv.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv.getTextSize(text, font, font_scale, thickness)
    text_x, text_y = position
    text_x -= text_size[0] // 2
    text_y += text_size[1] // 2
    cv.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, lineType=cv.LINE_AA)

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
        
        # Obtém a resolução da tela
        screen = get_monitors()[0]
        screen_width = int(screen.width - (screen.width / 8))
        screen_height = int(screen.height - (screen.height / 8))

        # Redimensiona o frame para se ajustar à tela
        frame = cv.resize(frame, (screen_width, screen_height))
        frame = cv.flip(frame, 1)

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = hands.process(frame)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                # Get the coordinates of the index fingertip
                indicador_x = int(hand_landmark.landmark[8].x * screen_width)
                indicador_y = int(hand_landmark.landmark[8].y * screen_height)

                draw_yellow_dot(frame, (indicador_x, indicador_y))

        setButtonPos(screen_width, screen_height)
        hls = results.multi_hand_landmarks
        if hls and len(hls) == 1:
            getHandMove(hls[0])

        for buttons_number, button_info in buttons.items():
            x, y = button_info["pos"]
            w, h = button_info["size"]
            draw_rounded_rectangle(frame, (x, y), (x + w, y + h), (0, 128, 255), shadow=True)
            draw_text(frame, button_info["label"], (x + w // 2, y + h // 2), 1, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                draw_yellow_dot(frame, (
                int(hand_landmark.landmark[8].x * screen_width), int(hand_landmark.landmark[8].y * screen_height)))
                
        cv.imshow('Menu - Jogo do Pong', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
