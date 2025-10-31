import cv2 as cv
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors
import time
import subprocess
import sys

# Definir constantes para as raquetes
RAQUETE_WIDTH = 20
RAQUETE_HEIGHT = 150
RAQUETE_VELOCIDADE = 5

# Definir constantes para a bola
BOLA_SIZE = 15
BOLA_VELOCIDADE_X = 6
BOLA_VELOCIDADE_Y = 6

# Definir a duração do jogo em segundos (3 minutos)
DURACAO_JOGO = 3 * 60

player1_score = 0
player2_score = 0

player1_y = 0
player2_y = 0
bola_x = 0
bola_y = 0
bola_dir_x = 0
bola_dir_y = 0

resetSquare = {
    "x1": 0,
    "y1": 0,
    "x2": 0,
    "y2": 0,
}

exitSquare = {
    "x1": 0,
    "y1": 0,
    "x2": 0,
    "y2": 0,
}

cont = 0

set = False

def set_squares(frame_height, frame_width):
    global resetSquare, exitSquare

    resetSquare["x1"] = 100
    resetSquare["y1"] = int((frame_height / 2) - 82)
    resetSquare["x2"] = 300
    resetSquare["y2"] = int((frame_height / 2) + 82)

    exitSquare["x1"] = frame_width - 100
    exitSquare["y1"] = int((frame_height / 2) - 82)
    exitSquare["x2"] = frame_width - 300
    exitSquare["y2"] = int((frame_height / 2) + 82)

def getHandMove(hand_landmarks, frame_width, frame_height):
    global cont

    landmarks = hand_landmarks.landmark

    indicador_x = int(landmarks[8].x * frame_width)
    indicador_y = int(landmarks[8].y * frame_height)

    finger_in_square = False  # Flag para indicar se o dedo está em algum quadrado

    if resetSquare["x1"] < indicador_x < resetSquare["x2"] and resetSquare["y1"] < indicador_y < resetSquare["y2"]:
        finger_in_square = True
        cont += 1
        if cont >= 25:  # Se o dedo estiver no local por 50 frames
            reset(frame_width, frame_height)
    if exitSquare["x1"] > indicador_x > exitSquare["x2"] and exitSquare["y1"] < indicador_y < exitSquare["y2"]:
        finger_in_square = True
        cont += 1
        if cont >= 25:  # Se o dedo estiver no local por 50 frames
            exit()

    if not finger_in_square:
        cont = 0  # Resetar o contador se o dedo não estiver em nenhum quadrado

def update_ball_position():
    global bola_x, bola_y, bola_dir_x, bola_dir_y, player1_score, player2_score, BOLA_VELOCIDADE_X, BOLA_VELOCIDADE_Y

    # Atualizar posição da bola
    bola_x += BOLA_VELOCIDADE_X * bola_dir_x
    bola_y += BOLA_VELOCIDADE_Y * bola_dir_y

    # Verificar colisões com as paredes
    if bola_y <= 0 or bola_y >= screen_height:
        bola_dir_y *= -1

    # Verificar colisão com as raquetes
    if bola_x <= RAQUETE_WIDTH and player1_y <= bola_y <= player1_y + RAQUETE_HEIGHT:
        bola_dir_x *= -1
        BOLA_VELOCIDADE_X += 2
        BOLA_VELOCIDADE_Y += 2
    elif bola_x >= screen_width - RAQUETE_WIDTH and player2_y <= bola_y <= player2_y + RAQUETE_HEIGHT:
        bola_dir_x *= -1
        BOLA_VELOCIDADE_X += 2
        BOLA_VELOCIDADE_Y += 2
    elif bola_x <= 0:
        player2_score += 1
        BOLA_VELOCIDADE_X = 5
        BOLA_VELOCIDADE_Y = 5
        reset_ball()
    elif bola_x >= screen_width:
        BOLA_VELOCIDADE_X = 5
        BOLA_VELOCIDADE_Y = 5
        player1_score += 1
        reset_ball()

# Função para resetar a bola ao centro
def reset_ball():
    global bola_x, bola_y, bola_dir_x, bola_dir_y
    bola_x = screen_width // 2
    bola_y = screen_height // 2
    bola_dir_x = np.random.choice([-1, 1])
    bola_dir_y = np.random.choice([-1, 1])

def inicial_set(screen_width, screen_height):
    global set, player1_y, player2_y, bola_x, bola_y, bola_dir_x, bola_dir_y
    if not set:
        # Variáveis para o jogo
        player1_y = screen_height // 2 - RAQUETE_HEIGHT // 2
        player2_y = screen_height // 2 - RAQUETE_HEIGHT // 2
        bola_x = screen_width // 2
        bola_y = screen_height // 2
        bola_dir_x = np.random.choice([-1, 1])
        bola_dir_y = np.random.choice([-1, 1])
        set = True

def reset(screen_width, screen_height):
    global set, BOLA_VELOCIDADE_X, BOLA_VELOCIDADE_Y, start_time, cont, player1_score, player2_score

    set = False

    # Definir constantes para a bola
    BOLA_VELOCIDADE_X = 6
    BOLA_VELOCIDADE_Y = 6

    cont = 0

    # Inicia o temporizador
    start_time = time.time()

    player1_score = 0
    player2_score = 0

    inicial_set(screen_width, screen_height)

def exit():
    # Fecha todas as janelas do OpenCV
    cv.destroyAllWindows()

    # Inicia o novo script usando subprocess.Popen
    subprocess.Popen([sys.executable, "./Pong/pong_menu.py"])

    # Encerra o script atual
    sys.exit()

# Código de detecção de mãos com MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

vid = cv.VideoCapture(0)

# Inicia o temporizador
start_time = time.time()

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:
    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Obtém a resolução da tela
        screen = get_monitors()[0]
        screen_width = int(screen.width - (screen.width / 10))
        screen_height = int(screen.height - (screen.height / 10))

        # Redimensiona o frame para se ajustar à tela
        frame = cv.resize(frame, (screen_width, screen_height))

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

                # Obtém as coordenadas da ponta do dedo médio
                middle_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_tip_x = int(middle_finger_tip.x * screen_width)
                middle_finger_tip_y = int(middle_finger_tip.y * screen_height)

                # Ajuste das coordenadas para a largura da janela
                middle_finger_tip_x = screen_width - middle_finger_tip_x

                # Define a posição da raquete do jogador
                # Jogador 1 (esquerda)
                if middle_finger_tip_x < screen_width // 2:
                    player1_y = middle_finger_tip_y - RAQUETE_HEIGHT // 2
                else:  # Jogador 2 (direita)
                    player2_y = middle_finger_tip_y - RAQUETE_HEIGHT // 2

        # Calcula o tempo restante
        elapsed_time = time.time() - start_time
        remaining_time = max(DURACAO_JOGO - int(elapsed_time), 0)
        minutes = remaining_time // 60
        seconds = remaining_time % 60
        timer_text = f"{minutes:02}:{seconds:02}"

        frame = cv.flip(frame, 1)

        hls = results.multi_hand_landmarks
        if hls and len(hls) == 1 and remaining_time == 0:
            getHandMove(hls[0], screen_width, screen_height)

        # Atualiza a posição da bola
        if remaining_time > 0:
            update_ball_position()
        else:
            cv.rectangle(frame, (int(screen_width/2 - 300), int(screen_height/2 - 100)), (int(screen_width/2 + 300), int(screen_height/2 + 100)), (0, 0, 0), -1)

            player_win = ''

            if player1_score > player2_score:
                player_win = "Player 1 Win!!"
            elif player1_score < player2_score:
                player_win = "Player 2 Win!!"
            else:
                player_win = "Draw!!"

            cv.putText(frame, player_win, (int(screen_width/2 - 275), int(screen_height/2 + 25)),
                   cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            
            cv.rectangle(frame, (resetSquare["x1"], resetSquare["y1"]),
                    (resetSquare["x2"], resetSquare["y2"]), (174, 173, 178), 3)
            cv.rectangle(frame, (exitSquare["x1"], exitSquare["y1"]),
                    (exitSquare["x2"], exitSquare["y2"]), (174, 173, 178), 3)

            cv.putText(frame, "RESET", (screen_width - 255, int((screen_height / 2) - 90)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(frame, "EXIT", (150, int((screen_height / 2) - 90)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        if not set:
            inicial_set(screen_width, screen_height)
            set_squares(screen_height, screen_width)


        # Desenha as raquetes e a bola
        cv.rectangle(frame, (0, player1_y), (RAQUETE_WIDTH, player1_y + RAQUETE_HEIGHT), (0, 0, 255), -1)
        cv.rectangle(frame, (screen_width - RAQUETE_WIDTH, player2_y), (screen_width, player2_y + RAQUETE_HEIGHT), (0, 255, 0), -1)
        cv.circle(frame, (bola_x, bola_y), BOLA_SIZE, (255, 0, 0), -1)

        # Exibe a pontuação
        cv.putText(frame, f"Player 1: {player1_score}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, f"Player 2: {player2_score}", (screen_width - 200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Adiciona o traço vertical no meio da tela
        overlay = frame.copy()
        cv.line(overlay, (screen_width // 2, 0), (screen_width // 2, screen_height), (0, 0, 0), 2)
        alpha = 0.5
        frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Exibe o temporizador no centro superior da tela
        text_size = cv.getTextSize(timer_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (screen_width - text_size[0]) // 2
        cv.putText(frame, timer_text, (text_x, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)           

        cv.imshow('Pong', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
