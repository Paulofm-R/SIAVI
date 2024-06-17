import cv2 as cv
import mediapipe as mp
import numpy as np
import random
import time
from screeninfo import get_monitors
import subprocess
import sys

# Carregar a imagem da vida
life_image = cv.imread('./img/lifes.png', cv.IMREAD_UNCHANGED)
life_image = cv.resize(life_image, (50, 50))  # Redimensionar a imagem se necessário

# Converter a imagem de RGBA para RGB
if life_image.shape[2] == 4:
    life_image = cv.cvtColor(life_image, cv.COLOR_BGRA2BGR)
    

RAQUETE_WIDTH = 175
RAQUETE_HEIGHT = 25
RAQUETE_VELOCIDADE = 10

# Definir constantes para a bola

player1_score = 0
player1_max_score = 0
player1_x = 0
player2_x = 0

bolas = {
    1: {
        "x": 0,
        "y": 0,
        "dir_x": 0,
        "dir_y": 0,
        "velocidade_x": 15,
        "velocidade_y": 15,
        "bola_tamanho": random.randint(15, 30),
        "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    }
}

lifes = 3
lifes_cont = 0

set = False
quantidade_raquetes = 1

ultimo_ponto_tempo = {}  # Dicionário para armazenar o tempo do último ponto de cada bola

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

def update_ball_position(screen_width, screen_height):
    global player1_score, quantidade_raquetes, ultimo_ponto_tempo, lifes
    # Criar uma cópia do dicionário bolas antes de iterar sobre ele
    bolas_copy = bolas.copy()
    current_time = time.time()

    for bola_id, bola in bolas_copy.items():
        # Atualizar posição da bola
        bola["x"] += bola["velocidade_x"] * bola["dir_x"]
        bola["y"] += bola["velocidade_y"] * bola["dir_y"]

        # Verificar colisões com as paredes
        if bola["y"] <= 0 or bola["y"] >= screen_height:
            bola["dir_y"] *= -1

        if quantidade_raquetes == 1:
        # Verificar colisão com as raquetes
            if bola_id not in ultimo_ponto_tempo or current_time - ultimo_ponto_tempo[bola_id] > 1:
                if bola["y"] >= screen_height - RAQUETE_HEIGHT and player1_x <= bola["x"] <= player1_x + RAQUETE_WIDTH:
                    player1_score += 1
                    ultimo_ponto_tempo[bola_id] = current_time

                    bola["dir_y"] *= -1

                    bola["velocidade_x"] += 2
                    bola["velocidade_y"] += 2
        
        if quantidade_raquetes == 2:
        # Verificar colisão com as raquetes
            if bola_id not in ultimo_ponto_tempo or current_time - ultimo_ponto_tempo[bola_id] > 1:
                if (bola["y"] >= screen_height - RAQUETE_HEIGHT and player1_x <= bola["x"] <= player1_x + RAQUETE_WIDTH) or (bola["y"] >= screen_height - RAQUETE_HEIGHT and player2_x <= bola["x"] <= player2_x + RAQUETE_WIDTH):
                    player1_score += 1
                    ultimo_ponto_tempo[bola_id] = current_time


                    bola["dir_y"] *= -1

                    bola["velocidade_x"] += 3
                    bola["velocidade_y"] += 3
        
        if lifes_cont >= 10:
                lifes += 1
                lifes_cont = 0

        if player1_score > 5 and len(bolas) == 1:
            adicionar_bola(screen_width, screen_height)
        if player1_score > 15 and len(bolas) == 2:
            adicionar_bola(screen_width, screen_height)
            quantidade_raquetes += 1
        if player1_score > 30 and len(bolas) == 3:
            adicionar_bola(screen_width, screen_height)
                    

        # Verificar colisão com as bordas laterais
        if bola["x"] <= 0 or bola["x"] >= screen_width:
            bola["dir_x"] *= -1

        # Verificar se a bola ultrapassou a parte inferior da tela
        if bola["y"] >= screen_height:
            reset_ball(bola, screen_width, screen_height)


def reset_ball(bola, screen_width, screen_height):
    global lifes

    lifes -= 1

    bola['x'] = screen_width // 2
    bola['y'] = screen_height // 2
    bola['dir_x'] = np.random.choice([-1, 1])
    bola['dir_y'] = np.random.choice([-1, 1])
    bola["velocidade_x"] = 15
    bola["velocidade_y"] = 15


def inicial_set(screen_width, screen_height):
    global set, player1_y, bolas
    if not set:
        # Variáveis para o jogo
        player1_y = screen_height // 2 - RAQUETE_HEIGHT // 2

        # Inicializar bola 1 no dicionário bolas
        bolas[1] = {
            "x": screen_width // 2,
            "y": screen_height // 2,
            "dir_x": np.random.choice([-1, 1]),
            "dir_y": np.random.choice([-1, 1]),
            "velocidade_x": 15,
            "velocidade_y": 15,
            "bola_tamanho": random.randint(10, 20),
            "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        }
        # Definir set como True
        set = True


def adicionar_bola(screen_width, screen_height):
    bolas[len(bolas) + 1] = {
        "x": random.randint(0, screen_width),
        "y": random.randint(0, screen_height),
        "dir_x": random.choice([-1, 1]),
        "dir_y": random.choice([-1, 1]),
        "velocidade_x": 15,
        "velocidade_y": 15,
        "bola_tamanho": random.randint(15, 30),
        "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    }

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
            reset()
    if exitSquare["x1"] > indicador_x > exitSquare["x2"] and exitSquare["y1"] < indicador_y < exitSquare["y2"]:
        finger_in_square = True
        cont += 1
        if cont >= 25:  # Se o dedo estiver no local por 50 frames
            exit()

    if not finger_in_square:
        cont = 0  # Resetar o contador se o dedo não estiver em nenhum quadrado

def reset():
    global RAQUETE_WIDTH, RAQUETE_HEIGHT, RAQUETE_VELOCIDADE, player1_score, player1_x, player2_x, bolas, lifes, set

    RAQUETE_WIDTH = 175
    RAQUETE_HEIGHT = 25
    RAQUETE_VELOCIDADE = 10

    # Definir constantes para a bola

    player1_score = 0
    player1_x = 0
    player2_x = 0

    bolas = {
        1: {
            "x": 0,
            "y": 0,
            "dir_x": 0,
            "dir_y": 0,
            "velocidade_x": 25,
            "velocidade_y": 25,
            "bola_tamanho": random.randint(15, 30),
            "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        }
    }

    set = False
    lifes = 3

def exit():
    # Fecha todas as janelas do OpenCV
    cv.destroyAllWindows()
    cv.waitKey(1)  # Garante que todas as janelas sejam fechadas

    # Inicia o novo script usando subprocess.Popen
    subprocess.Popen([sys.executable, "./Pong/pong_menu.py"])

    # Encerra o script atual
    sys.exit()

# Código de detecção de mãos com MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

vid = cv.VideoCapture(0)

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

                if quantidade_raquetes == 1:
                    player1_x = middle_finger_tip_x - RAQUETE_WIDTH // 2
                elif quantidade_raquetes == 2:
                    if middle_finger_tip_x < screen_width // 2:
                        player1_x = middle_finger_tip_x - RAQUETE_WIDTH // 2
                    else:  # Jogador 2 (direita)
                        player2_x = middle_finger_tip_x - RAQUETE_WIDTH // 2

        frame = cv.flip(frame, 1)

        hls = results.multi_hand_landmarks
        if hls and len(hls) == 1 and lifes == 0:
            getHandMove(hls[0], screen_width, screen_height)

        set_squares(screen_height, screen_width)


        if lifes > 0:
            # Atualiza a posição da bola
            update_ball_position(screen_width, screen_height)

            # Resize frame to increase its size
            frame = cv.resize(frame, (screen_width, screen_height))

            if not set:
                inicial_set(screen_width, screen_height)

            # Desenha as raquetes e a bola
            cv.rectangle(frame, (player1_x, screen_height), (player1_x +
                            RAQUETE_WIDTH, screen_height - RAQUETE_HEIGHT), (0, 0, 255), -1)
            if quantidade_raquetes == 2:
                cv.rectangle(frame, (player2_x, screen_height), (player2_x +
                            RAQUETE_WIDTH, screen_height - RAQUETE_HEIGHT), (255, 0, 0), -1)

            for bola_id, bola in bolas.items():
                cv.circle(frame, (bola["x"], bola["y"]),
                        bola["bola_tamanho"], (bola["rgb"][0], bola["rgb"][1], bola["rgb"][2]), -1)
            
            for i in range(lifes):
                x_offset = screen_width - ((i * 50) + 60)
                y_offset = 30
                frame[y_offset:y_offset+life_image.shape[0], x_offset:x_offset+life_image.shape[1]] = life_image


        else:
            cv.rectangle(frame, (int(screen_width/2 - 300), int(screen_height/2 - 100)), (int(screen_width/2 + 300), int(screen_height/2 + 100)), (0, 0, 0), -1)
            cv.putText(frame, f"GAME OVER", (int(screen_width/2 - 275), int(screen_height/2 + 25)),
                   cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)
            
            cv.rectangle(frame, (resetSquare["x1"], resetSquare["y1"]),
                    (resetSquare["x2"], resetSquare["y2"]), (174, 173, 178), 3)
            cv.rectangle(frame, (exitSquare["x1"], exitSquare["y1"]),
                    (exitSquare["x2"], exitSquare["y2"]), (174, 173, 178), 3)

            cv.putText(frame, "RESET", (screen_width - 255, int((screen_height / 2) - 90)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(frame, "EXIT", (150, int((screen_height / 2) - 90)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Exibe a pontuação
        cv.putText(frame, f"Player 1: {player1_score}", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Pong - Single Hard', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
