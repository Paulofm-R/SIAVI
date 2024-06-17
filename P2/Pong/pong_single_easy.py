import cv2 as cv
import mediapipe as mp
import numpy as np
import random
import time
from screeninfo import get_monitors

RAQUETE_WIDTH = 150
RAQUETE_HEIGHT = 20
RAQUETE_VELOCIDADE = 5

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
        "velocidade_x": 20,
        "velocidade_y": 20,
        "bola_tamanho": random.randint(15, 30),
        "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    }
}

ultimo_ponto_tempo = {}  # Dicionário para armazenar o tempo do último ponto de cada bola
set = False
quantidade_raquetes = 1

def update_ball_position(frame_width, frame_height):
    global player1_score, quantidade_raquetes, ultimo_ponto_tempo
    # Criar uma cópia do dicionário bolas antes de iterar sobre ele
    bolas_copy = bolas.copy()
    current_time = time.time()
    
    for bola_id, bola in bolas_copy.items():
        # Atualizar posição da bola
        bola["x"] += bola["velocidade_x"] * bola["dir_x"]
        bola["y"] += bola["velocidade_y"] * bola["dir_y"]

        # Verificar colisões com as paredes
        if bola["y"] <= 0 or bola["y"] >= frame_height:
            bola["dir_y"] *= -1

        if quantidade_raquetes == 1:
            # Verificar colisão com as raquetes
            if bola["y"] >= frame_height - RAQUETE_HEIGHT and player1_x <= bola["x"] <= player1_x + RAQUETE_WIDTH:
                if bola_id not in ultimo_ponto_tempo or current_time - ultimo_ponto_tempo[bola_id] > 1:
                    player1_score += 1
                    ultimo_ponto_tempo[bola_id] = current_time
                    bola["dir_y"] *= -1

                    bola["velocidade_x"] += 2
                    bola["velocidade_y"] += 2

                    if player1_score > 5 and len(bolas) == 1:
                        adicionar_bola(frame_width, frame_height)
                    if player1_score > 10 and len(bolas) == 2:
                        adicionar_bola(frame_width, frame_height)
                        quantidade_raquetes += 1

        if quantidade_raquetes == 2:
            # Verificar colisão com as raquetes
            if (bola["y"] >= frame_height - RAQUETE_HEIGHT and player1_x <= bola["x"] <= player1_x + RAQUETE_WIDTH) or (bola["y"] >= frame_height - RAQUETE_HEIGHT and player2_x <= bola["x"] <= player2_x + RAQUETE_WIDTH):
                if bola_id not in ultimo_ponto_tempo or current_time - ultimo_ponto_tempo[bola_id] > 1:
                    player1_score += 1
                    ultimo_ponto_tempo[bola_id] = current_time
                    bola["dir_y"] *= -1

                    bola["velocidade_x"] += 3
                    bola["velocidade_y"] += 3

                    if player1_score > 5 and len(bolas) == 1:
                        adicionar_bola(frame_width, frame_height)
                    if player1_score > 10 and len(bolas) == 2:
                        adicionar_bola(frame_width, frame_height)
                        quantidade_raquetes += 1

        # Verificar colisão com as bordas laterais
        if bola["x"] <= 0 or bola["x"] >= frame_width:
            bola["dir_x"] *= -1

        # Verificar se a bola ultrapassou a parte inferior da tela
        if bola["y"] >= frame_height:
            reset_ball(bola, frame_width, frame_height)

def reset_ball(bola, frame_width, frame_height):
    bola['x'] = frame_width // 2
    bola['y'] = frame_height // 2
    bola['dir_x'] = np.random.choice([-1, 1])
    bola['dir_y'] = np.random.choice([-1, 1])
    bola["velocidade_x"] = 6
    bola["velocidade_y"] = 6

def inicial_set(frame_width, frame_height):
    global set, player1_y, bolas
    if not set:
        # Variáveis para o jogo
        player1_y = frame_height // 2 - RAQUETE_HEIGHT // 2

        # Inicializar bola 1 no dicionário bolas
        bolas[1] = {
            "x": frame_width // 2,
            "y": frame_height // 2,
            "dir_x": np.random.choice([-1, 1]),
            "dir_y": np.random.choice([-1, 1]),
            "velocidade_x": 4,
            "velocidade_y": -6,
            "bola_tamanho": random.randint(10, 20),
            "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        }
        # Definir set como True
        set = True

def adicionar_bola(frame_width, frame_height):
    bolas[len(bolas) + 1] = {
        "x": random.randint(0, frame_width),
        "y": random.randint(0, frame_height),
        "dir_x": random.choice([-1, 1]),
        "dir_y": random.choice([-1, 1]),
        "velocidade_x": random.randint(1, 10),
        "velocidade_y": random.randint(1, 10),
        "bola_tamanho": random.randint(10, 20),
        "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    }

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

        # Atualiza a posição da bola
        update_ball_position(screen_width, screen_height)

        frame = cv.flip(frame, 1)

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

        # Exibe a pontuação
        cv.putText(frame, f"Player 1: {player1_score}", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('Pong', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
