import cv2 as cv
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors
import time
import subprocess
import sys
import random


powerups =("MoreTime", "2xVelocityBall", "smallerrackets", "moreBalls", "obstacles", "racketsbiggest")

# Definir constantes para as raquetes
RAQUETE_WIDTH = 20
RAQUETE_HEIGHT = 150
RAQUETE_VELOCIDADE = 5

# Definir constantes para a bola
bolas = {
    1: {
        "x": 0,
        "y": 0,
        "dir_x": 0,
        "dir_y": 0,
        "velocidade_x": 6,
        "velocidade_y": 6,
        "bola_tamanho": 15,
        "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    }
}

# Definir a duração do jogo em segundos (3 minutos)
DURACAO_JOGO = 3 * 60

player1_score = 0
player2_score = 0

player1_y = 0
player2_y = 0

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

# Defina constantes para o intervalo de roleta e duração do efeito
ROLETA_INTERVALO = 30  # segundos

# Variáveis para a roleta
ultima_roleta_tempo = time.time()
roleta_ativa = False
roleta_powerup = None
roleta_start_time = 0

# Variáveis de estado dos power-ups
powerup_end_time = 0
two_ball_active = False
additional_ball = None
obstacles_active = False
obstacle1 = {"x": 0, "y": 0, "width": 0, "height": 0}
obstacle2 = {"x": 0, "y": 0, "width": 0, "height": 0}


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

    # Criar uma cópia do dicionário bolas antes de iterar sobre ele
    bolas_copy = bolas.copy()

    for bola_id, bola in bolas_copy.items():
        # Atualizar posição da bola
        bola["x"] += bola["velocidade_x"] * bola["dir_x"]
        bola["y"] += bola["velocidade_y"] * bola["dir_y"]


        # Verificar colisões com as paredes
        if bola["y"] <= 0 or bola["y"] >= screen_height:
            bola["dir_y"] *= -1

        # Verificar colisão com as raquetes
        if bola["x"] <= RAQUETE_WIDTH and player1_y <= bola["y"] <= player1_y + RAQUETE_HEIGHT:
            bola["dir_x"] *= -1
            bola["velocidade_x"] += 2
            bola["velocidade_y"] += 2
        elif bola["x"] >= screen_width - RAQUETE_WIDTH and player2_y <= bola["y"] <= player2_y + RAQUETE_HEIGHT:
            bola["dir_x"] *= -1
            bola["velocidade_x"] += 2
            bola["velocidade_y"] += 2
        elif bola["x"] <= 0:
            player2_score += 1
            bola["velocidade_x"] = 5
            bola["velocidade_y"] = 5
            reset_ball(bola, screen_width, screen_height)
        elif bola["x"] >= screen_width:
            bola["velocidade_x"] = 5
            bola["velocidade_y"] = 5
            player1_score += 1
            reset_ball(bola, screen_width, screen_height)
        
        # Verificar colisão com os obstáculos
        if obstacles_active:
            # Obstacle 1
            if (obstacle1["x"] <= bola["x"] <= obstacle1["x"] + obstacle1["width"]) and (obstacle1["y"] <= bola["y"] <= obstacle1["y"] + obstacle1["height"]):
                if obstacle1["x"] <= bola["x"] <= obstacle1["x"] + 10 or obstacle1["x"] + obstacle1["width"] - 10 <= bola["x"] <= obstacle1["x"] + obstacle1["width"]:
                    bola["dir_x"] *= -1
                elif obstacle1["y"] <= bola["y"] <= obstacle1["y"] + 10 or obstacle1["y"] + obstacle1["height"] - 10 <= bola["y"] <= obstacle1["y"] + obstacle1["height"]:
                    bola["dir_y"] *= -1

            # Obstacle 2
            if (obstacle2["x"] <= bola["x"] <= obstacle2["x"] + obstacle2["width"]) and (obstacle2["y"] <= bola["y"] <= obstacle2["y"] + obstacle2["height"]):
                if obstacle2["x"] <= bola["x"] <= obstacle2["x"] + 10 or obstacle2["x"] + obstacle2["width"] - 10 <= bola["x"] <= obstacle2["x"] + obstacle2["width"]:
                    bola["dir_x"] *= -1
                elif obstacle2["y"] <= bola["y"] <= obstacle2["y"] + 10 or obstacle2["y"] + obstacle2["height"] - 10 <= bola["y"] <= obstacle2["y"] + obstacle2["height"]:
                    bola["dir_y"] *= -1             

# Função para resetar a bola ao centro
def reset_ball(bola, screen_width, screen_height):
    bola['x'] = screen_width // 2
    bola['y'] = screen_height // 2
    bola['dir_x'] = np.random.choice([-1, 1])
    bola['dir_y'] = np.random.choice([-1, 1])
    bola["velocidade_x"] = 6
    bola["velocidade_y"] = 6

def inicial_set(screen_width, screen_height):
    global set, player1_y, player2_y, bola
    if not set:
        # Variáveis para o jogo
        player1_y = screen_height // 2 - RAQUETE_HEIGHT // 2
        player2_y = screen_height // 2 - RAQUETE_HEIGHT // 2
        bolas[1] = {
            "x": screen_width // 2,
            "y": screen_height // 2,
            "dir_x": np.random.choice([-1, 1]),
            "dir_y": np.random.choice([-1, 1]),
            "velocidade_x": 6,
            "velocidade_y": 6,
            "bola_tamanho": 15,
            "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        }
        set = True

def reset(screen_width, screen_height):
    global set, start_time, cont, player1_score, player2_score, bolas

    set = False

    bolas = {
        1: {
            "x": 0,
            "y": 0,
            "dir_x": 0,
            "dir_y": 0,
            "velocidade_x": 6,
            "velocidade_y": 6,
            "bola_tamanho": 15,
            "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        }
    }

    cont = 0

    # Inicia o temporizador
    start_time = time.time()

    player1_score = 0
    player2_score = 0

    inicial_set(screen_width, screen_height)

def exit():
    # Fecha todas as janelas do OpenCV
    cv.destroyAllWindows()
    cv.waitKey(1)  # Garante que todas as janelas sejam fechadas

    # Inicia o novo script usando subprocess.Popen
    subprocess.Popen([sys.executable, "./Pong/pong_menu.py"])

    # Encerra o script atual
    sys.exit()

def apply_powerup(powerup):
    global DURACAO_JOGO, BOLA_VELOCIDADE_X, BOLA_VELOCIDADE_Y, RAQUETE_HEIGHT, two_ball_active, additional_ball, obstacles_active, obstacle1, obstacle2
    
    if powerup == "MoreTime":
        DURACAO_JOGO += 60
    elif powerup == "2xVelocityBall":
        for bola_id, bola in bolas.items():
                bola["velocidade_x"] *= 2
                bola["velocidade_y"] *= 2
    elif powerup == "smallerrackets":
        RAQUETE_HEIGHT = int(RAQUETE_HEIGHT * 0.75)
    elif powerup == "moreBalls":
        bolas[len(bolas) + 1] = {
            "x": screen_width // 2,
            "y": screen_height // 2,
            "dir_x": random.choice([-1, 1]),
            "dir_y": random.choice([-1, 1]),
            "velocidade_x": 6,
            "velocidade_y": 6,
            "bola_tamanho": 15,
            "rgb": [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        }
    elif powerup == "obstacles":
        obstacles_active = True

        obstacle1 = {"x": screen_width // 3, "y": screen_height // random.randint(3, 4), "width": 50 * random.randint(1, 2), "height": 50 * random.randint(1, 2)}
        obstacle2 = {"x": 2 * screen_width // 3, "y": 2 * screen_height // random.randint(3, 4), "width": 50 * random.randint(1, 2), "height": 50 * random.randint(1, 2)}
    elif powerup == "racketsbiggest":
        RAQUETE_HEIGHT = int(RAQUETE_HEIGHT * 1.25)

def draw_roleta(frame):
    global roleta_powerup, roleta_ativa

    if roleta_ativa:
        cv.rectangle(frame, (screen_width // 2 - 150, screen_height // 2 - 150), (screen_width // 2 + 150, screen_height // 2 + 150), (0, 0, 0), -1)
        cv.putText(frame, "ROULETTE!", (screen_width // 2 - 100, screen_height // 2 - 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if roleta_powerup is None:
            roleta_powerup = random.choice(powerups)

        cv.putText(frame, roleta_powerup, (screen_width // 2 - 100, screen_height // 2), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


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

        if obstacles_active:
            cv.rectangle(frame, (obstacle1["x"], obstacle1["y"]), (obstacle1["x"] + obstacle1["width"], obstacle1["y"] + obstacle1["height"]), (255, 255, 0), -1)
            cv.rectangle(frame, (obstacle2["x"], obstacle2["y"]), (obstacle2["x"] + obstacle2["width"], obstacle2["y"] + obstacle2["height"]), (255, 255, 0), -1)

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
        for bola_id, bola in bolas.items():
                cv.circle(frame, (bola["x"], bola["y"]),
                        bola["bola_tamanho"], (bola["rgb"][0], bola["rgb"][1], bola["rgb"][2]), -1)

        # Exibe a pontuação
        cv.putText(frame, f"Player 1: {player1_score}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, f"Player 2: {player2_score}", (screen_width - 200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Adiciona o traço vertical no meio da tela
        overlay = frame.copy()
        cv.line(overlay, (screen_width // 2, 0), (screen_width // 2, screen_height), (0, 0, 0), 2)
        alpha = 0.5
        frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        current_time = time.time()

        if current_time - ultima_roleta_tempo >= ROLETA_INTERVALO:
            roleta_ativa = True
            roleta_start_time = current_time
            ultima_roleta_tempo = current_time

        if roleta_ativa:
            draw_roleta(frame)
            if current_time - roleta_start_time >= 3:
                apply_powerup(roleta_powerup)
                roleta_ativa = False
                roleta_powerup = None
        


        # Exibe o temporizador no centro superior da tela
        text_size = cv.getTextSize(timer_text, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (screen_width - text_size[0]) // 2
        cv.putText(frame, timer_text, (text_x, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)           

        cv.imshow('Pong', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(1) & 0xFF == ord('r'):
            reset()

vid.release()
cv.destroyAllWindows()
