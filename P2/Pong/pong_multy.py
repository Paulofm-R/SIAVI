import cv2 as cv
import mediapipe as mp
import numpy as np

# Definir constantes para as raquetes
RAQUETE_WIDTH = 20
RAQUETE_HEIGHT = 150
RAQUETE_VELOCIDADE = 5

# Definir constantes para a bola
BOLA_SIZE = 15
BOLA_VELOCIDADE_X = 6
BOLA_VELOCIDADE_Y = 6

player1_score = 0
player2_score = 0

player1_y = 0
player2_y = 0
bola_x = 0
bola_y = 0
bola_dir_x = 0
bola_dir_y = 0

set = False

def update_ball_position():
    global bola_x, bola_y, bola_dir_x, bola_dir_y, player1_score, player2_score, BOLA_VELOCIDADE_X, BOLA_VELOCIDADE_Y

    # Atualizar posição da bola
    bola_x += BOLA_VELOCIDADE_X * bola_dir_x
    bola_y += BOLA_VELOCIDADE_Y * bola_dir_y

    # Verificar colisões com as paredes
    if bola_y <= 0 or bola_y >= frame_height:
        bola_dir_y *= -1

    # Verificar colisão com as raquetes
    if bola_x <= RAQUETE_WIDTH and player1_y <= bola_y <= player1_y + RAQUETE_HEIGHT:
        bola_dir_x *= -1
        BOLA_VELOCIDADE_X += 2
        BOLA_VELOCIDADE_Y += 2
    elif bola_x >= frame_width - RAQUETE_WIDTH and player2_y <= bola_y <= player2_y + RAQUETE_HEIGHT:
        bola_dir_x *= -1
        BOLA_VELOCIDADE_X += 2
        BOLA_VELOCIDADE_Y += 2
    elif bola_x <= 0:
        player2_score += 1
        BOLA_VELOCIDADE_X = 5
        BOLA_VELOCIDADE_Y = 5
        reset_ball()
    elif bola_x >= frame_width:
        BOLA_VELOCIDADE_X = 5
        BOLA_VELOCIDADE_Y = 5
        player1_score += 1
        reset_ball()

# Função para resetar a bola ao centro


def reset_ball():
    global bola_x, bola_y, bola_dir_x, bola_dir_y
    bola_x = frame_width // 2
    bola_y = frame_height // 2
    bola_dir_x = np.random.choice([-1, 1])
    bola_dir_y = np.random.choice([-1, 1])

def inicial_set(frame_width, frame_height):
    global set, player1_y, player2_y, bola_x, bola_y, bola_dir_x, bola_dir_y
    if not set:
        # Variáveis para o jogo
        player1_y = frame_height // 2 - RAQUETE_HEIGHT // 2
        player2_y = frame_height // 2 - RAQUETE_HEIGHT // 2
        bola_x = frame_width // 2
        bola_y = frame_height // 2
        bola_dir_x = np.random.choice([-1, 1])
        bola_dir_y = np.random.choice([-1, 1])
        set = True


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

                # Obtém as coordenadas da ponta do dedo médio
                middle_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_tip_x = int(middle_finger_tip.x * frame_width)
                middle_finger_tip_y = int(middle_finger_tip.y * frame_height)

                # Ajuste das coordenadas para a largura da janela
                middle_finger_tip_x = frame_width - middle_finger_tip_x

                # Define a posição da raquete do jogador
                # Jogador 1 (esquerda)
                if middle_finger_tip_x < frame_width // 2:
                    player1_y = middle_finger_tip_y - RAQUETE_HEIGHT // 2
                else:  # Jogador 2 (direita)
                    player2_y = middle_finger_tip_y - RAQUETE_HEIGHT // 2

        # Atualiza a posição da bola
        update_ball_position()

        frame = cv.flip(frame, 1)
        # Resize frame to increase its size
        frame = cv.resize(frame, (frame_width, frame_height))

        if not set:
            inicial_set(frame_width, frame_height)

        # Desenha as raquetes e a bola
        cv.rectangle(frame, (0, player1_y), (RAQUETE_WIDTH, player1_y + RAQUETE_HEIGHT), (0, 0, 255), -1)
        cv.rectangle(frame, (frame_width - RAQUETE_WIDTH, player2_y), (frame_width, player2_y + RAQUETE_HEIGHT), (0, 255, 0), -1)
        cv.circle(frame, (bola_x, bola_y), BOLA_SIZE, (255, 0, 0), -1)

        # Exibe a pontuação
        cv.putText(frame, f"Player 1: {player1_score}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, f"Player 2: {player2_score}", (frame_width - 200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        cv.imshow('Pong', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

vid.release()
cv.destroyAllWindows()
