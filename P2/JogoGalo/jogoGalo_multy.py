import cv2 as cv
import mediapipe as mp
import random
import subprocess
import sys
from screeninfo import get_monitors

win = ''
player1_wins = 0
player2_wins = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cont = 0

player_turn = random.choice(["player1", "player2"])
moves = 0

squares = {
    1: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    }, 2: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    }, 3: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    },
    4: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    },
    5: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    }, 6: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    }, 7: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    }, 8: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    }, 9: {
        "x1": 0,
        "y1": 0,
        "x2": 0,
        "y2": 0,
        "player1": False,
        "player2": False,
    },
}

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


def set_squares(screen_height, screen_width):
    global squares, resetSquare
    # Número de linhas e colunas
    num_rows = 3
    num_cols = 3

    # Tamanho dos quadrados
    square_size = min(screen_width, screen_height) // max(num_rows, num_cols)

    # Desenhar os quadrados na tela
    for i in range(num_rows):
        for j in range(num_cols):
            square_number = i * num_cols + j + 1
            x1 = j * square_size + (screen_width - num_cols * square_size) // 2
            y1 = i * square_size + (screen_height - num_rows * square_size) // 2
            x2 = x1 + square_size
            y2 = y1 + square_size

            squares[square_number]["x1"] = x1
            squares[square_number]["y1"] = y1
            squares[square_number]["x2"] = x2
            squares[square_number]["y2"] = y2

    resetSquare["x1"] = 100
    resetSquare["y1"] = int((screen_height / 2) - 82)
    resetSquare["x2"] = 300
    resetSquare["y2"] = int((screen_height / 2) + 82)

    exitSquare["x1"] = screen_width - 100
    exitSquare["y1"] = int((screen_height / 2) - 82)
    exitSquare["x2"] = screen_width - 300
    exitSquare["y2"] = int((screen_height / 2) + 82)


def getHandMove(hand_landmarks):
    global squares, cont, player_turn, moves

    landmarks = hand_landmarks.landmark

    indicador_x = int(landmarks[8].x * screen_width)
    indicador_y = int(landmarks[8].y * screen_height)

    finger_in_square = False  # Flag para indicar se o dedo está em algum quadrado

    if win == "" and moves < 9:
        for square_id, square_data in squares.items():
            x1, y1, x2, y2 = square_data["x1"], square_data["y1"], square_data["x2"], square_data["y2"]
            if x1 < indicador_x < x2 and y1 < indicador_y < y2:
                if not square_data['player1'] and not square_data['player2']:
                    finger_in_square = True
                    cont += 1
                    if cont >= 25:  # Se o dedo estiver no local por 50 frames
                        square_data[player_turn] = True
                        cont = 0  # Quando colocar o circulo ou o x

                        # verificar se alguem ganhou
                        check_winner(squares)
                        if win != "":
                            print(win)
                        else:
                            moves += 1

                        # trocar a vez do jogador
                        player_turn = "player1" if player_turn == "player2" else "player2"
    else:
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

def check_winner(squares):
    global win, player1_wins, player2_wins
    # Verificar linhas
    for i in range(1, 10, 3):
        if squares[i]["player1"] and squares[i+1]["player1"] and squares[i+2]["player1"]:
            win = "player1"
        elif squares[i]["player2"] and squares[i+1]["player2"] and squares[i+2]["player2"]:
            win = "player2"

    # Verificar colunas
    for i in range(1, 4):
        if squares[i]["player1"] and squares[i+3]["player1"] and squares[i+6]["player1"]:
            win = "player1"
        elif squares[i]["player2"] and squares[i+3]["player2"] and squares[i+6]["player2"]:
            win = "player2"

    # Verificar diagonal principal
    if squares[1]["player1"] and squares[5]["player1"] and squares[9]["player1"]:
        win = "player1"
    elif squares[1]["player2"] and squares[5]["player2"] and squares[9]["player2"]:
        win = "player2"

    # Verificar diagonal secundária
    if squares[3]["player1"] and squares[5]["player1"] and squares[7]["player1"]:
        win = "player1"
    elif squares[3]["player2"] and squares[5]["player2"] and squares[7]["player2"]:
        win = "player2"

    if win == "player1":
        player1_wins += 1
    elif win == "player2":
        player2_wins += 1

def reset():
    global win, squares, moves, player_turn

    win = ""
    moves = 0
    player_turn = random.choice(["player1", "player2"])
    for square in squares.values():
        square["player1"] = False
        square["player2"] = False

def exit():
    # Fecha todas as janelas do OpenCV
    cv.destroyAllWindows()
    cv.waitKey(1)  # Garante que todas as janelas sejam fechadas

    # Inicia o novo script usando subprocess.Popen
    subprocess.Popen([sys.executable, "./JogoGalo/jogoGalo_menu.py"])

    # Encerra o script atual
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

        set_squares(screen_height, screen_width)

        for square_number, square_info in squares.items():
            x1, y1, x2, y2 = square_info["x1"], square_info["y1"], square_info["x2"], square_info["y2"]
            cv.rectangle(frame, (int(x1), int(y1)),
                         (int(x2), int(y2)), (150, 150, 150), 1)

            if square_info['player1']:
                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)
                cv.circle(frame, (centro_x, centro_y), 75, (0, 0, 255), 5)
            elif square_info['player2']:
                centro_x = int((x1 + x2) / 2)
                centro_y = int((y1 + y2) / 2)
                cv.line(frame, (centro_x - 50, centro_y - 50),
                        (centro_x + 50, centro_y + 50), (255, 0, 0), 5)
                cv.line(frame, (centro_x - 50, centro_y + 50),
                        (centro_x + 50, centro_y - 50), (255, 0, 0), 4)

        hls = results.multi_hand_landmarks
        if hls and len(hls) == 1:
            getHandMove(hls[0])

        if win != "" or moves >= 9:  # Para não ser afetado com a inversão da tela
            cv.rectangle(frame, (resetSquare["x1"], resetSquare["y1"]),
                         (resetSquare["x2"], resetSquare["y2"]), (174, 173, 178), 3)
            cv.rectangle(frame, (exitSquare["x1"], exitSquare["y1"]),
                         (exitSquare["x2"], exitSquare["y2"]), (174, 173, 178), 3)

        frame = cv.flip(frame, 1)

        # Exibir quem joga
        if player_turn == "player1":
            cv.rectangle(frame, (40, 18),
                    (240, 60), (0, 255, 0), -1)
        else:
            cv.rectangle(frame, (screen_width - 210, 18),
                    (screen_width - 10, 60), (0, 255, 0), -1)

        # Exibe a pontuação
        cv.putText(frame, f"Player 1: {player1_wins}", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, f"Player 2: {player2_wins}", (screen_width -
                   200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if win != "" or moves >= 9:
            final_sentence = ""
            if win != "":
                final_sentence = f"{win.upper()} WIN!!"
            else:
                final_sentence = "DRAW!!"

            cv.rectangle(frame, (int((screen_width / 2) - 200), int((screen_height / 2) - 75)),
                         (int((screen_width / 2) + 200), int((screen_height / 2) + 75)), (0, 0, 0), -1)
            cv.putText(frame, final_sentence, (int((screen_width / 2) - 165), int(
                (screen_height / 2) + 20)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv.putText(frame, "RESET", (screen_width - 255, int((screen_height / 2) - 90)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(frame, "EXIT", (150, int((screen_height / 2) - 90)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


vid.release()
cv.destroyAllWindows()
