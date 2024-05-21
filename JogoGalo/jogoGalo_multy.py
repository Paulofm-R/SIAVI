import cv2 as cv
import mediapipe as mp

win = ''
player1_wins = 0
player2_wins = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cont = 0

player_turn = "player1"

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


def set_squares(frame_height, frame_width):
    global squares, resetSquare
    # Número de linhas e colunas
    num_rows = 3
    num_cols = 3

    # Tamanho dos quadrados
    square_size = min(frame_width, frame_height) // max(num_rows, num_cols)

    # Desenhar os quadrados na tela
    for i in range(num_rows):
        for j in range(num_cols):
            square_number = i * num_cols + j + 1
            x1 = j * square_size + (frame_width - num_cols * square_size) // 2
            y1 = i * square_size + (frame_height - num_rows * square_size) // 2
            x2 = x1 + square_size
            y2 = y1 + square_size

            squares[square_number]["x1"] = x1
            squares[square_number]["y1"] = y1
            squares[square_number]["x2"] = x2
            squares[square_number]["y2"] = y2

    resetSquare["x1"] = 100
    resetSquare["y1"] = int((frame_height / 2) - 82)
    resetSquare["x2"] = 300
    resetSquare["y2"] = int((frame_height / 2) + 82)


def getHandMove(hand_landmarks):
    global squares, cont, player_turn

    landmarks = hand_landmarks.landmark

    indicador_x = int(landmarks[8].x * frame_width)
    indicador_y = int(landmarks[8].y * frame_height)

    finger_in_square = False  # Flag para indicar se o dedo está em algum quadrado

    if win == "":
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

                        # trocar a vez do jogador
                        player_turn = "player1" if player_turn == "player2" else "player2"
    else:
        if resetSquare["x1"] < indicador_x < resetSquare["x2"] and resetSquare["y1"] < indicador_y < resetSquare["y2"]:
            finger_in_square = True
            cont += 1
            if cont >= 25:  # Se o dedo estiver no local por 50 frames
                reset()

    if not finger_in_square:
        cont = 0  # Resetar o contador se o dedo não estiver em nenhum quadrado


def reset():
    global win, squares

    win = ""
    for square in squares.values():
        square["player1"] = False
        square["player2"] = False


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

        set_squares(frame_height, frame_width)

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

        if win != "":  # Para não ser afetado com a inversão da tela
            cv.rectangle(frame, (resetSquare["x1"], resetSquare["y1"]),
                         (resetSquare["x2"], resetSquare["y2"]), (174, 173, 178), 3)

        frame = cv.flip(frame, 1)

        # Exibe a pontuação
        cv.putText(frame, f"Player 1: {player1_wins}", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame, f"Player 2: {player2_wins}", (frame_width -
                   200, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if win != "":
            cv.rectangle(frame, (int((frame_width / 2) - 200), int((frame_height / 2) - 75)),
                         (int((frame_width / 2) + 200), int((frame_height / 2) + 75)), (0, 0, 0), -1)
            cv.putText(frame, f"{win.upper()} WIN!!", (int((frame_width / 2) - 170), int(
                (frame_height / 2) + 20)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            cv.putText(frame, "RESTAR", (frame_width - 255, int((frame_height / 2) - 90)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Resize frame to increase its size
        frame = cv.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

        cv.imshow('frame', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


vid.release()
cv.destroyAllWindows()
