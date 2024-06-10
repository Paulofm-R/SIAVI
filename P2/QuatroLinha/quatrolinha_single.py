import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

# Set up MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Constants for defining game board and cell size
COLS = 8
ROWS = 6
CELL_WIDTH = 60
CELL_HEIGHT = 60
GRID_WIDTH = COLS * CELL_WIDTH
GRID_HEIGHT = ROWS * CELL_HEIGHT
X_OFFSET = (640 - GRID_WIDTH) // 2
Y_OFFSET = (480 - GRID_HEIGHT) // 2

# Load gear icon image
gear_icon_path = "./QuatroLinha/gear_icon.png"  # Path to your gear icon image
gear_icon = cv2.imread(gear_icon_path, cv2.IMREAD_UNCHANGED)

# Initialize game board
board = np.zeros((ROWS, COLS))
player_turn = 1
game_over = False
selected_column = 0  # Initialize selected column

# Função para mapear a coordenada x da mão para o índice da coluna
def map_hand_to_column(hand_x):
    return min(max(int((hand_x - X_OFFSET) / CELL_WIDTH), 0), COLS - 1)

# Função para deixar o token cair em uma coluna
def drop_token(column):
    for row in range(ROWS - 1, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player_turn
            return row, column
    return None

# Função para desenhar o tabuleiro
def draw_grid(frame):
    for i in range(COLS + 1):
        cv2.line(frame, (X_OFFSET + i * CELL_WIDTH, Y_OFFSET),
                 (X_OFFSET + i * CELL_WIDTH, Y_OFFSET + GRID_HEIGHT),
                 (255, 255, 255), 2)
    for j in range(ROWS + 1):
        cv2.line(frame, (X_OFFSET, Y_OFFSET + j * CELL_HEIGHT),
                 (X_OFFSET + GRID_WIDTH, Y_OFFSET + j * CELL_HEIGHT),
                 (255, 255, 255), 2)

    cv2.rectangle(frame, (X_OFFSET + selected_column * CELL_WIDTH, Y_OFFSET),
                  (X_OFFSET + (selected_column + 1) * CELL_WIDTH, Y_OFFSET + GRID_HEIGHT),
                  (0, 255, 0), 2)

    gear_icon_resized = cv2.resize(gear_icon, (40, 40))
    gear_icon_resized_bgr = gear_icon_resized[:, :, :3]
    frame[Y_OFFSET - 50:Y_OFFSET - 10, X_OFFSET + GRID_WIDTH - 50:X_OFFSET + GRID_WIDTH - 10] = gear_icon_resized_bgr

# Função para verificar se houve uma vitória
def check_win(board, player):
    for row in range(ROWS):
        for col in range(COLS - 3):
            if board[row][col] == player and board[row][col + 1] == player and board[row][col + 2] == player and \
                    board[row][col + 3] == player:
                return True

    for col in range(COLS):
        for row in range(ROWS - 3):
            if board[row][col] == player and board[row + 1][col] == player and board[row + 2][col] == player and \
                    board[row + 3][col] == player:
                return True

    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if board[row][col] == player and board[row + 1][col + 1] == player and board[row + 2][col + 2] == player and \
                    board[row + 3][col + 3] == player:
                return True

    for row in range(ROWS - 3):
        for col in range(3, COLS):
            if board[row][col] == player and board[row + 1][col - 1] == player and board[row + 2][col - 2] == player and \
                    board[row + 3][col - 3] == player:
                return True

    return False

# Função para a jogada do computador
def computer_play():
    available_columns = [col for col in range(COLS) if 0 in board[:, col]]
    if available_columns:
        return random.choice(available_columns)
    else:
        return None

# Main loop
cap = cv2.VideoCapture(0)
play_delay = 2
last_play_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and not game_over:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]

            if math.sqrt((hand_x - thumb_x) ** 2 + (hand_y - thumb_y) ** 2) < 30:
                selected_column = map_hand_to_column(hand_x)
                current_time = time.time()
                if current_time - last_play_time >= play_delay:
                    if player_turn == 1:
                        play_result = drop_token(selected_column)
                        if play_result:
                            if check_win(board, player_turn):
                                print("Player", player_turn, "wins!")
                                game_over = True
                            else:
                                player_turn = 2
                                last_play_time = current_time
                                print("Player", player_turn, "plays.")                        

    if player_turn == 2:
        computer_column = computer_play()
        if computer_column is not None:
            play_result = drop_token(computer_column)
            if play_result:
                if check_win(board, player_turn):
                    print("Player", player_turn, "wins!")
                    game_over = True
                else:
                    player_turn = 1
                    print("Player", player_turn, "plays.")

    draw_grid(frame)

    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] == 1:
                cv2.circle(frame,
                           (X_OFFSET + c * CELL_WIDTH + CELL_WIDTH // 2, Y_OFFSET + r * CELL_HEIGHT + CELL_HEIGHT // 2),
                           min(CELL_WIDTH, CELL_HEIGHT) // 3, (0, 0, 255), -1)
            elif board[r][c] == 2:
                cv2.circle(frame,
                           (X_OFFSET + c * CELL_WIDTH + CELL_WIDTH // 2, Y_OFFSET + r * CELL_HEIGHT + CELL_HEIGHT // 2),
                           min(CELL_WIDTH, CELL_HEIGHT) // 3, (0, 255, 255), -1)

    cv2.imshow('Connect Four - Single', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
