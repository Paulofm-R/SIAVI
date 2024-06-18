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

if gear_icon is None:
    print("Error: Gear icon image not found at path:", gear_icon_path)

# Initialize game board
board = np.zeros((ROWS, COLS))
player_turn = 1
game_over = False
selected_column = None  # Initialize selected column as None


# Reset button parameters
RESET_BUTTON_X = 50  # X position of the reset button
RESET_BUTTON_Y = 620  # Y position of the reset button
RESET_BUTTON_SIZE = 100  # Size of the reset button
reset_active = False  # Flag to indicate if reset is active







# Constants for gesture sensitivity and smoothing
THUMBS_UP_THRESHOLD = 0.9  # Adjust as needed
SMOOTHING_ALPHA = 0.5  # Smoothing factor
THUMBS_UP_FRAME_COUNT = 15  # Number of frames to hold the gesture

# State variables for gesture detection
gesture_state = "No Gesture"
gesture_start_time = 0

# Function to map hand x coordinate to column index
def map_hand_to_column(hand_x):
    return min(max(int((hand_x - X_OFFSET) / CELL_WIDTH), 0), COLS - 1)

# Function to drop a token in a column
def drop_token(column):
    for row in range(ROWS - 1, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player_turn
            return row, column
    return None

# Function to draw the game board
def draw_grid(frame):
    for i in range(COLS + 1):
        cv2.line(frame, (X_OFFSET + i * CELL_WIDTH, Y_OFFSET),
                 (X_OFFSET + i * CELL_WIDTH, Y_OFFSET + GRID_HEIGHT),
                 (255, 255, 255), 2)
    for j in range(ROWS + 1):
        cv2.line(frame, (X_OFFSET, Y_OFFSET + j * CELL_HEIGHT),
                 (X_OFFSET + GRID_WIDTH, Y_OFFSET + j * CELL_HEIGHT),
                 (255, 255, 255), 2)

    if selected_column is not None:
        cv2.rectangle(frame, (X_OFFSET + selected_column * CELL_WIDTH, Y_OFFSET),
                      (X_OFFSET + (selected_column + 1) * CELL_WIDTH, Y_OFFSET + GRID_HEIGHT),
                      (0, 255, 0), 2)

    if gear_icon is not None:
        gear_icon_resized = cv2.resize(gear_icon, (40, 40))
        gear_icon_resized_bgr = gear_icon_resized[:, :, :3]
        alpha_channel = gear_icon_resized[:, :, 3]
        alpha_mask = alpha_channel / 255.0

        for c in range(3):
            frame[Y_OFFSET - 50:Y_OFFSET - 10, X_OFFSET + GRID_WIDTH - 50:X_OFFSET + GRID_WIDTH - 10, c] = \
                (alpha_mask * gear_icon_resized_bgr[:, :, c] + (1 - alpha_mask) * frame[Y_OFFSET - 50:Y_OFFSET - 10,
                                                                                  X_OFFSET + GRID_WIDTH - 50:X_OFFSET + GRID_WIDTH - 10,
                                                                                  c])

# Function to check for a win
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

# Function for the computer's play
def computer_play():
    available_columns = [col for col in range(COLS) if 0 in board[:, col]]
    best_score = -float('inf')
    best_column = random.choice(available_columns)

    for col in available_columns:
        row = next(r for r in range(ROWS - 1, -1, -1) if board[r][col] == 0)
        board[row][col] = 2
        if check_win(board, 2):
            board[row][col] = 0
            return col
        board[row][col] = 0

    for col in available_columns:
        row = next(r for r in range(ROWS - 1, -1, -1) if board[r][col] == 0)
        board[row][col] = 1
        if check_win(board, 1):
            board[row][col] = 0
            return col
        board[row][col] = 0

    for col in available_columns:
        row = next(r for r in range(ROWS - 1, -1, -1) if board[r][col] == 0)
        board[row][col] = 2
        score = evaluate_board(board, 2)
        board[row][col] = 0
        if score > best_score:
            best_score = score
            best_column = col

    return best_column

# Function to evaluate the board and score potential moves
def evaluate_board(board, player):
    score = 0
    center_array = [int(i) for i in list(board[:, COLS // 2])]
    center_count = center_array.count(player)
    score += center_count * 3

    for row in range(ROWS):
        row_array = [int(i) for i in list(board[row, :])]
        for col in range(COLS - 3):
            window = row_array[col:col + 4]
            score += evaluate_window(window, player)

    for col in range(COLS):
        col_array = [int(i) for i in list(board[:, col])]
        for row in range(ROWS - 3):
            window = col_array[row:row + 4]
            score += evaluate_window(window, player)

    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            window = [board[row + i][col + i] for i in range(4)]
            score += evaluate_window(window, player)

    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            window = [board[row + 3 - i][col + i] for i in range(4)]
            score += evaluate_window(window, player)

    return score

# Function to evaluate a window of 4 cells
def evaluate_window(window, player):
    score = 0
    opponent = 1 if player == 2 else 2

    if window.count(player) == 4:
        score += 100
    elif window.count(player) == 3 and window.count(0) == 1:
        score += 5
    elif window.count(player) == 2 and window.count(0) == 2:
        score += 2

    if window.count(opponent) == 3 and window.count(0) == 1:
        score -= 4

    return score


 # Draw reset button
    cv2.rectangle(frame, (RESET_BUTTON_X, RESET_BUTTON_Y),
                  (RESET_BUTTON_X + RESET_BUTTON_SIZE, RESET_BUTTON_Y + RESET_BUTTON_SIZE),
                  (255, 255, 255), 2)
    cv2.putText(frame, "RESET", (RESET_BUTTON_X + 10, RESET_BUTTON_Y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)




# Function to display the win screen and reset button
def display_win_screen(frame, player):
    text = f"{'Computer' if player == 2 else 'Player'} {player} Wins!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA

                )
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

    # Draw reset button
    reset_text = "Press 'r' to Reset"
    reset_text_size = cv2.getTextSize(reset_text, font, 1, 2)[0]
    reset_text_x = (frame.shape[1] - reset_text_size[0]) // 2
    reset_text_y = text_y + text_size[1] + reset_text_size[1]
    cv2.putText(frame, reset_text, (reset_text_x, reset_text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Check for reset input
    if cv2.waitKey(1) & 0xFF == ord('r'):
        reset_game()

# Function to reset the game
def reset_game():
    global board, player_turn, game_over, selected_column
    board = np.zeros((ROWS, COLS))
    player_turn = 1
    game_over = False
    selected_column = None

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

            # Draw landmarks for debugging
            cv2.circle(frame, (int(hand_x), int(hand_y)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(thumb_x), int(thumb_y)), 5, (0, 0, 255), -1)

            # Detect thumbs up gesture (index finger and thumb close together)
            if math.sqrt((hand_x - thumb_x) ** 2 + (hand_y - thumb_y) ** 2) < 30:
                if gesture_state == "No Gesture":
                    gesture_state = "Thumbs Up"
                    gesture_start_time = time.time()
                elif gesture_state == "Thumbs Up" and time.time() - gesture_start_time >= THUMBS_UP_FRAME_COUNT / 30:
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

    else:
        gesture_state = "No Gesture"

    if player_turn == 2 and not game_over:
        computer_column = computer_play()
        if computer_column is not None:
            play_result = drop_token(computer_column)
            if play_result:
                if check_win(board, player_turn):
                    print("Computer wins!")
                    game_over = True
                else:
                    player_turn = 1
                    print("Player 1 plays.")

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

    if game_over:
        display_win_screen(frame, player_turn)

    cv2.imshow('Connect Four - Single', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()


