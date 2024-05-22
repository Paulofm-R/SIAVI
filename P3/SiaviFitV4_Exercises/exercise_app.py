import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Variables to count hand elevations
hand_elevation_count = 0
hand_above_head = False

# Variables to count jumping jacks
jumping_jack_count = 0
legs_apart = False

# Variables to count push-ups
push_up_count = 0
push_up_down = False

# Function to detect hand gestures
def detect_hand_gestures(hand_landmarks):
    thumb_is_open = hand_landmarks[4].y < hand_landmarks[3].y < hand_landmarks[2].y < hand_landmarks[1].y
    index_is_open = hand_landmarks[8].y < hand_landmarks[6].y < hand_landmarks[5].y
    middle_is_open = hand_landmarks[12].y < hand_landmarks[10].y < hand_landmarks[9].y
    ring_is_open = hand_landmarks[16].y < hand_landmarks[14].y < hand_landmarks[13].y
    pinky_is_open = hand_landmarks[20].y < hand_landmarks[18].y < hand_landmarks[17].y

    if thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "thumb"
    elif not thumb_is_open and index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "index"
    elif not thumb_is_open and index_is_open and middle_is_open and not ring_is_open and not pinky_is_open:
        return "index_middle"
    else:
        return "unknown"

# Function to detect hand elevations
def detect_hand_elevations(image, landmarks):
    global hand_elevation_count, hand_above_head
    left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    head_top = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y - 0.1]

    if left_hand[1] < head_top[1] or right_hand[1] < head_top[1]:
        if not hand_above_head:
            hand_elevation_count += 1
            hand_above_head = True
    else:
        hand_above_head = False

    cv2.putText(image, f'Hand Elevations: {hand_elevation_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Function to detect jumping jacks
def detect_jumping_jacks(image, landmarks):
    global jumping_jack_count, legs_apart
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    head_top = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y - 0.1]

    if (left_ankle[0] < 0.3 and right_ankle[0] > 0.7) and (left_wrist[1] < head_top[1] and right_wrist[1] < head_top[1]):
        if not legs_apart:
            jumping_jack_count += 1
            legs_apart = True
    else:
        legs_apart = False

    cv2.putText(image, f'Jumping Jacks: {jumping_jack_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Function to detect push-ups
def detect_push_ups(image, landmarks):
    global push_up_count, push_up_down
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    if shoulder_y > 0.5:
        if not push_up_down:
            push_up_down = True
    else:
        if push_up_down:
            push_up_count += 1
            push_up_down = False

    cv2.putText(image, f'Push-Ups: {push_up_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    global cap, mp_pose, pose, mp_drawing, mp_hands, hands, mp_drawing_styles

    selected_exercise = None

    cv2.namedWindow('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Mediapipe Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        hand_results = hands.process(image)

        image.flags.writeable = True

        instructions = [
            "Show a gesture to select an exercise:",
            "Thumb up: Hand Elevations",
            "Index finger: Jumping Jacks",
            "Index and middle fingers: Push-Ups"
        ]

        y0, dy = 30, 30
        for i, line in enumerate(instructions):
            y = y0 + i * dy
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 255, 255), 2, cv2.LINE_AA)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

                gesture = detect_hand_gestures(hand_landmarks.landmark)
                if gesture == "thumb":
                    selected_exercise = "Hand Elevations"
                    detect_exercise = detect_hand_elevations
                    break
                elif gesture == "index":
                    selected_exercise = "Jumping Jacks"
                    detect_exercise = detect_jumping_jacks
                    break
                elif gesture == "index_middle":
                    selected_exercise = "Push-Ups"
                    detect_exercise = detect_push_ups
                    break

        if selected_exercise:
            print(f"Starting {selected_exercise} exercise...")
            break

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Mediapipe Feed', image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            detect_exercise(image, landmarks)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        except Exception as e:
            print(f"An error occurred: {e}")
            pass

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
