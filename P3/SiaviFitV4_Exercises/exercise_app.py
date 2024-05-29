import cv2
import mediapipe as mp
import customtkinter as ctk
from tkinter import messagebox, Label, Toplevel
import screeninfo
from PIL import Image, ImageTk
import numpy as np
import os

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Variables to count exercises
hand_elevation_count = 0
hand_above_head = False
jumping_jack_count = 0
legs_apart = False
push_up_count = 0
push_up_down = False

selected_exercise = None
highlighted_button_index = 0
buttons = []
countdown_label = None

# Variables for sets and repetitions
current_set = 1
total_sets = 1
repetitions = 1

# Load pre-trained models for gender and age prediction
model_dir = "models"
face_proto = os.path.join(model_dir, "deploy.prototxt")
face_model = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
age_proto = os.path.join(model_dir, "age_deploy.prototxt")
age_model = os.path.join(model_dir, "age_net.caffemodel")
gender_proto = os.path.join(model_dir, "gender_deploy.prototxt")
gender_model = os.path.join(model_dir, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-22)', '(23-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

gender_age_label = None
detected_gender_age = ""
detected_once = False

# Function to detect gender and age
def detect_gender_age(frame):
    global detected_gender_age, detected_once
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), MODEL_MEAN_VALUES, swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            if face.shape[0] > 0 and face.shape[1] > 0:
                face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # Predict Gender
                gender_net.setInput(face_blob)
                gender_preds = gender_net.forward()
                gender = GENDER_LIST[gender_preds[0].argmax()]

                # Predict Age
                age_net.setInput(face_blob)
                age_preds = age_net.forward()
                age = AGE_LIST[age_preds[0].argmax()]

                detected_gender_age = f"{gender}, {age}"
                detected_once = True
                set_sliders_based_on_age(age)  # Set sliders based on detected age

    return frame


# Function to detect hand gestures
def detect_hand_gestures(hand_landmarks):
    thumb_is_open = hand_landmarks[4].y < hand_landmarks[3].y < hand_landmarks[2].y < hand_landmarks[1].y
    index_is_open = hand_landmarks[8].y < hand_landmarks[6].y < hand_landmarks[5].y
    middle_is_open = hand_landmarks[12].y < hand_landmarks[10].y < hand_landmarks[9].y
    ring_is_open = hand_landmarks[16].y < hand_landmarks[14].y < hand_landmarks[13].y
    pinky_is_open = hand_landmarks[20].y < hand_landmarks[18].y < hand_landmarks[17].y

    if not thumb_is_open and not index_is_open and not middle_is_open and not ring_is_open and not pinky_is_open:
        return "fist"
    elif index_is_open and middle_is_open and ring_is_open and pinky_is_open:
        return "open"
    return "unknown"


# Function to detect hand elevations
def detect_hand_elevations(image, landmarks):
    global hand_elevation_count, hand_above_head
    left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_hand = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    head_top = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y - 0.1]

    if left_hand[1] < head_top[1] or right_hand[1] < head_top[1]:
        if not hand_above_head:
            hand_elevation_count += 1
            hand_above_head = True
    else:
        hand_above_head = False

    cv2.putText(image, f'Hand Elevations: {hand_elevation_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA)


# Function to detect jumping jacks
def detect_jumping_jacks(image, landmarks):
    global jumping_jack_count, legs_apart
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    head_top = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y - 0.1]

    if (left_ankle[0] < 0.3 and right_ankle[0] > 0.7) and (
            left_wrist[1] < head_top[1] and right_wrist[1] < head_top[1]):
        if not legs_apart:
            jumping_jack_count += 1
            legs_apart = True
    else:
        legs_apart = False

    cv2.putText(image, f'Jumping Jacks: {jumping_jack_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                2, cv2.LINE_AA)


# Function to detect push-ups
def detect_push_ups(image, landmarks):
    global push_up_count, push_up_down
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    if shoulder_y > 0.5:
        if not push_up_down:
            push_up_down = True
    else:
        if push_up_down:
            push_up_count += 1
            push_up_down = False

    cv2.putText(image, f'Push-Ups: {push_up_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)


def start_exercise():
    global cap, mp_pose, pose, mp_drawing, mp_hands, hands, selected_exercise
    global hand_elevation_count, jumping_jack_count, push_up_count, current_set

    if selected_exercise is None:
        messagebox.showwarning("Exercise Selection", "Please select an exercise!")
        return

    cv2.namedWindow("Exercise", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Exercise", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while current_set <= total_sets:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            if selected_exercise == "Hand Elevations":
                detect_hand_elevations(image, landmarks)
                if hand_elevation_count >= repetitions:
                    hand_elevation_count = 0
                    current_set += 1
            elif selected_exercise == "Jumping Jacks":
                detect_jumping_jacks(image, landmarks)
                if jumping_jack_count >= repetitions:
                    jumping_jack_count = 0
                    current_set += 1
            elif selected_exercise == "Push-Ups":
                detect_push_ups(image, landmarks)
                if push_up_count >= repetitions:
                    push_up_count = 0
                    current_set += 1

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        except Exception as e:
            print(f"An error occurred: {e}")

        cv2.putText(image, f'Set: {current_set}/{total_sets}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Exercise", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def highlight_button(index):
    for i, btn in enumerate(buttons):
        if i == index:
            btn.configure(fg_color="green")
        else:
            btn.configure(fg_color="gray")


def set_exercise(exercise):
    global selected_exercise, current_set, hand_elevation_count, jumping_jack_count, push_up_count
    selected_exercise = exercise
    current_set = 1
    hand_elevation_count = 0
    jumping_jack_count = 0
    push_up_count = 0
    show_countdown(6)  # Start the countdown with 6 seconds


def show_countdown(seconds):
    def countdown():
        nonlocal seconds
        if seconds > 0:
            countdown_label.configure(text=f"{seconds} seconds remaining")
            seconds -= 1
            root.after(1000, countdown)
        else:
            countdown_label.pack_forget()  # Remove countdown label
            start_exercise()  # Start the exercise

    countdown_label.pack(pady=20)
    countdown()  # Start the countdown


def set_sliders_based_on_age(age):
    global sets_slider, reps_slider, difficulty_slider
    if age in ['(0-2)', '(4-6)', '(8-12)']:
        sets_slider.set(1)
        reps_slider.set(10)
        difficulty_slider.set(1)
    elif age in ['(15-22)', '(23-32)']:
        sets_slider.set(3)
        reps_slider.set(12)
        difficulty_slider.set(3)
    elif age in ['(38-43)', '(48-53)', '(60-100)']:
        sets_slider.set(2)
        reps_slider.set(6)
        difficulty_slider.set(2)


def create_ui():
    global buttons, highlighted_button_index, root, countdown_label, total_sets, repetitions, sets_slider, reps_slider, difficulty_slider, camera_label, gender_age_label, detected_once
    root = ctk.CTk()
    root.title("Exercise Selection")

    screen = screeninfo.get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    # Set the position and size of the Tkinter window
    root.attributes('-fullscreen', True)
    root.geometry(f"{screen_width}x{screen_height}")

    # Load icons after root is created
    dumbbell_icon = ImageTk.PhotoImage(Image.open("dumbbell_icon.png").resize((50, 50), Image.LANCZOS))
    sports_icon = ImageTk.PhotoImage(Image.open("sports_icon.png").resize((50, 50), Image.LANCZOS))
    pushup_icon = ImageTk.PhotoImage(Image.open("pushup_icon.png").resize((50, 50), Image.LANCZOS))

    # Create the frame on the left for exercise buttons
    left_frame = ctk.CTkFrame(root, width=400, height=300, corner_radius=15, fg_color="#2e2e26")
    left_frame.pack(side="left", pady=20, padx=20, fill="both", expand=True)

    label = ctk.CTkLabel(left_frame, text="Select an Exercise", font=("Helvetica", 50))
    label.pack(pady=35)

    instruction_label = ctk.CTkLabel(left_frame,
                                     text="Wave your left hand up or down to highlight.\nMake a fist to select.",
                                     font=("Helvetica", 20))
    instruction_label.pack(pady=30)

    btn_hand_elevations = ctk.CTkButton(left_frame, text="Hand Elevations", corner_radius=32, width=300, height=60,
                                        fg_color="gray",
                                        image=dumbbell_icon, compound="left",
                                        command=lambda: set_exercise("Hand Elevations"))
    btn_hand_elevations.pack(pady=10)

    btn_jumping_jacks = ctk.CTkButton(left_frame, text="Jumping Jacks", corner_radius=32, width=300, height=60,
                                      fg_color="gray",
                                      image=sports_icon, compound="left",
                                      command=lambda: set_exercise("Jumping Jacks"))
    btn_jumping_jacks.pack(pady=10)

    btn_push_ups = ctk.CTkButton(left_frame, text="Push-Ups", corner_radius=32, width=300, height=60, fg_color="gray",
                                 image=pushup_icon, compound="left",
                                 command=lambda: set_exercise("Push-Ups"))
    btn_push_ups.pack(pady=10)

    countdown_label = ctk.CTkLabel(left_frame, text="", font=("Helvetica", 24))
    buttons = [btn_hand_elevations, btn_jumping_jacks, btn_push_ups]
    highlight_button(highlighted_button_index)

    # Create the frame on the right for sliders
    right_frame = ctk.CTkFrame(root, width=400, height=300, corner_radius=15, fg_color="#2e2e26")
    right_frame.pack(side="right", pady=20, padx=20, fill="both", expand=True)

    sets_label = ctk.CTkLabel(right_frame, text="Number of Sets", font=("Helvetica", 20))
    sets_label.pack(pady=10)
    sets_slider = ctk.CTkSlider(right_frame, from_=1, to=3, number_of_steps=2, width=300)
    sets_slider.pack(pady=10)
    sets_slider.set(1)

    sets_help_label = ctk.CTkLabel(right_frame,
                                   text="Adjust the number of sets (1 to 3) using the slider.\nUse a fist gesture to change the value.",
                                   font=("Helvetica", 14))
    sets_help_label.pack(pady=5)

    reps_label = ctk.CTkLabel(right_frame, text="Number of Repetitions", font=("Helvetica", 20))
    reps_label.pack(pady=10)
    reps_slider = ctk.CTkSlider(right_frame, from_=1, to=12, number_of_steps=11, width=300)
    reps_slider.pack(pady=10)
    reps_slider.set(1)

    reps_help_label = ctk.CTkLabel(right_frame,
                                   text="Adjust the number of repetitions (1 to 12) using the slider.\nUse a fist gesture to change the value.",
                                   font=("Helvetica", 14))
    reps_help_label.pack(pady=5)

    difficulty_label = ctk.CTkLabel(right_frame, text="Difficulty", font=("Helvetica", 20))
    difficulty_label.pack(pady=10)
    difficulty_slider = ctk.CTkSlider(right_frame, from_=1, to=3, number_of_steps=2, width=300)
    difficulty_slider.pack(pady=10)
    difficulty_slider.set(1)

    difficulty_help_label = ctk.CTkLabel(right_frame,
                                         text="Adjust the difficulty level (1 to 3) using the slider.\nUse a fist gesture to change the value.",
                                         font=("Helvetica", 14))
    difficulty_help_label.pack(pady=5)

    # Create a label for displaying the camera feed
    camera_label = Label(root)
    camera_label.pack(pady=20)

    # Create a label for displaying gender and age information
    gender_age_label = ctk.CTkLabel(root, text="", font=("Helvetica", 24))
    gender_age_label.pack(pady=20)

    def update_ui():
        global highlighted_button_index, total_sets, repetitions, detected_gender_age, detected_once
        total_sets = int(sets_slider.get())
        repetitions = int(reps_slider.get())

        ret, frame = cap.read()

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        if selected_exercise is None and not detected_once:
            # Detect and label gender and age only if exercise is not started
            frame = detect_gender_age(frame)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        hand_results = hands.process(image)
        image.flags.writeable = True

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5:  # Left hand
                    gesture = detect_hand_gestures(hand_landmarks.landmark)
                    if gesture == "fist":
                        buttons[highlighted_button_index].invoke()
                    elif gesture == "open":
                        if hand_landmarks.landmark[9].y < 0.4:  # Adjusted threshold for more reliable detection
                            highlighted_button_index = max(0, highlighted_button_index - 1)
                        elif hand_landmarks.landmark[9].y > 0.6:  # Adjusted threshold for more reliable detection
                            highlighted_button_index = min(len(buttons) - 1, highlighted_button_index + 1)
                        highlight_button(highlighted_button_index)
                else:  # Right hand
                    # Adjust sliders with right hand gestures
                    gesture = detect_hand_gestures(hand_landmarks.landmark)
                    if gesture == "fist":
                        if hand_landmarks.landmark[9].y < 0.4:
                            sets_slider.set(min(sets_slider.get() + 1, 3))
                        elif hand_landmarks.landmark[9].y > 0.6:
                            sets_slider.set(max(sets_slider.get() - 1, 1))
                    elif gesture == "open":
                        if hand_landmarks.landmark[9].y < 0.4:
                            reps_slider.set(min(reps_slider.get() + 1, 12))
                        elif hand_landmarks.landmark[9].y > 0.6:
                            reps_slider.set(max(reps_slider.get() - 1, 1))
                    elif gesture == "open":
                        if hand_landmarks.landmark[9].y < 0.4:
                            difficulty_slider.set(min(difficulty_slider.get() + 1, 3))
                        elif hand_landmarks.landmark[9].y > 0.6:
                            difficulty_slider.set(max(difficulty_slider.get() - 1, 1))

        # Convert image to PhotoImage for Tkinter
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        root.after(100, update_ui)

    def display_gender_age_label():
        global detected_gender_age
        gender_age_label.configure(text=detected_gender_age)

    def schedule_gender_age_display():
        if detected_once:
            root.after(3000, display_gender_age_label)

    root.after(100, update_ui)
    root.after(3000, schedule_gender_age_display)  # Schedule the display of gender and age after 3 seconds
    root.mainloop()


if __name__ == "__main__":
    create_ui()
