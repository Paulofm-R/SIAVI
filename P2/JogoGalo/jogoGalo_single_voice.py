import customtkinter as ctk
import tkinter as tk
import speech_recognition as sr
import threading
import random

# Configurações iniciais da interface
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Jogo do Galo - Jogador vs Computador")
        self.root.geometry("425x500")
        self.board = [" "]*9
        self.current_player = "X"
        self.player_score = 0
        self.computer_score = 0
        self.buttons = []
        self.create_widgets()
        self.voice_thread = threading.Thread(target=self.voice_command_listener, daemon=True)
        self.voice_thread.start()

    def create_widgets(self):
        self.player_score_label = ctk.CTkLabel(self.root, text=f"Jogador: {self.player_score}", font=("Arial", 18))
        self.player_score_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.computer_score_label = ctk.CTkLabel(self.root, text=f"Computador: {self.computer_score}", font=("Arial", 18))
        self.computer_score_label.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        for i in range(9):
            button = ctk.CTkButton(self.root, text=str(i+1), width=120, height=120, font=("Arial", 24), command=lambda i=i: self.player_move(i))
            button.grid(row=(i//3) + 1, column=i%3, padx=6, pady=6)
            self.buttons.append(button)

    def player_move(self, index):
        if self.board[index] == " " and self.current_player == "X":
            self.board[index] = "X"
            self.buttons[index].configure(text="X")
            if not self.check_winner():
                self.current_player = "O"
                self.computer_move()

    def computer_move(self):
        move = self.find_winning_move("O")
        if move is None:
            move = self.find_winning_move("X")
        if move is None:
            move = self.choose_random_move()

        if move is not None:
            self.board[move] = "O"
            self.buttons[move].configure(text="O")
            self.current_player = "X"
            self.check_winner()

    def find_winning_move(self, player):
        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for combo in winning_combinations:
            values = [self.board[i] for i in combo]
            if values.count(player) == 2 and values.count(" ") == 1:
                return combo[values.index(" ")]
        return None

    def choose_random_move(self):
        available_moves = [i for i, x in enumerate(self.board) if x == " "]
        if available_moves:
            return random.choice(available_moves)
        return None

    def check_winner(self):
        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for combo in winning_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != " ":
                winner = self.board[combo[0]]
                self.update_score(winner)
                self.show_winner(winner)
                return True
        if " " not in self.board:
            self.show_winner("Empate")
            return True
        return False

    def update_score(self, winner):
        if winner == "X":
            self.player_score += 1
            self.player_score_label.configure(text=f"Jogador: {self.player_score}")
        elif winner == "O":
            self.computer_score += 1
            self.computer_score_label.configure(text=f"Computador: {self.computer_score}")

    def show_winner(self, winner):
        if winner == "Empate":
            message = "Empate!"
        else:
            message = f"Jogador {winner} venceu!"
        self.reset_board()
        result_label = ctk.CTkLabel(self.root, text=message, font=("Arial", 24))
        result_label.grid(row=5, column=0, columnspan=3, pady=10)
        self.root.after(2000, result_label.destroy)

    def reset_board(self):
        self.board = [" "]*9
        for i, button in enumerate(self.buttons):
            button.configure(text=str(i+1))
        self.current_player = "X"

    def voice_command_listener(self):
        recognizer = sr.Recognizer()
        while True:
            try:
                with sr.Microphone() as source:
                    print("Listening for commands...")
                    audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio, language='pt-PT')
                print(f"Você disse: {command}")
                self.process_voice_command(command)
            except sr.UnknownValueError:
                print("Não entendi o comando de voz.")
            except sr.RequestError:
                print("Erro ao conectar ao serviço de reconhecimento de voz.")

    def process_voice_command(self, command):
        command = command.lower()
        mapping = {
            "um": 0, "dois": 1, "três": 2,
            "quatro": 3, "cinco": 4, "seis": 5,
            "sete": 6, "oito": 7, "nove": 8
        }
        if command in mapping:
            self.player_move(mapping[command])

if __name__ == "__main__":
    root = ctk.CTk()
    game = TicTacToe(root)
    root.mainloop()
