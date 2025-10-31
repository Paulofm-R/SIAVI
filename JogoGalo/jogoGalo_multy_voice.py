import customtkinter as ctk
import tkinter as tk
import speech_recognition as sr
import threading
import random
import subprocess
import sys

# Configurações iniciais da interface
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Jogo do Galo - Jogador vs Jogador")
        self.root.geometry("425x500")
        self.board = [" "]*9
        self.current_player = random.choice(["X", "O"])
        self.player_x_score = 0
        self.player_o_score = 0
        self.buttons = []
        self.create_widgets()
        self.update_highlight()
        self.voice_thread = threading.Thread(target=self.voice_command_listener, daemon=True)
        self.voice_thread.start()

    def create_widgets(self):
        self.player_x_frame = ctk.CTkFrame(self.root, border_color="red", border_width=0)
        self.player_x_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.player_x_score_label = ctk.CTkLabel(self.player_x_frame, text=f"Jogador X: {self.player_x_score}", font=("Arial", 18))
        self.player_x_score_label.pack()

        self.player_o_frame = ctk.CTkFrame(self.root, border_color="red", border_width=0)
        self.player_o_frame.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.player_o_score_label = ctk.CTkLabel(self.player_o_frame, text=f"Jogador O: {self.player_o_score}", font=("Arial", 18))
        self.player_o_score_label.pack()

        for i in range(9):
            button = ctk.CTkButton(self.root, text=str(i+1), width=120, height=120, font=("Arial", 24), command=lambda i=i: self.player_move(i))
            button.grid(row=(i//3) + 1, column=i%3, padx=6, pady=6)
            self.buttons.append(button)

    def player_move(self, index):
        if self.board[index] == " ":
            self.board[index] = self.current_player
            self.buttons[index].configure(text=self.current_player)
            if not self.check_winner():
                self.current_player = "O" if self.current_player == "X" else "X"
                self.update_highlight()

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
            self.player_x_score += 1
            self.player_x_score_label.configure(text=f"Jogador X: {self.player_x_score}")
        elif winner == "O":
            self.player_o_score += 1
            self.player_o_score_label.configure(text=f"Jogador O: {self.player_o_score}")

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
        self.current_player = random.choice(["X", "O"])
        self.update_highlight()

    def update_highlight(self):
        if self.current_player == "X":
            self.player_x_score_label.configure(text_color="green")
            self.player_o_score_label.configure(text_color="white")
        else:
            self.player_x_score_label.configure(text_color="white")
            self.player_o_score_label.configure(text_color="green")

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
            "sete": 6, "oito": 7, "nove": 8,
            "sair": "exit"
        }
        if command in mapping:
            if command == "sair":
                self.exit_game()  # Chama a função exit_game
            else:
                self.player_move(mapping[command])
    
    def exit_game(self):
        subprocess.Popen([sys.executable, "./JogoGalo/jogoGalo_menu.py"])  # Inicia o novo script
        self.root.destroy()  # Fecha e destrói a janela do tkinter

if __name__ == "__main__":
    root = ctk.CTk()
    game = TicTacToe(root)
    root.mainloop()
