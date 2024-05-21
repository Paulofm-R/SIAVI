import tkinter as tk
import subprocess

def open_script(script_name):
    # Fecha a janela principal
    root.destroy()
    # Abre o novo script
    subprocess.run(['python', script_name])

root = tk.Tk()
root.title("Jogo do Galo Menu")
root.geometry("800x400")  # Define o tamanho da janela

# Configura a grade para alinhar os botões em uma linha
root.columnconfigure([0, 1, 2], weight=1, minsize=100)
root.rowconfigure(0, weight=1, minsize=100)

btn1 = tk.Button(root, text="Jogador vs Computador", command=lambda: open_script('./jogoGalo_single.py'), width=20, height=10)
btn2 = tk.Button(root, text="Jogador vs Jogador", command=lambda: open_script('./jogoGalo_multy.py'), width=20, height=10)

btn1.grid(row=0, column=0, padx=10, pady=10)
btn2.grid(row=0, column=1, padx=10, pady=10)

root.mainloop()
