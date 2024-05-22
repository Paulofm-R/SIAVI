import tkinter as tk
import subprocess
import sys

def open_script(script_name):
    # Fecha a janela principal
    root.destroy()
    # Abre o novo script
    subprocess.Popen([sys.executable, script_name])

    # Encerra o script atual
    sys.exit()

root = tk.Tk()
root.title("Pong Menu")
root.geometry("800x400")  # Define o tamanho da janela

# Configura a grade para alinhar os botões em uma linha
root.columnconfigure([0, 1, 2], weight=1, minsize=100)
root.rowconfigure(0, weight=1, minsize=100)

btnBack = tk.Button(root, text="Voltar", command=lambda: open_script('./menu.py'), width=5, height=2)
btn1 = tk.Button(root, text="Pong para 1 jogador", command=lambda: open_script('./Pong/pong_single.py'), width=20, height=10)
btn2 = tk.Button(root, text="Pong para 2 jogador", command=lambda: open_script('./Pong/pong_multy.py'), width=20, height=10)

btnBack.place(x=35,y=25)
btn1.grid(row=0, column=0, padx=10, pady=10)
btn2.grid(row=0, column=1, padx=10, pady=10)

root.mainloop()
