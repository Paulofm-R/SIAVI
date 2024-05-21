from random import randint
from copy import deepcopy

cores = {'limpar': '\033[m', 'vermelho': '\033[1;31m'}

vitorias_j = vitorias_c = cont = 0  # contador de jogadas e de vitorias
jogo = [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]  # tabela do jogo, '_' para espaços livres


def tabela():
    """
    mostrar a tabela do jogo
    :return: sem returno
    """
    for l in range(len(jogo)):
        for c in range(len(jogo[l])):
            print(jogo[l][c], end=' ')
        print()
    print()


def ocupado(l, c, jogos):
    """
    verificar se o lugar já foi jogada anteriormente
    :param l: posição na linha
    :param c:  posição na coluna
    :param jogos: tabela
    :return: se o lugar esta ocupado (o=False) ou esta livre (o=True)
    """
    if jogo[l][c] not in '_':
        o = False
    else:
        del jogos[l][c]
        o = True
    return o


def ganha_j(jogos):
    """
    ver se o jogador ganhou a partida
    :param jogos: tabela
    :return: se o jogador ganhou a partida (j=True)
    """
    j = False
    for l in range(len(jogo)):  # ver por linha
        if jogos[l][0] == jogos[l][1] == jogos[l][2] == 'O':
            j = True
            break
    for c in range(len(jogo)):  # ver por coluna
        if jogos[0][c] == jogos[1][c] == jogos[2][c] == 'O':
            j = True
            break
    if jogos[0][0] == jogos[1][1] == jogos[2][2] == 'O' or jogos[0][2] == jogos[1][1] == jogos[2][0] == 'O':  # ver por diagonal
        j = True
    return j


def ganha_c(jogos):
    """
    ver se o computador ganhou a partida
    :param jogos: tabela
    :return: se o computador ganhou a partida (com=True)
    """
    com = False
    for l in range(0, 3):  # ver por linha
        if jogos[l][0] == jogos[l][1] == jogos[l][2] == 'X':
            com = True
            break
    for c in range(0, 3):  # ver por coluna
        if jogos[0][c] == jogos[1][c] == jogos[2][c] == 'X':
            com = True
            break
    if jogos[0][0] == jogos[1][1] == jogos[2][2] == 'X' or jogos[0][2] == jogos[1][1] == jogos[2][0] == 'X':  # ver por diagonal
        com = True
    return com


def computador_vitoria():
    """
    computador verificar se ele tem hipotese de ganhar
    :return: se o computador tem hipotese ganhar (com=True)
    """
    com = False
    for l in range(0, 3):
        for c in range(0, 3):
            jogo_backup = deepcopy(jogo)  # criar um backup da tabela
            o = ocupado(l, c, jogo_backup)
            if o:
                jogo_backup[l].insert(c, 'X')  # inserir um X no backup da tabela
                com = ganha_c(jogo_backup)
                if com:
                    del jogo[l][c]
                    jogo[l].insert(c, 'X')   # inserir um X na tabela
                    break
            jogo_backup.clear()  # limpar o backup da tabela
        if com:
            break
    return com


def jogador_vitoria():
    """
    computador verificar se o jogador tem hipoteses de ganhar
    :return: se o jogador tem hipotese de ganhar (j=True)
    """
    j = False
    for l in range(0, 3):
        for c in range(0, 3):
            jogo_backup = deepcopy(jogo)  # criar um backup da tabela
            o = ocupado(l, c, jogo_backup)
            if o:
                jogo_backup[l].insert(c, 'O')  # inserir um O no backup da tabela
                j = ganha_j(jogo_backup)
                if j:
                    del jogo[l][c]
                    jogo[l].insert(c, 'X')  # inserir um X na tabela
                    break
            jogo_backup.clear()
        if j:
            break
    return j


def jogada_aleatoria():
    """
    computador jogar aleatoriamente
    :return: Sem returno
    """
    while True:
        l = randint(0, 2)
        c = randint(0, 2)
        o = ocupado(l, c, jogo)
        if o:
            jogo[l].insert(c, 'X')  # inserir um X na tabela
            break


# Programa Principal
while True:
    print(f'{"JOGO DO GALO":-^20}')
    print('JOGADOR vs COMPUTADOR')
    while True:

        while True:  # vez do jogador
            tabela()
            print('Jogador')
            while True:
                try:  # jogador escolher a posição que quer jogar
                    jogador_l = int(input('Em qual linha quer jogar (1,2,3)? ')) - 1
                    jogador_c = int(input('Em qual coluna quer jogar (1,2,3)? ')) - 1
                    if jogador_l < 0 or jogador_l > 2 or jogador_c < 0 or jogador_c > 2:  # verificar se o jogador escolheu uma posição invalida
                        raise ValueError()
                except ValueError:
                    print(f'{cores["vermelho"]}Valor inserido invalido!{cores["limpar"]}')
                else:
                    break
            o = ocupado(jogador_l, jogador_c, jogo)
            if o:  # jogar na posição escolhida
                jogo[jogador_l].insert(jogador_c, 'O')  # inserir um O na tabela
                break
            else:  # avisar caso a posição escolhida já tiver sido jogada anteriormente
                print(f'{cores["vermelho"]}Essa posição já foi jogada anteriormente{cores["limpar"]}')
        cont += 1  # somar mais uma na jogada
        j = ganha_j(jogo)
        if j:  # se o jogador venceu
            print('Jogador venceu!')
            vitorias_j += 1  # somar uma vitoria as vitorias do jogador
            break
        if cont == 9:  # se já foi jogado 9 vezes
            print('Não houve vencedor')
            break

        # vez do computador
        tabela()
        print('Computador')
        com = computador_vitoria()
        if not com:
            j = jogador_vitoria()
            if not j:
                jogada_aleatoria()
        com = ganha_c(jogo)
        cont += 1  # somar mais uma na jogada
        if com:  # se o computador venceu
            print('Computador venceu')
            vitorias_c += 1
            break

    tabela()
    # mostrar a pontuação
    print('-' * 20)
    print('PONTUAÇÃO')
    print(f'Jogador: {vitorias_j}')
    print(f'Computador: {vitorias_c}')
    print('-' * 20)

    # perguntar se quer continuar ou não
    while True:
        try:
            resp = str(input('Quer continuar [S/N]? ')).upper().strip()[0]
            if resp not in 'SN':  # se o jogador inserir outro valor sem ser S ou N dar erro
                raise ValueError()
        except ValueError:
            print(f'{cores["vermelho"]}Valor inserido invalido{cores["limpar"]}')
        else:
            break

    if resp in 'N':  # Se não quiserer continuar parar o programa
        break
    else:  # se quiser continuar voltar a ao inicio
        cont = 0  # limpar a contagem de jogadas na partida
        jogo.clear()  # limpar a tabela
        jogo = [['_', '_', '_'], ['_', '_', '_'], ['_', '_', '_']]  # criar uma tabela nova

# mostrar quem venceu
if vitorias_j > vitorias_c:
    print('Vencedor: Jogador')
elif vitorias_c > vitorias_j:
    print('Vencedor: Comtador')
else:
    print('Ficou empatado!')
print(f'{"Fim do jogo":-^21}')