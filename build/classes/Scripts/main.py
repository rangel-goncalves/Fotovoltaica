import numpy as np
from datetime import datetime, timedelta
import openpyxl
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import argparse
import plotly.graph_objs as go
import plotly.io as pio

def constant_two(VMPP, IMPP, VOC, ISC):
    return ((VMPP / VOC - 1) / np.log(1 - IMPP / ISC))

def constant_one(VMPP, IMPP, VOC, ISC):
    return (1 - IMPP / ISC) * np.exp((-1 * VMPP) / (constant_two(VMPP, IMPP, VOC, ISC) * VOC))

# Calcula a corrente do módulo fotovoltaico (FV)
def IP(VP, VMPP, IMPP, VOC, ISC):
    return ISC * (1 - constant_one(VMPP, IMPP, VOC, ISC) * (np.exp(VP / (constant_two(VMPP, IMPP, VOC, ISC) * VOC) )- 1))

#Calcula corrente de curto-circuito
def ISC(G,T,ISCS,alpha,GS = 1000, TS = 298):
    return ISCS*(G/GS)*(1+alpha*(T-TS))

#Calcula tensão de circuito aberto
def VOC(T,beta,VOCS,TS = 298):
    return VOCS + (T-273)*beta*(T-TS)

#Calcula Corrente no ponto de máxima potência(MPP)
def IMPP(G,T,IMPPS,alpha,GS = 1000, TS = 298):
    return IMPPS*(G/GS)*(1+alpha*(T-TS))

#Calcula Tensão no ponto de máxima potência(MPP)
def VMPP(T,beta,VMPPS,TS = 298):
    return VMPPS + (T-273)*beta*(T-TS)

#Calcula tensão de circuito aberto transladada para uma dada irradiância G
def VOCm(G,VMPP,IMPP,VOC,ISC,VOCS,GS=1000):
    return constant_two(VMPP,IMPP,VOC,ISC)*VOCS*np.log(1+((1-(G/GS))/constant_one(VMPP,IMPP,VOC,ISC)))

#Calcula o coeficiente de correção em função da irradiância
def delta_V(G,VMPP,IMPP,VOC,ISC,VOCS):
    return VOCS - VOCm(G,VMPP,IMPP,VOC,ISC,VOCS)

#Fórmulas de tensão mais precisas (considerando o fator de correção)
def adjusted_VOC(T,beta,VOCS,G,VMPP,IMPP,VOC,ISC):
    return VOC(T,beta,VOCS) - delta_V(G,VMPP,IMPP,VOC,ISC,VOCS)

def adjusted_VMPP(T,beta,VMPPS,G,VMPP,IMPP,VOC,ISC,VOCS):
    return VMPP(T,beta,VMPPS) - delta_V(G,VMPP,IMPP,VOC,ISC,VOCS)

#Cálculo da tensão no módulo FV (formula de IP invertida)
def VP(IP,VMPP,IMPP,VOC,VOCS,ISC):
    return constant_two(VMPP,IMPP,VOC,ISC)*VOCS*np.log(1+((1-(IP/ISC))/constant_one(VMPP,IMPP,VOC,ISC)))

#Cálculo da resistência em série em função dos parâmetros do painel
def RS(VMPP,IMPP,VOC,ISC):
    return (constant_two(VMPP,IMPP,VOC,ISC)*(VOC/ISC))*(1/(1+constant_one(VMPP,IMPP,VOC,ISC)))

# Função principal que recebe G e T e calcula os parâmetros
def calcular_painel(G, T):
    T = T+273
    # Coeficientes de temperatura
    alpha = 0.0005  # Coeficiente de Corrente (0.05% / °C)
    beta = -0.0026  # Coeficiente de Tensão (-0.26% / °C)

    # Parâmetros do painel CS7L-605MS
    ISCS_R = 18.52
    VOCS_R = 41.5  # Tensão de Circuito Aberto (V)
    IMPPS_R = 17.25  # Corrente no Ponto de Máxima Potência (A)
    VMPPS_R = 35.1  # Tensão no Ponto de Máxima Potência (V)

    # Calculando os parâmetros com base em G e T
    ISCS = ISC(G, T, ISCS_R, alpha, GS=1000, TS=298)
    VOCS = VOC(T, beta, VOCS_R, TS=298)
    IMPPS = IMPP(G, T, IMPPS_R, alpha, GS=1000, TS=298)
    VMPPS = VMPP(T, beta, VMPPS_R, TS=298)

    # Faixa de valores de tensão (V)
    V_range = np.linspace(0, VOCS, 1000)

    # Calcula as correntes correspondentes para cada valor de tensão
    I_values = [IP(V, VMPPS, IMPPS, VOCS, ISCS) for V in V_range]

    return  I_values, V_range

def calcular_irradiancia(data_hora_str, irradiancia_global, beta, gamma_p, lat, long_local, long_meridiano, horario_verao=0):
    # Convertendo a string da data e hora para objeto datetime com o novo formato
    data_hora = datetime.strptime(data_hora_str, "%Y-%m-%d %H:%M:%S")

    # Calculando o dia do ano
    dia = data_hora.timetuple().tm_yday

    # Determinando a equação do tempo (em horas)
    B = (360/365) * (dia - 81)
    EoT = 9.87 * np.sin(np.radians(2*B)) - 7.53 * np.cos(np.radians(B)) - 1.5 * np.sin(np.radians(B))
    EoT /= 60  # Convertendo para horas

    # Hora local
    hora_local = data_hora.hour + data_hora.minute / 60

    # Calculando a hora solar
    hora_solar = hora_local - ((long_local - long_meridiano) / 15) + EoT + horario_verao

    # Calculando a declinação solar
    declinacao_solar = 23.45 * np.sin(np.radians(360 * (284 + dia) / 365))

    # Calculando o ângulo horário
    omega = 15 * (hora_solar - 12)

    # Calculando o ângulo zenital do sol
    theta_z = np.degrees(np.arccos(np.sin(np.radians(lat)) * np.sin(np.radians(declinacao_solar)) +
                                  np.cos(np.radians(lat)) * np.cos(np.radians(declinacao_solar)) * np.cos(np.radians(omega))))

    # Calculando o azimute solar
    gamma_solar = np.degrees(np.arctan2(np.sin(np.radians(omega)),
                                        np.cos(np.radians(omega)) * np.sin(np.radians(lat)) -
                                        np.tan(np.radians(declinacao_solar)) * np.cos(np.radians(lat))))

    # Cálculo do ângulo de incidência
    theta_i = np.degrees(np.arccos(np.sin(np.radians(theta_z)) * np.cos(np.radians(gamma_p - gamma_solar)) * np.sin(np.radians(beta)) +
                                    np.cos(np.radians(theta_z)) * np.cos(np.radians(beta))))

    # Cálculo da irradiância incidente
    return irradiancia_global * np.cos(np.radians(theta_i))

# Função para escolher o arquivo usando uma janela de diálogo
def escolher_arquivo():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal
    caminho_arquivo = filedialog.askopenfilename(title="Selecione o arquivo Excel",
                                                 filetypes=[("Excel files", "*.xlsx")])
    return caminho_arquivo

# Função para buscar os valores de radiação e temperatura para uma data específica
def buscar_dados_por_data(df, data_procurada):
    # Convertendo a string para o formato datetime
    data_procurada = datetime.strptime(data_procurada, "%Y-%m-%d %H:%M:%S")

    # Filtrando os dados que correspondem à data e hora procuradas
    filtro = df[df['Data_Hora'] == data_procurada]

    if not filtro.empty:
        # Pegando os valores de radiação e temperatura da célula
        radiacao = filtro['Radiação'].values[0]
        temp_cel = filtro['Temp_Cel'].values[0]
        consumo = filtro['Demanda_Avg'].values[0]
        return radiacao, temp_cel, consumo
    else:
        raise ValueError("Data não encontrada no arquivo")

# Função para ler o arquivo Excel e retornar as colunas desejadas
def ler_dados_excel(caminho_arquivo):
    # Lendo o arquivo Excel
    df = pd.read_excel(caminho_arquivo)

    # Verificando se as colunas necessárias estão presentes
    if all(col in df.columns for col in ['Data_Hora', 'Radiação', 'Temp_Cel']):
        # Retornando o DataFrame completo
        return df
    else:
        raise ValueError("Colunas 'Data_Hora', 'Radiação' ou 'Temp_Cel' não encontradas no arquivo")

def calcular_potencia_ao_longo_do_dia(NumPlacas):
    """
    Função que calcula a potência gerada ao longo de um dia com base no número de placas solares.
    """

    # Definir o intervalo de tempo do dia
    data_inicial = datetime.strptime("2019-01-11 00:00:00", "%Y-%m-%d %H:%M:%S")
    data_final = datetime.strptime("2019-01-11 23:59:00", "%Y-%m-%d %H:%M:%S")
    delta = timedelta(minutes=1)  # Incremento de 1 minuto

    # Lista para armazenar os valores ao longo do dia
    resultados = []
    horarios = []

    # Carregar os dados do Excel
    #df = ler_dados_excel(escolher_arquivo())

    # Loop para cada minuto do dia
    while data_inicial <= data_final:
      # Converter o timestamp atual para string no formato desejado
      data_procurada = data_inicial.strftime("%Y-%m-%d %H:%M:%S")

      # Buscar dados de radiação e temperatura com base na data
      radiacao, temp_cel,consumo = buscar_dados_por_data(df, data_procurada)

      # Calcular irradiância
      G_inc = calcular_irradiancia(data_procurada, radiacao, beta, gamma_p, lat, longi, long_meridiano, horario_verao)

      # Calcular os valores do painel solar
      I_values, V_range = calcular_painel(G_inc, temp_cel)

      # Armazenar o produto de V_range[-1] e I_values[0] multiplicado pelo número de placas
      resultados.append(NumPlacas * V_range[-1] * I_values[0])
      horarios.append(data_inicial)

      # Incrementar o tempo
      data_inicial += delta

    return resultados, horarios

def calcular_energia_sistema(NumPlacas):
    """
    Função que calcula quando o sistema está injetando energia na rede ou consumindo.
    """

    # Definir o intervalo de tempo do dia
    data_inicial = datetime.strptime("2019-01-11 00:00:00", "%Y-%m-%d %H:%M:%S")
    data_final = datetime.strptime("2019-01-11 23:59:00", "%Y-%m-%d %H:%M:%S")
    delta = timedelta(minutes=1)  # Incremento de 1 minuto

    # Lista para armazenar os valores ao longo do dia
    resultados = []
    horarios = []
    status_injecao = []  # Lista para armazenar o status: 1 se estiver injetando, -1 se estiver consumindo
    consumo = []
    # Loop para cada minuto do dia
    while data_inicial <= data_final:
        # Converter o timestamp atual para string no formato desejado
        data_procurada = data_inicial.strftime("%Y-%m-%d %H:%M:%S")

        # Buscar dados de radiação e temperatura com base na data
        radiacao, temp_cel,P_consumo = buscar_dados_por_data(df, data_procurada)

        # Calcular irradiância
        G_inc = calcular_irradiancia(data_procurada, radiacao, beta, gamma_p, lat, longi, long_meridiano, horario_verao)

        # Calcular os valores do painel solar
        I_values, V_range = calcular_painel(G_inc, temp_cel)

        # Calcular a potência produzida pelo sistema solar
        P_solar = NumPlacas * V_range[-1] * I_values[0]

        # Comparar com o consumo
        if P_solar > P_consumo:
            status_injecao.append(1)  # Está injetando energia na rede
        else:
            status_injecao.append(-1)  # Está usando energia da rede

        # Armazenar resultados e horários
        resultados.append(P_solar)
        horarios.append(data_inicial)
        consumo.append(P_consumo)
        # Incrementar o tempo
        data_inicial += delta

    return resultados, horarios, status_injecao,consumo
##
#########################
"""
beta = 0  # Inclinação do painel
gamma_p = 0  # Orientação do painel
lat = -9.55766947266527  # Latitude da usina
long_local = -35.78090672062049  # Longitude da usina
long_meridiano = -45  # Meridiano central do fuso horário
horario_verao = 0  # Ajuste de horário de verão
df = ler_dados_excel(escolher_arquivo())

data_procurada = "2019-01-11 12:00:00"
radiacao, temp_cel,consumo = buscar_dados_por_data(df, data_procurada)
G_inc = calcular_irradiancia(data_procurada, radiacao, beta, gamma_p, lat, long_local, long_meridiano, horario_verao)
I_values, V_range = calcular_painel(G_inc, temp_cel)
plt.plot(V_range, I_values, label=f'Curva {temp_cel}')

plt.title('Curva I-V do Módulo Fotovoltaico')
plt.xlabel('Tensão (V)')
plt.ylabel('Corrente (I) [A]')
plt.grid(True)
plt.legend()
plt.xlim(0, 50)
yticks = np.arange(0, I_values[0], 2)
xticks = np.arange(0, 50, 5)
plt.yticks(yticks)
plt.xticks(xticks)
plt.show()
"""
def print_defaut_painel(G_list,T_list):
    plt.figure(figsize=(8, 6))

    for G, T in zip(G_list, T_list):
        I_values, V_range = calcular_painel(G, T)
        plt.plot(V_range, I_values, label=f'G={G}(W/m^2),T={T}(C)')

    plt.title('Curva I-V do Módulo Fotovoltaico')
    plt.xlabel('Tensão (V)')
    plt.ylabel('Corrente (I) [A]')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 50)
    yticks = np.arange(0, 20, 1)
    xticks = np.arange(0, 50, 5)
    plt.yticks(yticks)
    plt.xticks(xticks)
    plt.show()

def teste():
    print('testei')

def painel_insta(data_procurada, NumPlacas):
    #df = ler_dados_excel(caminho_arquivo)  # !!!Rodar o bloco de cima pra ele ler o arquivo!!!
    radiacao, temp_cel, consumo = buscar_dados_por_data(df, data_procurada)
    G_inc = calcular_irradiancia(data_procurada, radiacao, beta, gamma_p, lat, longi, long_meridiano, 0)
    I_values, V_range = calcular_painel(G_inc, temp_cel)
    V_range = V_range * NumPlacas
    p = I_values * V_range
    Pot_max = p[np.argmax(p)]
    Vd_max_pot = V_range[np.argmax(p)]
    # Plotar os resultados
    fig, ax1 = plt.subplots(figsize=(6, 3))

    color = 'tab:blue'
    ax1.set_xlabel('Tensão de Saída (V)')
    ax1.set_ylabel('Corrente de Saída (A)', color=color)
    ax1.plot(V_range, I_values, color=color, label='Corrente de Saída (A)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Potência (W)', color=color)
    ax2.plot(V_range, p, color=color, linestyle='--', label='Potência (W)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.scatter(Vd_max_pot, Pot_max, color='green', marker='x', s=50, label=f"Potência Máxima: {Pot_max:.2f} W")
    plt.legend(fontsize=8)
    fig.tight_layout()
    plt.title(f'Temperatura: {temp_cel}, Radiação Incidente: {G_inc:.2f} Radiação global: {radiacao}')
    plt.grid(True)
    plt.show()
    """
    plt.plot(V_range, I_values, label=f'Curva {temp_cel}')

    plt.title('Curva I-V do Módulo Fotovoltaico')
    plt.xlabel('Tensão (V)')
    plt.ylabel('Corrente (I) [A]')
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 50)
    yticks = np.arange(0, I_values[0], 2)
    xticks = np.arange(0, 50, 5)
    plt.yticks(yticks)
    plt.xticks(xticks)
    plt.show()
    """
def pot_dia_inversor_painel(NumPlacas):
    #p, min = calcular_potencia_ao_longo_do_dia(NumPlacas)
    p, min, status_injecao, P_consumo = calcular_energia_sistema(NumPlacas)
    p = np.nan_to_num(p, nan=0.0)
    plimitada = np.clip(p, None, plim)  # Limite a potência máxima
    # Plotar os resultados ao longo do dia
    # Criar uma figura e um conjunto de subplots
    max_p = np.max(p)

    fig, axs = plt.subplots(2, 1, figsize=(10, 15))  # 3 linhas, 1 coluna

    # Plotar no primeiro subplot
    axs[0].plot(min, p, color='b')
    axs[0].set_ylabel('P(W)')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_ylim(0, max_p)
    # Plotar no segundo subplot
    axs[1].plot(min, plimitada, color='b')
    axs[1].plot(min, P_consumo, color='r', label='Consumo de Energia (W)')
    axs[1].set_ylabel('P(W)')

    axs[1].grid(True)
    axs[1].legend()

    axs[1].set_ylim(0, max_p)

    # Ajustar layout
    plt.tight_layout()
    plt.show()

def pot_dia(NumPlacas):
    # Dados simulados de potência e tempo
    p, min = calcular_potencia_ao_longo_do_dia(NumPlacas)
    p = np.nan_to_num(p, nan=0.0)
    # Criar o gráfico interativo
    fig = go.Figure()

    # Adicionar a linha de potência
    fig.add_trace(go.Scatter(
        x=min,  # Eixo X: Minutos/Horário
        y=p,  # Eixo Y: Potência
        mode='lines',
        name='Potência (W)',
        hoverinfo='x+y'  # Exibir o valor de x (tempo) e y (potência) ao passar o mouse
    ))

    # Definir títulos e eixos
    fig.update_layout(
        title='Potência ao longo do dia em 2019-01-11',
        xaxis_title='Horário',
        yaxis_title='Potência (W)',
        hovermode='x unified',  # Permitir que o valor de y apareça na linha vertical
        xaxis=dict(tickangle=-45)  # Rotacionar as legendas do eixo X
    )
    # Exibir o gráfico
    pio.show(fig)

def status_painel_rede(data_procurada, NumPlacas):
    radiacao, temp_cel, consumo = buscar_dados_por_data(df, data_procurada)
    G_inc = calcular_irradiancia(data_procurada, radiacao, beta, gamma_p, lat, longi, long_meridiano,
                                 horario_verao)
    I_values, V_range = calcular_painel(G_inc, temp_cel)
    I_values = np.nan_to_num(I_values, nan=0.0)
    V_range = V_range * NumPlacas
    p= I_values * V_range
    pp = np.max(p)
    if pp > plim:
        pp = plim
    Vp_inversor = 220 * np.sqrt(2)  # Tensão de pico
    Ip_inversor = 2 * pp / Vp_inversor  # Corrente de pico (ajuste conforme necessário)
    theta = np.pi  # Ângulo de fase (ajuste conforme necessário)
    w = 2 * np.pi * 60  # Frequência angular (60 Hz)
    t = np.linspace(0, 0.02, 1440)  # Vetor de tempo

    # Cálculo da tensão, corrente e potência instantânea
    v = Vp_inversor * np.cos(w * t)
    i = Ip_inversor * np.cos(w * t - theta)
    cos = np.cos(theta)

    if cos < 0.000000001 and cos > -0.000000001:
        cos = 0

    pa = ((Vp_inversor * Ip_inversor) / 2 * (np.cos(2 * w * t))) * cos + (Vp_inversor * Ip_inversor) / 2 * cos
    pra = ((Vp_inversor * Ip_inversor) / 2) * np.sin(2 * w * t) * np.sin(theta)
    # p = i*v
    pot_inversor = pa - pra

    # Criando os subplots lado a lado
    fig, (ax1, ax3) = plt.subplots(ncols=2, figsize=(14, 6))

    # Primeiro subplot (Potência)
    ax1.plot(t, (-1) * pra, 'b-', label='pra')
    ax1.plot(t, pot_inversor, 'g-', label='Potência (W)')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Tensão (V) / Potência (W)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    ax2 = ax1.twinx()  # Eixo secundário para a corrente
    ax2.plot(t, pa, 'r', label='pa')
    ax2.set_ylabel('Pa (W)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_title('Gráfico de Potência')
    ax1.grid(True)

    # Segundo subplot (Tensão e Corrente)
    ax3.plot(t, v, 'b-', label='Tensão (V)')
    ax3.set_xlabel('Tempo (s)')
    ax3.set_ylabel('Tensão (V)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')

    ax4 = ax3.twinx()  # Eixo secundário para a corrente
    ax4.plot(t, i, 'r', label='Corrente (A)')
    ax4.set_ylabel('Corrente (A)', color='r')
    ax4.tick_params(axis='y', labelcolor='r')

    ax3.set_title('Gráfico de Tensão e Corrente')
    ax3.grid(True)

    # Ajustando layout
    fig.tight_layout()
    plt.show()

def pot_indutiva(data_procurada, NumPlacas):
    radiacao, temp_cel, consumo = buscar_dados_por_data(df, data_procurada)
    G_inc = calcular_irradiancia(data_procurada, radiacao, beta, gamma_p, lat, longi, long_meridiano,
                                 horario_verao)
    I_values, V_range = calcular_painel(G_inc, temp_cel)
    I_values = np.nan_to_num(I_values, nan=0.0)
    V_range = V_range * NumPlacas
    p = I_values * V_range
    pp = np.max(p)
    if pp > plim:
        pp = plim
    Vp_inversor = 220 * np.sqrt(2)  # Tensão de pico
    Ip_inversor = 2 * pp / Vp_inversor  # Corrente de pico (ajuste conforme necessário)
    theta = np.pi  # Ângulo de fase (ajuste conforme necessário)
    w = 2 * np.pi * 60  # Frequência angular (60 Hz)
    t = np.linspace(0, 0.02, 1440)  # Vetor de tempo
    # Cálculo da tensão, corrente e potência instantânea
    v = Vp_inversor * np.cos(w * t)
    i = Ip_inversor * np.cos(w * t - theta)

    L = 50 * 10 ** -3  # Indutância em Henrys
    # Cálculo da tensão indutiva
    # VL = -np.max(i) * (w * L) * np.sin(w * t)
    VL = -Ip_inversor * (w * L) * np.sin(w * t)

    # Tensão no fotovoltaico
    Vfv = VL + v

    # Plotando os gráficos
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t, Vfv, 'b-', label='Vfv (V)')
    ax1.set_xlabel('Tempo (s)')
    ax1.set_ylabel('Tensão (V)')
    ax1.legend(loc='upper left')  # Adicionando legenda ao eixo ax1

    ax2 = ax1.twinx()  # Eixo secundário para a corrente
    ax2.plot(t, VL, 'g-', label='VL (V)')
    ax2.set_ylabel('Tensão (V)', color='r')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.plot(t, v, 'r-', label='v (V)')
    ax2.legend(loc='upper right')  # Adicionando legenda ao eixo ax2

    plt.title('Tensão no Fotovoltaico, Indutiva e da Rede')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plim = 2000
    parser = argparse.ArgumentParser(description="Escolha a função para executar.")
    parser.add_argument("func", choices=["func1", "func2", "func3", "func4", "func5", "func6", "func7"], help="Função para executar")
    # Adicionando argumentos para func3
    parser.add_argument("--caminho_arquivo", type=str, help="Planilha")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--long", type=float, help="Longitude")
    parser.add_argument("--datetime", type=str, help="Data e hora no formato 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--beta", type=float, help="Valor de beta")
    parser.add_argument("--gamma", type=float, help="Valor de gamma")
    parser.add_argument("--numplacas", type=int, help="Numero de placas")
    args = parser.parse_args()

    beta = args.beta  # Inclinação do painel
    gamma_p = args.gamma  # Orientação do painel
    lat = args.lat  # Latitude da usina
    longi = args.long  # Longitude da usina
    long_meridiano = -45  # Meridiano central do fuso horário
    horario_verao = 0  # Ajuste de horário de verão
    df = ler_dados_excel(args.caminho_arquivo)

    if args.func == "func1":
        G_list = [1000, 1000, 1000, 1000]
        T_list = [5, 25, 45, 65]  # Correspondentes a 5°C, 25°C, 45°C, 65°C
        print_defaut_painel(G_list, T_list)
    elif args.func == "func2":
        G_list = [1000, 800, 600, 200]
        T_list = [25, 25, 25, 25]  # Correspondentes a 5°C, 25°C, 45°C, 65°C
        print_defaut_painel(G_list, T_list)
    elif args.func == "func3":
        # Verifica se todos os argumentos necessários foram fornecidos
        if args.lat is None or args.long is None or args.datetime is None or args.beta is None or args.gamma is None:
            print("Por favor, forneça todos os argumentos necessários para func3.")
        else:
            painel_insta(args.datetime, args.numplacas)
    elif args.func == "func4":
        #pot interatida do dia
        pot_dia(args.numplacas)
    elif args.func == "func5":
        #pot e tensao+corrente // usar junto com a func3
        status_painel_rede(args.datetime,args.numplacas)
    elif args.func == "func6":
        #corte do inversor
        pot_dia_inversor_painel(args.numplacas)
    elif args.func == "func7":
        #indutiva
        pot_indutiva(args.datetime,args.numplacas)