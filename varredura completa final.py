import os
import numpy as np
import math
from tifffile import imwrite, imread
from pycromanager import Core
import datetime


# --- CONFIGURAÇÕES DE DIRETÓRIO ---
timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
diretorio_saida = r"D:\Users\Pedro Guedes\Varredura_"+ timestamp
os.makedirs(diretorio_saida, exist_ok = True)

# --- CÁLCULOS ---
def calibrar_plano_foco(pontos):
    """Calcula os coeficientes do plano Z = aX + bY + c usando Mínimos Quadrados."""
    A = np.c_[[p[0] for p in pontos], [p[1] for p in pontos], np.ones(len(pontos))]
    Z = np.array([p[2] for p in pontos])
    coeficientes, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    return coeficientes

def calcular_z(x, y, coeficientes):
    """Retorna o valor de Z ideal para uma coordenada (X, Y) baseada no plano."""
    return (coeficientes[0] * x) + (coeficientes[1] * y) + coeficientes[2]

core = Core()

# Identificação de Hardware
camera = core.get_camera_device()
xy_stage = core.get_xy_stage_device()
z_stage = core.get_focus_device()

fov_x, fov_y = 930.0, 930.0

print("--- CONFIGURAÇÃO DE MAPEAMENTO ---")
input("1. Mova o estágio para o PONTO INICIAL (ex: canto superior esquerdo da amostra) e pressione ENTER...")
x_inicial, y_inicial = core.get_x_position(xy_stage), core.get_y_position(xy_stage)
print(f"Ponto Inicial registrado: X={x_inicial:.2f}, Y={y_inicial:.2f}")

input("2. Mova o estágio para o PONTO FINAL (ex: canto inferior direito da amostra) e pressione ENTER...")
x_final, y_final = core.get_x_position(xy_stage), core.get_y_position(xy_stage)
print(f"Ponto Final registrado: X={x_final:.2f}, Y={y_final:.2f}")

direcao_x = 1 if x_final > x_inicial else -1
direcao_y = 1 if y_final > y_inicial else -1

dist_x = abs(x_final - x_inicial)
dist_y = abs(y_final - y_inicial)

qtd_passos_x = math.ceil(dist_x/fov_x)
malha_x = [x_inicial + (i*fov_x*direcao_x) for i in range(qtd_passos_x)]
qtd_passos_y = math.ceil(dist_y/fov_y)
malha_y = [y_inicial + (i*fov_y*direcao_y) for i in range(qtd_passos_y)]

minimo_pontos = 3
while True:
    entrada = input(f"Quantos pontos serão utilizados para ajustar a varredura? (Mínimo {minimo_pontos}): ")
    
    try:
        # Tenta converter o texto digitado em um número inteiro
        qtd_pontos_calibracao = int(entrada)
        
        # Verifica se o número atende à restrição física
        if qtd_pontos_calibracao < minimo_pontos:
            print(f"[Aviso] Você digitou {qtd_pontos_calibracao}. É necessário pelo menos {minimo_pontos} pontos para calibrar o estágio. Tente novamente.\n")
        else:
            print(f"[Sucesso] O sistema será calibrado utilizando {qtd_pontos_calibracao} pontos.\n")
            break 
            
    except ValueError:
        # Captura o erro caso o usuário digite letras, símbolos ou deixe em branco
        print("[Erro] Entrada inválida! Por favor, digite apenas números inteiros (ex: 2, 3, 4).\n")

pontos_coletados = []
for p in range(qtd_pontos_calibracao):
    input(f"\nMova o estágio para o Ponto de Calibração {p+1}, AJUSTE O FOCO, e pressione ENTER...")
    x_atual = core.get_x_position(xy_stage)
    y_atual = core.get_y_position(xy_stage)
    z_atual = core.get_position(z_stage)
    
    pontos_coletados.append((x_atual, y_atual, z_atual))
    print(f"-> Ponto {p+1} salvo: X={x_atual:.1f}, Y={y_atual:.1f}, Z={z_atual:.2f}")

coeficientes_plano = calibrar_plano_foco(pontos_coletados)
print("\n[Sucesso] Matriz do plano de foco calculada!")

#-----calculo do maior z-----
valores_z = []
for i, n in enumerate(malha_x):
    for j, m in enumerate(malha_y):
        valores_z.append(calcular_z(malha_x[i],malha_y[j],coeficientes_plano))


# Segurança: para o Live mode se estiver rodando
if core.is_sequence_running():
    core.stop_sequence_acquisition()

print(f"\nÁrea a ser mapeada: {dist_x:.2f} x {dist_y:.2f} µm")
print(f"Grade gerada: {qtd_passos_x} x {qtd_passos_y} imagens (Total: {qtd_passos_x * qtd_passos_y})")
input(f"O maior valor de Z obtido foi {max(valores_z)}. Se este valor for seguro, aperte ENTER")
print(f"Iniciando captura. Arquivos serão salvos em: {diretorio_saida}")


core.set_exposure(15.0)
for i, n in enumerate(malha_x):
    for j, m in enumerate(malha_y):
        # 1. Movimentação
        core.set_xy_position(xy_stage, n, m)
        core.wait_for_device(xy_stage)
        core.set_position(z_stage, calcular_z(malha_x[i],malha_y[j],coeficientes_plano))
        core.wait_for_device(z_stage)
        # 2. Captura dos dados brutos
        core.snap_image()
        core.wait_for_device(camera)
        pixels_brutos = core.get_image()
        
        # 3. Obtém as dimensões reais configuradas no Micro-Manager
        largura = core.get_image_width()
        altura = core.get_image_height()
        bytes_por_pixel = core.get_bytes_per_pixel() # Verifica a profundidade da câmera
        
        # 4. Organiza os pixels e extrai o canal correto para evitar a imagem preta
        try:
            # Reorganiza para (Altura, Largura, 4) assumindo o padrão BGRA
            imagem_bgra = pixels_brutos.reshape((altura, largura, 4))
            
            # Extrai os primeiros 3 canais (BGR) e inverte a ordem para RGB com [:, :, ::-1]
            imagem_colorida = imagem_bgra[:, :, :3][:, :, ::-1] 
            
        except ValueError:
            # Caso a câmara não seja colorida, mantém o plano B (escala de cinza)
            tipo_dado = np.uint8 if bytes_por_pixel == 1 else np.uint16
            imagem_colorida = pixels_brutos.view(tipo_dado).reshape((altura, largura))
            
        # 5. Define o nome do arquivo dinamicamente
        nome_arquivo = f"img_pos_X{i}_Y{j}.tif"
        caminho_completo = os.path.join(diretorio_saida, nome_arquivo)
        
        # 6. Salva a matriz no formato nativo, sem corromper os bits
        imwrite(caminho_completo, imagem_colorida)
        
        # Diagnóstico de luz no terminal para garantir que não está preto
        # (Corrigido para imagem_colorida.max() para evitar NameError)
        print(f"Salvo: {nome_arquivo} | Sinal Max: {imagem_colorida.max()} | Min: {imagem_colorida.min()}")

# 7. Retorno à origem e finalização
core.set_xy_position(xy_stage, x_inicial, y_inicial)
print("Escaneamento totalmente concluído.")

GRID_X = len(malha_x)  # Número de posições no eixo X
GRID_Y = len(malha_y) # Número de posições no eixo Y

print("Iniciando montagem do mosaico...")

linhas_do_mosaico = []

# 1. Monta o mosaico linha por linha (Eixo Y)
for j in range(GRID_Y):
    imagens_da_linha = []
    
    # 2. Pega todas as colunas daquela linha (Eixo X)
    for i in range(GRID_X):
        nome_arquivo = f"img_pos_X{i}_Y{j}.tif"
        caminho = os.path.join(diretorio_saida, nome_arquivo)
        
        try:
            # Lê a imagem do disco
            img = imread(caminho)
            imagens_da_linha.append(img)
        except FileNotFoundError:
            print(f"ERRO: Arquivo {caminho} não encontrado!")
            exit()
            
    # 3. Gruda as imagens horizontalmente (lado a lado)
    linha_montada = np.hstack(imagens_da_linha)
    linhas_do_mosaico.append(linha_montada)

# 4. Gruda todas as linhas verticalmente (uma em cima da outra)
mosaico_final = np.vstack(linhas_do_mosaico)

# 5. Salva o super arquivo final
caminho_salvamento = os.path.join(diretorio_saida, f"mosaico_final_{timestamp}.tif")
imwrite(caminho_salvamento, mosaico_final)

print(f"Sucesso! Mosaico salvo com dimensões: {mosaico_final.shape}")