import time
import numpy as np
import cv2
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from numpy.linalg import lstsq
from funcoes import unwrap, power_iteration, phase_correlation

print("Iniciando Avaliação...")
df_gabarito = pd.read_pickle("artificial_dataset.pkl")
dados_totais = []

for i in range(1, len(df_gabarito)):
    arquivo_atual = df_gabarito['filename'].iloc[i]
    imagem_referencia = df_gabarito['filename'].iloc[i - 1]

    dx_real = df_gabarito['delta_x'].iloc[i]
    dy_real = df_gabarito['delta_y'].iloc[i]

    img1 = cv2.imread(imagem_referencia, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(arquivo_atual, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Erro ao carregar as imagens. Pulando...")
        continue

    omega_u_base = fftfreq(img1.shape[0])
    omega_v_base = fftfreq(img1.shape[1])

    # ALGORITMO 1: SVD Base
    inicio_scipy = time.perf_counter()
    F_svd = fft2(img1)
    G_svd = fft2(img2)
    Q = F_svd * G_svd.conjugate() / (np.abs(F_svd) * (np.abs(G_svd)))

    u, s, v = svds(Q, 1)
    u, v = u[:, 0], v[0, :]
    u_angle, v_angle = np.angle(u), np.angle(v)
    omega_u = fftfreq(len(u))
    omega_v = fftfreq(len(v))

    shift = True
    if shift is True:
        u, v = fftshift(u), fftshift(v)
        omega_u, omega_v = fftshift(omega_u), fftshift(omega_v)

    u_diff = np.diff(u_angle)
    v_diff = np.diff(v_angle)
    u_unwrap = unwrap(u_diff)
    v_unwrap = unwrap(v_diff)
    u_restored = np.cumsum(u_unwrap)
    v_restored = np.cumsum(v_unwrap)
    u_restored = u_restored - np.mean(u_restored)
    v_restored = v_restored - np.mean(v_restored)

    a = lstsq(np.array([omega_u[1:200], np.ones_like(omega_u[1:200])]).transpose(),
              u_restored[1:200], rcond=None)[0]
    b = lstsq(np.array([omega_v[1:200], np.ones_like(omega_v[1:200])]).transpose(),
              v_restored[1:200], rcond=None)[0]

    dx_svd = -b[0] / (2 * np.pi)
    dy_svd = -a[0] / (2 * np.pi)
    tempo_svd_ms = (time.perf_counter() - inicio_scipy) * 1000

    #Coleta dos Frames pra investigação
    frames_investigacao = [350,550]

    if i in frames_investigacao:
        q = np.real(ifft2(Q))
        nome_arquivo_debug = f"debug_svd_frame_{i}.npz"
        np.savez(
            nome_arquivo_debug,
            q=q, Q=Q, u=u, v=v,
            u_angle=u_angle, v_angle=v_angle,
            omega_u=omega_u, omega_v=omega_v,
            u_restored=u_restored, v_restored=v_restored,
            a=a, b=b
        )
        print(f"[DEBUG] Variáveis do SVD salvas no arquivo: {nome_arquivo_debug}")
    # =====================================================================

    # ALGORITMO 2: SVD com Power Iteration
    inicio_scipy = time.perf_counter()
    F_svd = fft2(img1)
    G_svd = fft2(img2)
    Q = F_svd * G_svd.conjugate() / (np.abs(F_svd) * (np.abs(G_svd)))

    u, v, num_iter= power_iteration(Q)
    u_angle, v_angle = np.angle(u), np.angle(v)
    omega_u = fftfreq(len(u))
    omega_v = fftfreq(len(v))

    shift = True
    if shift is True:
        u, v = fftshift(u), fftshift(v)
        omega_u, omega_v = fftshift(omega_u), fftshift(omega_v)

    u_diff = np.diff(u_angle)
    v_diff = np.diff(v_angle)
    u_unwrap = unwrap(u_diff)
    v_unwrap = unwrap(v_diff)
    u_restored = np.cumsum(u_unwrap)
    v_restored = np.cumsum(v_unwrap)
    u_restored = u_restored - np.mean(u_restored)
    v_restored = v_restored - np.mean(v_restored)

    a = lstsq(np.array([omega_u[1:200], np.ones_like(omega_u[1:200])]).transpose(),
              u_restored[1:200], rcond=None)[0]
    b = lstsq(np.array([omega_v[1:200], np.ones_like(omega_v[1:200])]).transpose(),
              v_restored[1:200], rcond=None)[0]

    dx_pi = -b[0] / (2 * np.pi)
    dy_pi = -a[0] / (2 * np.pi)
    tempo_pi_ms = (time.perf_counter() - inicio_scipy) * 1000

    # ALGORITMO 3: Phase Correlation
    inicio_pc = time.perf_counter()
    result_pc = phase_correlation(img1, img2)
    pos_pc = np.unravel_index(np.argmax(result_pc), result_pc.shape)
    dy_pc, dx_pc = pos_pc
    h_pc, w_pc = result_pc.shape
    if dx_pc > w_pc // 2: dx_pc = dx_pc - w_pc
    if dy_pc > h_pc // 2: dy_pc = dy_pc - h_pc
    tempo_pc_ms = (time.perf_counter() - inicio_pc) * 1000

    # ALGORITMO 4: Dense Optical Flow (Farneback)
    inicio_of = time.perf_counter()
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx_of = -np.median(flow[..., 0])
    dy_of = -np.median(flow[..., 1])
    tempo_of_ms = (time.perf_counter() - inicio_of) * 1000

    # Cálculo dos Erros
    mag_real = np.sqrt(dx_real ** 2 + dy_real ** 2)
    if mag_real == 0: mag_real = 1e-06

    erro_x_svd = abs(dx_real - dx_svd)
    erro_y_svd = abs(dy_real - dy_svd)
    erro_abs_svd = np.sqrt(erro_x_svd ** 2 + erro_y_svd ** 2)
    erro_rel_svd = (erro_abs_svd / mag_real) * 100

    erro_x_pi = abs(dx_real - dx_pi)
    erro_y_pi = abs(dy_real - dy_pi)
    erro_abs_pi = np.sqrt(erro_x_pi ** 2 + erro_y_pi ** 2)
    erro_rel_pi = (erro_abs_pi / mag_real) * 100

    erro_x_pc = abs(dx_real - dx_pc)
    erro_y_pc = abs(dy_real - dy_pc)
    erro_abs_pc = np.sqrt(erro_x_pc ** 2 + erro_y_pc ** 2)
    erro_rel_pc = (erro_abs_pc / mag_real) * 100

    erro_x_of = abs(dx_real - dx_of)
    erro_y_of = abs(dy_real - dy_of)
    erro_abs_of = np.sqrt(erro_x_of ** 2 + erro_y_of ** 2)
    erro_rel_of = (erro_abs_of / mag_real) * 100

    dados_totais.extend([
        {"Frame": i, "Algoritmo": "SVD base", "dx_real": dx_real, "dy_real": dy_real, "dx_est": dx_svd,
         "dy_est": dy_svd,
         "Erro X (px)": erro_x_svd, "Erro Y (px)": erro_y_svd, "Erro Abs (px)": erro_abs_svd,
         "Erro Relativo (%)": erro_rel_svd, "Tempo (ms)": tempo_svd_ms},

        {"Frame": i, "Algoritmo": "Power Iteration", "dx_real": dx_real, "dy_real": dy_real, "dx_est": dx_pi,
         "dy_est": dy_pi,
         "Erro X (px)": erro_x_pi, "Erro Y (px)": erro_y_pi, "Erro Abs (px)": erro_abs_pi,
         "Erro Relativo (%)": erro_rel_pi, "Tempo (ms)": tempo_pi_ms},

        {"Frame": i, "Algoritmo": "Phase Correlation", "dx_real": dx_real, "dy_real": dy_real, "dx_est": dx_pc,
         "dy_est": dy_pc,
         "Erro X (px)": erro_x_pc, "Erro Y (px)": erro_y_pc, "Erro Abs (px)": erro_abs_pc,
         "Erro Relativo (%)": erro_rel_pc, "Tempo (ms)": tempo_pc_ms},

        {"Frame": i, "Algoritmo": "Dense Optical Flow", "dx_real": dx_real, "dy_real": dy_real, "dx_est": dx_of,
         "dy_est": dy_of,
         "Erro X (px)": erro_x_of, "Erro Y (px)": erro_y_of, "Erro Abs (px)": erro_abs_of,
         "Erro Relativo (%)": erro_rel_of, "Tempo (ms)": tempo_of_ms}
    ])

print("Salvando os resultados dos algoritmos...")
df_resultados = pd.DataFrame(dados_totais)
df_resultados.to_pickle("dados_resultados.pkl")
df_resultados.to_csv("dados_resultados.csv")
