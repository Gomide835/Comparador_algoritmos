import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import linear_model

# Tabela:
plt.close('all')
# Para avaliar cada algoritmo
tabela = pd.read_pickle("dados_resultados.pkl")
tabela = tabela[tabela['Algoritmo'].isin([
'SVD base',
'Power Iteration',
'Phase Correlation',
'Dense Optical Flow'
])]
estatisticas = tabela.groupby('Algoritmo').agg({
    'Tempo (ms)': ['mean', 'max', 'min'],
    'Erro X (px)': ['mean', 'max', 'min'],
    'Erro Y (px)': ['mean', 'max', 'min'],
    'Erro Abs (px)': ['mean', 'max', 'min'],
    'Erro Relativo (%)': ['mean', 'max', 'min']
}).round(3)

estatisticas = estatisticas.rename(columns={'mean': 'Média', 'max': 'Máx', 'min': 'Mín'})

print(" " * 45 + "Tabela de Resultados:\n")
print(estatisticas.to_markdown())

estatisticas.to_csv("tabela_resultados_estatisticos.csv")

# Gráfico de Barras:
resumo = tabela.groupby('Algoritmo')[['Erro Abs (px)', 'Erro Relativo (%)', 'Tempo (ms)']].mean().reset_index()

fig, eixos = plt.subplots(1, 3, figsize=(15, 5))
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 1. Gráfico de Erro Absoluto
eixos[0].bar(resumo['Algoritmo'], resumo['Erro Abs (px)'], color=cores)
eixos[0].set_title('Erro Absoluto (Média)')
eixos[0].set_ylabel('Pixels')
eixos[0].tick_params(axis='x', rotation=15)

# 2. Gráfico de Erro Relativo
eixos[1].bar(resumo['Algoritmo'], resumo['Erro Relativo (%)'], color=cores)
eixos[1].set_title('Erro Relativo (Média)')
eixos[1].set_ylabel('%')
eixos[1].tick_params(axis='x', rotation=15)

# 3. Gráfico de Tempo de Execução
eixos[2].bar(resumo['Algoritmo'], resumo['Tempo (ms)'], color=cores)
eixos[2].set_title('Tempo de Execução (Média)')
eixos[2].set_ylabel('Milissegundos')
eixos[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig("comparativo_algoritmos_v2.png", dpi=300)

fig_adv, eixos_adv = plt.subplots(1, 3, figsize=(18, 5))
cores_algoritmos = {'SVD base': '#d62728', 'Power Iteration': '#2ca02c', 'Phase Correlation': '#ff7f0e',
                    'Dense Optical Flow': '#1f77b4'}

algoritmos_todos = tabela['Algoritmo'].unique()

algoritmos_plot = algoritmos_todos

# GRÁFICO 1: Erro por Frame
for algo in algoritmos_plot:
    df_algo = tabela[tabela['Algoritmo'] == algo]
    eixos_adv[0].plot(df_algo['Frame'], df_algo['Erro Abs (px)'], label=algo, color=cores_algoritmos[algo],
                      alpha=0.8)

eixos_adv[0].set_title('1. Erro Absoluto por Frame')
eixos_adv[0].set_xlabel('Número do Frame')
eixos_adv[0].set_ylabel('Erro (Pixels)')
eixos_adv[0].grid(True, linestyle='--', alpha=0.6)
eixos_adv[0].legend()

# GRÁFICO 2: Trajetória Real vs Estimada (Cumulativa)
df_base = tabela[tabela['Algoritmo'] == algoritmos_todos[0]]
traj_x_real = np.cumsum(df_base['dx_real'])
traj_y_real = np.cumsum(df_base['dy_real'])

eixos_adv[1].plot(traj_x_real, traj_y_real, label='Gabarito REAL', color='black', linewidth=3, linestyle='--')

for algo in algoritmos_plot:
    df_algo = tabela[tabela['Algoritmo'] == algo]
    traj_x_est = np.cumsum(df_algo['dx_est'])
    traj_y_est = np.cumsum(df_algo['dy_est'])
    eixos_adv[1].plot(traj_x_est, traj_y_est, label=algo, color=cores_algoritmos[algo], alpha=0.8)

eixos_adv[1].set_title('2. Trajetória (Real vs Estimada)')
eixos_adv[1].set_xlabel('Posição X (pixels)')
eixos_adv[1].set_ylabel('Posição Y (pixels)')
eixos_adv[1].invert_yaxis()
eixos_adv[1].axis('equal')
eixos_adv[1].grid(True, linestyle='--', alpha=0.6)
eixos_adv[1].legend()

# GRÁFICO 3: Variação de Gap Pixel (Estabilidade)
for algo in algoritmos_plot:
    df_algo = tabela[tabela['Algoritmo'] == algo]
    gap_x = df_algo['dx_est'] - df_algo['dx_real']
    eixos_adv[2].plot(df_algo['Frame'], gap_x, label=algo, color=cores_algoritmos[algo], alpha=0.7)

eixos_adv[2].set_title('3. Variação de Gap Pixel (Eixo X)')
eixos_adv[2].set_xlabel('Número do Frame')
eixos_adv[2].set_ylabel('Gap Estimativa - Real (pixels)')
eixos_adv[2].axhline(0, color='black', linestyle='-', linewidth=1)
eixos_adv[2].grid(True, linestyle='--', alpha=0.6)
eixos_adv[2].legend()

plt.tight_layout()
plt.savefig("graficos_avancados_rastreamento.png", dpi=300)
plt.show()

# Código pra coleta dos plots SVD
frames_investigacao = [350, 550]

for frame_id in frames_investigacao:
    nome_arq = f"debug_svd_frame_{frame_id}.npz"

    if os.path.exists(nome_arq):
        print(f"Analisando Frame {frame_id}...")
        dados = np.load(nome_arq)

        # Descompactando variáveis
        q, Q = dados['q'], dados['Q']
        u, v = dados['u'], dados['v']
        u_angle, v_angle = dados['u_angle'], dados['v_angle']
        omega_u, omega_v = dados['omega_u'], dados['omega_v']
        u_restored, v_restored = dados['u_restored'], dados['v_restored']
        a, b = dados['a'], dados['b']

        # Variáveis de apoio
        u_unwrap = np.unwrap(u_angle)
        v_unwrap = np.unwrap(v_angle)
        u_diff = np.diff(u_unwrap)
        v_diff = np.diff(v_unwrap)

        # Plot 1: matriz q
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(q[:200, :200])
        plt.xlabel(r'y [sample]'), plt.ylabel(r'x [sample]')
        title = plt.title(r'$\mathbf{q}$')
        plt.subplot(1, 2, 2)
        plt.imshow(np.angle(Q), extent=[0, 1, 0, 1])
        plt.xlabel(r'$\Omega_y$ [rad/sample]'), plt.ylabel(r'$\Omega_x$ [rad/sample]')
        plt.title(r'$\angle(\mathbf{Q})$')
        plt.tight_layout()

        # Plot 2: Vetores Singulares u e v
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 2, 1), plt.plot(omega_u, np.real(u)), plt.plot(omega_u, np.imag(u))
        plt.legend(['real', 'imag'])
        plt.title(r'$\mathbf{u}$')
        plt.subplot(2, 2, 3), plt.plot(omega_u, np.abs(u)), plt.plot(omega_u, u_angle)
        plt.legend(['magnitude', 'phase'])
        plt.xlabel(r'$\Omega_x$ [rad/sample]')

        plt.subplot(2, 2, 2), plt.plot(omega_v, np.real(v)), plt.plot(omega_v, np.imag(v))
        plt.legend(['real', 'imag'])
        plt.title(r'$\mathbf{v}$')
        plt.subplot(2, 2, 4), plt.plot(omega_v, np.abs(v)), plt.plot(omega_v, v_angle)
        plt.legend(['magnitude', 'phase'])
        plt.xlabel(r'$\Omega_y$ [rad/sample]')
        plt.tight_layout()

        # Plot 3: Unwrapping e Restauração
        min, max = np.min([np.min(u_restored), np.min(v_restored)]) * 1.1, np.max(
            [np.max(u_restored), np.max(v_restored)]) * 1.1
        plt.figure(figsize=(12, 7))

        plt.subplot(3, 2, 1), plt.stem(omega_u[1:], u_diff, markerfmt='.')
        plt.legend(['diff'])
        plt.title(r'$\angle\mathbf{u}$')
        plt.subplot(3, 2, 3), plt.stem(omega_u, u_unwrap, markerfmt='.')
        plt.legend(['unwrapped'])
        plt.subplot(3, 2, 5), plt.plot(omega_u[1:], u_restored)
        plt.legend(['restored'])
        plt.axis([plt.axis()[0], plt.axis()[1], min, max])
        plt.xlabel(r'$\Omega_x$ [rad/sample]')

        plt.subplot(3, 2, 2), plt.stem(omega_v[1:], v_diff, markerfmt='.')
        plt.legend(['diff'])
        plt.title(r'$\angle\mathbf{v}$')
        plt.subplot(3, 2, 4), plt.stem(omega_v, v_unwrap, markerfmt='.')
        plt.legend(['unwrapped'])
        plt.subplot(3, 2, 6), plt.plot(omega_v[1:], v_restored)
        plt.axis([plt.axis()[0], plt.axis()[1], min, max])
        plt.legend(['restored'])
        plt.xlabel(r'$\Omega_y$ [rad/sample]')
        plt.tight_layout()

        # Plot 4: linear fit
        plt.figure(figsize=(12, 2))
        plt.subplot(1, 2, 1)
        plt.plot(omega_u[1:], u_restored)
        plt.plot(omega_u, a[0] * omega_u + a[1])
        plt.axis([plt.axis()[0], plt.axis()[1], min, max])
        plt.legend([r'$\angle\mathbf{u}$', 'linear fit'])
        plt.xlabel(r'$\Omega_x$ [rad/sample]')

        plt.subplot(1, 2, 2)
        plt.plot(omega_v[1:], v_restored)
        plt.plot(omega_v, b[0] * omega_v + b[1])
        plt.axis([plt.axis()[0], plt.axis()[1], min, max])
        plt.legend([r'$\angle\mathbf{v}$', 'linear fit'])
        plt.xlabel(r'$\Omega_y$ [rad/sample]')

        # RANSAC para análise de ruído na fase
        ransac = linear_model.RANSACRegressor()
        X_reg = np.array([np.ones_like(u_unwrap) * (omega_u[2] - omega_u[1])]).transpose()
        ransac.fit(X_reg, u_unwrap)

        plt.figure(figsize=(10, 2))
        plt.plot(u_unwrap, '.', label='Pontos de Fase')
        plt.plot(np.arange(len(u_unwrap))[ransac.inlier_mask_], u_unwrap[ransac.inlier_mask_], 'g.', label='Inliers')
        plt.title(f'RANSAC Inliers - Frame {frame_id}')
        plt.legend()

        plt.show()  # Pausa para análise de cada frame