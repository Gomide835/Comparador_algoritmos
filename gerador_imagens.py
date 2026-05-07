import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from funcoes import fft_shift_2d #add_gaussian_noise #Exemplo de como você vai precisar importar depois

print("Gerando Trajetória...")
img_bgr = cv2.imread(r"C:\Users\victo\Desktop\Faculdade\I.C\comparador_algoritmo\PIPE_IMG.jpg")
if img_bgr is None:
    print("Erro: 'PIPE_IMG.jpg' não encontrada.")
    exit()

base_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

np.random.seed(42)
img_size = (480, 640)
downscale_factor = 1

trajectory_description = [
    {"shift_x": 0, "shift_y": 0, "N": 100, "shift_noise_x": 0, "shift_noise_y": 0},
    {"shift_x": 1000, "shift_y": 0, "N": 100, "shift_noise_x": 0, "shift_noise_y": 0},
    {"shift_x": 0, "shift_y": 0, "N": 100, "shift_noise_x": 0, "shift_noise_y": 0},
    {"shift_x": 0, "shift_y": 500, "N": 100, "shift_noise_x": 0, "shift_noise_y": 0},
    {"shift_x": 0, "shift_y": 0, "N": 100, "shift_noise_x": 0, "shift_noise_y": 0},
    {"shift_x": -1000, "shift_y": -500, "N": 100, "shift_noise_x": 0, "shift_noise_y": 0},
    {"shift_x": 0, "shift_y": 0, "N": 100, "shift_noise_x": 0, "shift_noise_y": 0}
]

x0, z0 = img_size[1] // 2 + 20, img_size[0] // 2 + 20

data = {
    "inspection": [],
    "filename": [],
    "order": [],
    "img": [],
    "delta_x": [],
    "delta_y": [],
}

x, z = [x0], [z0]
total_iter = np.sum([desc["N"] for desc in trajectory_description]) - len(trajectory_description)

curr_iter = 0
for description in trajectory_description:
    shift_x, shift_y = description["shift_x"], description["shift_y"]
    N = description["N"]
    delta_x, delta_y = shift_x / N, shift_y / N

    for i in range(1, N):
        rand_delta_x = np.random.normal(delta_x, description["shift_noise_x"])
        rand_delta_y = np.random.normal(delta_y, description["shift_noise_y"])

        x.append(rand_delta_x + x[-1])
        z.append(rand_delta_y + z[-1])

        int_x, int_z = np.round(x[-1]), np.round(z[-1])
        img_int = base_img[int(int_z - img_size[0] // 2): int(int_z + img_size[0] // 2),
        int(int_x - img_size[1] // 2): int(int_x + img_size[1] // 2)]
        float_x, float_z = x[-1] - int_x, z[-1] - int_z
        shifted_img = fft_shift_2d(img_int, dx=float_x, dy=float_z)

        #shifted_img = apply_lens_blur(shifted_img)
        #shifted_img = apply_full_blur(shifted_img)
        #shifted_img = add_gaussian_noise(shifted_img, sigma=15)
        #shifted_img = add_salt_and_pepper(shifted_img, amount=0.01)

        shifted_img = np.clip(shifted_img, 0, 255).astype(np.uint8)
        downsampled_img = cv2.resize(shifted_img, None, fx=1 / downscale_factor, fy=1 / downscale_factor)

        data['order'].append(i)
        data['filename'].append(f"img_{curr_iter:05d}.png")
        data['inspection'].append("pipe_img")
        data['delta_x'].append(rand_delta_x / downscale_factor)
        data['delta_y'].append(rand_delta_y / downscale_factor)
        data['img'].append(downsampled_img)

        cv2.imwrite(f"img_{curr_iter:05d}.png", downsampled_img)
        curr_iter += 1
        print(f"Gerando imagens: {curr_iter} / {total_iter} ({curr_iter / total_iter:.2%})", end='\r')

print("\nSalvando DataFrame...")
df = pd.DataFrame(data)
df.to_pickle("artificial_dataset.pkl")
df.to_csv("artificial_dataset.csv", index=False)

x_arr, z_arr = np.array(x), np.array(z)
plt.figure()
plt.plot(x_arr, z_arr, '-')
plt.axis("equal")
plt.title("Trajetória Gerada")
plt.show()