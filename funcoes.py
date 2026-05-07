import numpy as np
import cv2
from scipy.fft import fft2, ifft2

def unwrap(u_diff):
    u_unwrap = np.copy(u_diff)
    u_unwrap[u_diff < -np.pi] = u_unwrap[u_diff < -np.pi] + 2 * np.pi
    u_unwrap[u_diff > np.pi] = u_unwrap[u_diff > np.pi] - 2 * np.pi
    return u_unwrap

def power_iteration(Q, num_iterations=10, tol=1e-8):
    v = np.ones(Q.shape[1], dtype=np.complex128)
    v /= np.linalg.norm(v)
    iteracoes_realizadas = num_iterations
    # Critério de Parada Inteligente
    for i in range(num_iterations):
        v_new = Q.conj().T @ (Q @ v)
        v_new /= np.linalg.norm(v_new)
        if np.linalg.norm(v_new - v) < tol:
            iteracoes_realizadas = i + 1
            v = v_new
            break
        v = v_new
    u = Q @ v
    u /= np.linalg.norm(u)
    return u, np.conj(v), iteracoes_realizadas

def phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)
    r = np.fft.ifft2(R).real
    return r

def fft_shift_2d(imagem, dx, dy, janela=None):
    h, w = imagem.shape
    imagem_f = imagem.astype(np.float64)

    if janela is not None:
        imagem_f *= janela

    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    FX, FY = np.meshgrid(fx, fy)
    fase = np.exp(-2j * np.pi * (FX * dx + FY * dy))
    I_fft = fft2(imagem_f)
    I_deslocada = ifft2(I_fft * fase)
    return np.real(I_deslocada)


def add_gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise


def add_salt_and_pepper(img, amount=90 / 100, s_vs_p=0.5):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = (np.random.randint(0, img.shape[0], int(num_salt)),
              np.random.randint(0, img.shape[1], int(num_salt)))
    noisy[coords] = 255

    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = (np.random.randint(0, img.shape[0], int(num_pepper)),
              np.random.randint(0, img.shape[1], int(num_pepper)))
    noisy[coords] = 0
    return noisy


def apply_lens_blur(img, blur_kernel=(21, 21), feather=100):
    rows, cols = img.shape[:2]
    mask = np.zeros((rows, cols), dtype=np.float32)
    cv2.circle(mask, (cols // 2, rows // 2), min(rows, cols) // 2, 1, -1)

    mask = cv2.GaussianBlur(mask, (101, 101), feather)
    blurred = cv2.GaussianBlur(img, blur_kernel, 0)
    blended = img * mask + blurred * (1 - mask)
    return blended


def apply_full_blur(img, blur_kernel=(7, 7)):
    return cv2.GaussianBlur(img, blur_kernel, 0)