import math
import matplotlib.pyplot as plt
import numpy as np

# (Варіант 13)
lambd = 3.7
ap = 16.0
bp = 16.0

theta_deg = np.linspace(-90, 90, 3601)
theta_rad = np.radians(theta_deg)

FE = []
FH = []

for t in theta_rad:
    sin_t = np.sin(t)
    cos_t = np.cos(t)

    # Множник Гюйгенса
    F_I = (1 + cos_t) / 2.0

    # Площина E
    arg_E = (np.pi * bp / lambd) * sin_t
    if abs(arg_E) < 1e-9:
        F_CE = 1.0
    else:
        F_CE = abs(np.sin(arg_E) / arg_E)
    f_e = F_CE * F_I
    FE.append(f_e)

    # Площина H
    arg_H = (np.pi * ap / lambd) * sin_t
    denom_H = 1.0 - (2.0 * ap / lambd * sin_t) ** 2
    if abs(denom_H) < 1e-6:
        F_CH = np.pi / 4.0
    else:
        F_CH = abs(np.cos(arg_H) / denom_H)
    f_h = F_CH * F_I
    FH.append(f_h)

FE = np.array(FE)
FH = np.array(FH)

# Пошук ширини головної пелюстки на рівні 0.707
idx_E = np.where(FE[: len(FE) // 2] >= 0.707)[0][0]
theta_0707_E = abs(theta_deg[idx_E]) * 2.0

idx_H = np.where(FH[: len(FH) // 2] >= 0.707)[0][0]
theta_0707_H = abs(theta_deg[idx_H]) * 2.0

print(f'Ширина пелюстки у площині E (0.707): {theta_0707_E:.2f}°')
print(f'Ширина пелюстки у площині H (0.707): {theta_0707_H:.2f}°')

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(theta_deg, FE, 'b-', label=r'$F_E(\theta)$ (Площина E)', linewidth=1.5)
plt.plot(theta_deg, FH, 'r--', label=r'$F_H(\theta)$ (Площина H)', linewidth=1.5)
plt.axhline(
    y=0.707, color='g', linestyle=':', label='Рівень половинної потужності 0.707'
)

plt.title(
    fr'Нормовані ДС пірамідального рупора (Варіант 13: $\lambda={lambd}$ см, $a_p={ap}$ см, $b_p={bp}$ см)'
)
plt.xlabel(r'Кут $\theta$, град')
plt.ylabel(r'Нормована амплітуда $F(\theta)$')
plt.xlim(-90, 90)
plt.ylim(0, 1.05)
plt.xticks(np.arange(-90, 91, 15))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig('DS_Pyramidal_Horn_Variant13.png', dpi=300)
plt.show()