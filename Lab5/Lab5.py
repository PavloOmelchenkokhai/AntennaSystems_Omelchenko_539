import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані Варіант №13
lmbda = 3.15
N = 14
d = 2.0
a = 2.3
nu = 1

# Розрахунок параметрів хвилеводу та ДС
lmbda_kr = 2 * a
lmbda_xb = lmbda / np.sqrt(1 - (lmbda / lmbda_kr)**2)
k = 2 * np.pi / lmbda

# Розрахунок нулів ДС для обох знаків (ліва та права півплощина)
zeros_deg = []
for m in range(1, N):
    for sign in [1, -1]:
        arg = (lmbda * (sign * m)) / (d * N) + lmbda / lmbda_xb - (nu * lmbda) / (2 * d)
        if -1 <= arg <= 1:
            deg = np.degrees(np.arcsin(arg))
            if -90 <= deg <= 90:
                # унікальні значення
                if not any(np.isclose(deg, existing_deg, atol=1e-2) for _, existing_deg in zeros_deg):
                    zeros_deg.append((sign * m, deg))
zeros_deg.sort(key=lambda x: x[1])

# Розрахунок максимумів бічних пелюсток для обох знаків
maxima_deg = []
for n in range(1, N):
    for sign in [1, -1]:
        arg = sign * (((2 * n + 1) * lmbda) / (2 * d * N)) + lmbda / lmbda_xb - (nu * lmbda) / (2 * d)
        if -1 <= arg <= 1:
            deg = np.degrees(np.arcsin(arg))
            if -90 <= deg <= 90:
                if not any(np.isclose(deg, existing_deg, atol=1e-2) for _, existing_deg in maxima_deg):
                    maxima_deg.append((sign * n, deg))
maxima_deg.sort(key=lambda x: x[1])

theta_deg_rect = np.linspace(-90, 90, 3000)
theta_rad_rect = np.radians(theta_deg_rect)

cos_th = np.cos(theta_rad_rect)
cos_th[np.abs(cos_th) < 1e-9] = 1e-9
F_1H = np.abs(np.cos((np.pi / 2) * np.sin(theta_rad_rect)) / cos_th)

psi = k * d * np.sin(theta_rad_rect) - (2 * np.pi / lmbda_xb) * d + nu * np.pi

sin_psi_half = np.sin(psi / 2)
sin_N_psi_half = np.sin(N * psi / 2)

F_HC = np.zeros_like(psi)
mask = np.abs(sin_psi_half) < 1e-7
F_HC[mask] = 1.0
F_HC[~mask] = np.abs(sin_N_psi_half[~mask] / (N * sin_psi_half[~mask]))

F_H = F_1H * F_HC
F_H_norm = F_H / np.max(F_H)

idx_half = np.argmin(np.abs(F_H_norm[theta_deg_rect >= 0] - 0.707))
theta_07_val = theta_deg_rect[theta_deg_rect >= 0][idx_half]

# ГРАФІК 1: ПРЯМОКУТНА СИСТЕМА КООРДИНАТ
plt.figure(figsize=(14, 7))

plt.plot(theta_deg_rect, F_1H, 'r-', linewidth=1.2, label=r'$|F_{1H}(\Theta)|$ (Огинаюча щілини)')
plt.plot(theta_deg_rect, F_HC, 'k-', linewidth=1.2, label=r'$|F_{HC}(\Theta)|$ (Множник решітки)')
plt.plot(theta_deg_rect, F_H_norm, 'b--', linewidth=1.8, label=r'$|F_H(\Theta)|$ (Підсумкова ДС, Н-площина)')

plt.axhline(0.707, color='brown', linestyle='--', linewidth=1.0, label=f'Рівень 0.7 ($\\Theta_{{0.7}} \\approx {theta_07_val:.2f}^\\circ$)')
plt.plot([theta_07_val, theta_07_val], [0, 0.707], 'r--')
plt.scatter([theta_07_val], [0.707], color='red', s=40, zorder=5, label=r'Точка рівня $\Theta_{0.7}$')

first_zero = True
for m, deg in zeros_deg:
    val_zero = np.interp(deg, theta_deg_rect, F_H_norm)
    if first_zero:
        plt.scatter([deg], [val_zero], color='purple', s=40, zorder=5, label=r'Напрямки нулів ($\Theta_{0m}$)')
        first_zero = False
    else:
        plt.scatter([deg], [val_zero], color='purple', s=40, zorder=5)
    idx_label = abs(m)
    plt.text(deg, -0.05, f'$\\Theta_{{0{idx_label}}}$', fontsize=8, ha='center', va='top', color='purple')

first_max = True
for n, deg in maxima_deg:
    val_max = np.interp(deg, theta_deg_rect, F_H_norm)
    if first_max:
        plt.scatter([deg], [val_max], color='green', s=40, zorder=5, label=r'Максимуми $\Theta_{mn}$ (Н-площина)')
        first_max = False
    else:
        plt.scatter([deg], [val_max], color='green', s=40, zorder=5)
    idx_label = abs(n)
    plt.text(deg, val_max + 0.03, f'$\\Theta_{{m{idx_label}}}(H)$', fontsize=8, ha='center', va='bottom', color='green')

plt.title("Розрахункова ДС хвилеводно-щілинної антени", fontsize=13, fontweight='bold', pad=12)
plt.xlabel(r"Кут $\Theta^\circ$, град", fontsize=11)
plt.ylabel("Нормована амплітуда", fontsize=11)
plt.xlim([-90, 90])
plt.ylim([-0.09, 1.08])
plt.xticks(np.arange(-90, 91, 15))
plt.yticks([0.0, 0.5, 0.7, 1.0])
plt.grid(True, linestyle=':', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)

plt.legend(loc='upper right', fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.show()


# ГРАФІК 2: ПОЛЯРНА СИСТЕМА КООРДИНАТ
plt.figure(figsize=(8, 8))
ax_polar = plt.subplot(111, projection='polar')

ax_polar.set_theta_zero_location('N')
ax_polar.set_theta_direction(-1)

ax_polar.plot(theta_rad_rect, F_H_norm, 'b-', linewidth=2.0, label=r'Нормована ДС $|F_H(\Theta)|$')
ax_polar.plot(theta_rad_rect, F_1H, 'r:', linewidth=1.2, label=r'Огинаюча $|F_{1H}(\Theta)|$')

theta_arc = np.radians(np.linspace(-90, 90, 300))
ax_polar.plot(theta_arc, np.full_like(theta_arc, 0.707), 'r--', linewidth=1.2, label=r'Рівень 0.707 ($\Theta_{0.7}$)')

for m, deg in zeros_deg:
    val = np.interp(deg, theta_deg_rect, F_H_norm)
    ax_polar.scatter(np.radians(deg), val, color='purple', s=35, zorder=5)

for n, deg in maxima_deg:
    val = np.interp(deg, theta_deg_rect, F_H_norm)
    ax_polar.scatter(np.radians(deg), val, color='green', s=35, zorder=5)

ax_polar.set_title("Нормована ДС у полярній системі координат", fontsize=13, fontweight='bold', pad=20)
ax_polar.set_thetamin(-90)
ax_polar.set_thetamax(90)
ax_polar.set_ylim([0, 1.05])
ax_polar.set_yticks([0.25, 0.5, 0.707, 1.0])
ax_polar.set_yticklabels(['0.25', '0.50', '0.707', '1.00'])
ax_polar.grid(True, linestyle=':', alpha=0.7)

ax_polar.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0), fontsize=10)
plt.tight_layout()
plt.show()