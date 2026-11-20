import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.signal import find_peaks

lam = 0.0445
D = 1.2
f = 0.4

k = 2 * np.pi / lam
R0 = D / 2.0
p = 2 * f
v = 3.5 * (R0 / p)
u_max = k * R0
norm_denom = 0.74 * (jv(1, v) / v) + 0.13


def calc_term1(u, x):
    res = np.zeros_like(u, dtype=float)
    mask = np.isclose(u, x, atol=1e-7)
    u_reg = u[~mask]
    res[~mask] = (x * jv(1, x) * jv(0, u_reg) - u_reg * jv(1, u_reg) * jv(0, x)) / (x ** 2 - u_reg ** 2)
    res[mask] = 0.5 * (jv(0, x) ** 2 + jv(1, x) ** 2)
    return res


def calc_term2(u):
    res = np.zeros_like(u, dtype=float)
    mask = np.isclose(u, 0.0, atol=1e-9)
    u_reg = u[~mask]
    res[~mask] = jv(1, u_reg) / u_reg
    res[mask] = 0.5
    return res


def calc_term3(u, x15):
    res = np.zeros_like(u, dtype=float)
    mask = np.isclose(u, x15, atol=1e-7)
    u_reg = u[~mask]
    res[~mask] = (u_reg * jv(1, u_reg) * jv(2, x15) - x15 * jv(1, x15) * jv(2, u_reg)) / (x15 ** 2 - u_reg ** 2)
    d_num = (jv(1, x15) + x15 * 0.5 * (jv(0, x15) - jv(2, x15))) * jv(2, x15) - \
            x15 * jv(1, x15) * 0.5 * (jv(1, x15) - jv(3, x15))
    res[mask] = d_num / (-2 * x15)
    return res


def pattern_u(u):
    sin_theta = np.clip(u / u_max, 0.0, 1.0)
    theta = np.arcsin(sin_theta)
    t1 = 0.74 * calc_term1(u, v)
    t2 = 0.26 * calc_term2(u)
    t3 = 0.25 * calc_term3(u, 1.5 * v)
    factor = (np.cos(theta / 2.0) ** 2) / norm_denom
    F_E = factor * (t1 + t2 + t3)
    F_H = factor * (t1 + t2 - t3)
    return F_E, F_H


u = np.linspace(0, u_max, 100000)
F_E, F_H = pattern_u(u)


def get_key_points(u_arr, F_arr):
    level = 1 / np.sqrt(2)
    idx_hp = np.where(F_arr <= level)[0][0]
    u_hp = u_arr[idx_hp]
    theta_hp = np.rad2deg(np.arcsin(u_hp / u_max))

    zero_crossings = np.where(np.diff(np.sign(F_arr)))[0]
    u_zeros = []
    for idx in zero_crossings:
        u0 = u_arr[idx] - F_arr[idx] * (u_arr[idx + 1] - u_arr[idx]) / (F_arr[idx + 1] - F_arr[idx])
        u_zeros.append(u0)
    u_zeros = np.array(u_zeros)

    peaks_idx, _ = find_peaks(np.abs(F_arr), prominence=1e-4)
    u_peaks = u_arr[peaks_idx]
    F_peaks = F_arr[peaks_idx]

    return (u_hp, theta_hp, level), u_zeros, (u_peaks, F_peaks)


hp_H, zeros_H, (u_peaks_H, F_peaks_H) = get_key_points(u, F_H)
hp_E, zeros_E, (u_peaks_E, F_peaks_E) = get_key_points(u, F_E)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(14, 6.2))

color_H = '#1f77b4'
color_E = '#e69f00'

# Основні криві F_H(u) та F_E(u)
line_H, = ax.plot(u, F_H, color=color_H, linewidth=1.6, label=r'$F_H(u)$')
line_E, = ax.plot(u, F_E, color=color_E, linewidth=1.6, label=r'$F_E(u)$')

# Горизонтальна лінія осі 0
ax.axhline(0, color='gray', linewidth=0.6, alpha=0.5)

# Пунктирні лінії та маркери рівня 1/2 потужності
ax.plot([0, hp_H[0], hp_H[0]], [hp_H[2], hp_H[2], 0], color=color_H, linestyle=':', linewidth=1.0)
ax.plot([0, hp_E[0], hp_E[0]], [hp_E[2], hp_E[2], 0], color=color_E, linestyle=':', linewidth=1.0)

sc_hp_H = ax.scatter([hp_H[0]], [hp_H[2]], facecolors='none', edgecolors=color_H, s=45, linewidths=1.3, zorder=6,
                     label=rf'½ потужності H ($u = {hp_H[0]:.2f}, \theta = {hp_H[1]:.2f}^\circ$)')
sc_hp_E = ax.scatter([hp_E[0]], [hp_E[2]], facecolors=color_E, edgecolors=color_E, s=30, zorder=6,
                     label=rf'½ потужності E ($u = {hp_E[0]:.2f}, \theta = {hp_E[1]:.2f}^\circ$)')

# Без першого початкового маркера нижче 0, далі через 1 значення
sc_max_H = ax.scatter(u_peaks_H[1::2], F_peaks_H[1::2], marker='^', color=color_H, s=22, zorder=5, label='Максимуми H')
sc_max_E = ax.scatter(u_peaks_E[1::2], F_peaks_E[1::2], marker='^', color=color_E, s=22, zorder=5, label='Максимуми E')

# Маркери мінімумів модуля через 1 значення
sc_min_H = ax.scatter(zeros_H[::2], np.zeros_like(zeros_H[::2]), marker='x', color=color_H, s=22, linewidths=1.2, zorder=5,
                      label='Мінімуми модуля H')
sc_min_E = ax.scatter(zeros_E[::2], np.zeros_like(zeros_E[::2]), marker='x', color=color_E, s=22, linewidths=1.2, zorder=5,
                      label='Мінімуми модуля E')

ax.set_xlim(0, u_max)
ax.set_ylim(-0.2, 1.05)
ax.set_yticks(np.arange(-0.2, 1.01, 0.1))
ax.set_xticks(np.arange(0, int(u_max) + 1, 10))

ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#e0e0e0')
ax.set_xlabel(r'$u = kR_0\sin\theta$', fontsize=10)
ax.set_ylabel(r'Нормоване поле $F(u)$', fontsize=10)
ax.set_title(f'ДС дзеркальної антени у площинах H та E\n$\\lambda = {lam}$ м, $D = {D}$ м, $f = {f}$ м', fontsize=11,
             pad=10)

handles = [line_H, line_E, sc_max_H, sc_min_H, sc_max_E, sc_min_E, sc_hp_H, sc_hp_E]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.14),
          ncol=4, frameon=True, fontsize=8.5, handletextpad=0.5, columnspacing=1.2)

plt.tight_layout()
plt.savefig('ds_var13_styled.png', dpi=300, bbox_inches='tight')
plt.show()

# Полярний графік
theta_deg = np.linspace(-90, 90, 180001)
theta_rad = np.deg2rad(theta_deg)
u_pol = k * R0 * np.sin(np.abs(theta_rad))
t1_p = 0.74 * calc_term1(u_pol, v)
t2_p = 0.26 * calc_term2(u_pol)
t3_p = 0.25 * calc_term3(u_pol, 1.5 * v)
factor_p = (np.cos(theta_rad / 2.0) ** 2) / norm_denom
FE_pol = np.abs(factor_p * (t1_p + t2_p + t3_p))
FH_pol = np.abs(factor_p * (t1_p + t2_p - t3_p))

fig_pol, ax_pol = plt.subplots(figsize=(8.5, 7), subplot_kw={'projection': 'polar'})

p_line_H, = ax_pol.plot(theta_rad, FH_pol, color=color_H, linewidth=1.6, label=r'$|F_H(\theta)|$')
p_line_E, = ax_pol.plot(theta_rad, FE_pol, color=color_E, linewidth=1.6, label=r'$|F_E(\theta)|$')

level_hp = 1 / np.sqrt(2)
arc_ang = np.linspace(-np.pi / 2, np.pi / 2, 400)
p_line_hp, = ax_pol.plot(arc_ang, np.full_like(arc_ang, level_hp), color='gray', linestyle=':', linewidth=1.0,
                         label='Рівень 0,707')

p_hp_H = ax_pol.scatter(np.deg2rad([-hp_H[1], hp_H[1]]), [level_hp, level_hp], facecolors='none',
                        edgecolors=color_H, s=45, linewidths=1.3, zorder=6,
                        label=rf'½ потужності H ($\theta = {hp_H[1]:.2f}^\circ$)')
p_hp_E = ax_pol.scatter(np.deg2rad([-hp_E[1], hp_E[1]]), [level_hp, level_hp], facecolors=color_E,
                        edgecolors=color_E, s=30, zorder=6,
                        label=rf'½ потужності E ($\theta = {hp_E[1]:.2f}^\circ$)')

th_peaks_H = np.rad2deg(np.arcsin(np.clip(u_peaks_H[:3] / u_max, 0, 1)))
val_peaks_H = np.abs(F_peaks_H[:3])
p_max_H = ax_pol.scatter(np.deg2rad(np.concatenate([-th_peaks_H, th_peaks_H])),
                         np.concatenate([val_peaks_H, val_peaks_H]),
                         marker='^', color=color_H, s=22, zorder=5, label='Максимуми H')

th_peaks_E = np.rad2deg(np.arcsin(np.clip(u_peaks_E[:3] / u_max, 0, 1)))
val_peaks_E = np.abs(F_peaks_E[:3])
p_max_E = ax_pol.scatter(np.deg2rad(np.concatenate([-th_peaks_E, th_peaks_E])),
                         np.concatenate([val_peaks_E, val_peaks_E]),
                         marker='^', color=color_E, s=22, zorder=5, label='Максимуми E')

ax_pol.set_theta_zero_location('N')
ax_pol.set_theta_direction(-1)
ax_pol.set_thetamin(-90)
ax_pol.set_thetamax(90)
ax_pol.set_ylim(0, 1.05)
ax_pol.set_yticks(np.arange(0.2, 1.01, 0.2))
ax_pol.set_rlabel_position(82)
ax_pol.set_thetagrids(np.arange(-90, 91, 15))
ax_pol.grid(True, which='major', linestyle='-', linewidth=0.5, color='#e0e0e0')
ax_pol.set_title(f'Полярна ДС дзеркальної антени у площинах H та E\n$\\lambda = {lam}$ м, $D = {D}$ м, $f = {f}$ м',
                 fontsize=11, pad=18)

pol_handles = [p_line_H, p_line_E, p_line_hp, p_hp_H, p_hp_E, p_max_H, p_max_E]
ax_pol.legend(handles=pol_handles, loc='upper center', bbox_to_anchor=(0.5, 0.08),
              ncol=3, frameon=True, fontsize=8.5, handletextpad=0.5, columnspacing=1.2)

plt.tight_layout()
plt.savefig('ds_var13_polar.png', dpi=300, bbox_inches='tight')
plt.show()