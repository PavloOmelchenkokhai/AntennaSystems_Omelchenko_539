import matplotlib.pyplot as plt
import numpy as np

# Варіант 13
lambd = 3.7
ap = 16.0
bp = 16.0

# Діапазон кутів від 0 до 90 градусів (декартові графіки)
theta_deg = np.linspace(0, 90, 1801)
theta_rad = np.radians(theta_deg)

# Множник Гюйгенса
F_I = (1 + np.cos(theta_rad)) / 2.0

# Розрахунок для площини E
arg_E = (np.pi * bp / lambd) * np.sin(theta_rad)
with np.errstate(divide='ignore', invalid='ignore'):
  F_CE = np.where(
      np.abs(arg_E) < 1e-9, 1.0, np.abs(np.sin(arg_E) / arg_E)
  )
FE = F_CE * F_I

# Розрахунок для площини H
arg_H = (np.pi * ap / lambd) * np.sin(theta_rad)
denom_H = 1.0 - (2.0 * ap / lambd * np.sin(theta_rad)) ** 2

with np.errstate(divide='ignore', invalid='ignore'):
  F_CH = np.where(
      np.abs(denom_H) < 1e-6, np.pi / 4.0, np.abs(np.cos(arg_H) / denom_H)
  )
FH = F_CH * F_I

nulls_E = [13.37, 27.55, 43.93, 67.67]
maxes_E = [20.30, 35.32, 54.03]
theta_0707_E = 5.85

nulls_H = [20.30, 35.32, 54.03]
maxes_H = [27.55, 43.93, 67.67]
theta_0707_H = 7.85


# ДС у площині E (Декартові координати)
plt.figure(figsize=(10, 6))
plt.plot(
    theta_deg,
    F_I,
    'k--',
    linewidth=1.2,
    label=r'$|F_{IE}(\Theta)|$ (Множник Гюйгенса)',
)
plt.plot(
    theta_deg,
    F_CE,
    'k-.',
    linewidth=1.2,
    alpha=0.7,
    label=r'$|F_{CE}(\Theta)|$ (Множник апертури)',
)
plt.plot(
    theta_deg,
    FE,
    'b-',
    linewidth=2.0,
    label=r'$|F_E(\Theta)|$ (Підсумкова ДС)',
)

plt.plot(
    [0, theta_0707_E, theta_0707_E],
    [0.707, 0.707, 0],
    'r--',
    linewidth=1.2,
    label=fr'Рівень 0.707 ($\Theta_{{0.707}} \approx {theta_0707_E}^\circ$)',
)
plt.scatter([theta_0707_E], [0.707], color='red', zorder=5)

for i, th in enumerate(nulls_E):
  plt.scatter([th], [0], color='purple', zorder=5)
  plt.annotate(
      fr'$\Theta_{{0{i+1}}}$',
      (th, 0),
      textcoords='offset points',
      xytext=(0, -18),
      ha='center',
      fontsize=10,
      color='purple',
  )

for i, th in enumerate(maxes_E):
  val = np.interp(th, theta_deg, FE)
  plt.scatter([th], [val], color='green', zorder=5)
  plt.annotate(
      fr'$\Theta_{{m{i+1}}}$',
      (th, val),
      textcoords='offset points',
      xytext=(0, 8),
      ha='center',
      fontsize=10,
      color='green',
  )

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title(r'Рис. 3.8. Розрахункова ДС пірамідального рупора у площині E')
plt.xlabel(r'Кут $\Theta^\circ$, град')
plt.ylabel(r'Нормована амплітуда $|F_E(\Theta)|$')
plt.xlim(0, 90)
plt.ylim(0, 1.05)
plt.xticks(np.arange(0, 91, 15))
plt.yticks([0, 0.21, 0.5, 0.707, 1.0])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('DS_E_plane_Variant13.png', dpi=300)
plt.show()


# ДС у площині H (Декартові координати)
plt.figure(figsize=(10, 6))
plt.plot(
    theta_deg,
    F_I,
    'k--',
    linewidth=1.2,
    label=r'$|F_{IH}(\Theta)|$ (Множник Гюйгенса)',
)
plt.plot(
    theta_deg,
    F_CH,
    'k-.',
    linewidth=1.2,
    alpha=0.7,
    label=r'$|F_{CH}(\Theta)|$ (Множник апертури)',
)
plt.plot(
    theta_deg,
    FH,
    'r-',
    linewidth=2.0,
    label=r'$|F_H(\Theta)|$ (Підсумкова ДС)',
)

plt.plot(
    [0, theta_0707_H, theta_0707_H],
    [0.707, 0.707, 0],
    'r--',
    linewidth=1.2,
    label=fr'Рівень 0.707 ($\Theta_{{0.707}} \approx {theta_0707_H}^\circ$)',
)
plt.scatter([theta_0707_H], [0.707], color='red', zorder=5)

for i, th in enumerate(nulls_H):
  plt.scatter([th], [0], color='purple', zorder=5)
  plt.annotate(
      fr'$\Theta_{{0{i+1}}}$',
      (th, 0),
      textcoords='offset points',
      xytext=(0, -18),
      ha='center',
      fontsize=10,
      color='purple',
  )

for i, th in enumerate(maxes_H):
  val = np.interp(th, theta_deg, FH)
  plt.scatter([th], [val], color='green', zorder=5)
  plt.annotate(
      fr'$\Theta_{{m{i+1}}}$',
      (th, val),
      textcoords='offset points',
      xytext=(0, 8),
      ha='center',
      fontsize=10,
      color='green',
  )

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title(r'Рис. 3.9. Розрахункова ДС пірамідального рупора у площині H')
plt.xlabel(r'Кут $\Theta^\circ$, град')
plt.ylabel(r'Нормована амплітуда $|F_H(\Theta)|$')
plt.xlim(0, 90)
plt.ylim(0, 1.05)
plt.xticks(np.arange(0, 91, 15))
plt.yticks([0, 0.063, 0.5, 0.707, 1.0])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('DS_H_plane_Variant13.png', dpi=300)
plt.show()


# ==========================================
# 3. ПОЛЯРНІ (РАДІАЛЬНІ) ГРАФІКИ ДС (-90° ... 90°)
# ==========================================
theta_deg_sym = np.linspace(-90, 90, 1801)
theta_rad_sym = np.radians(theta_deg_sym)

F_I_sym = (1 + np.cos(theta_rad_sym)) / 2.0

# Площина E (симетрична)
arg_E_sym = (np.pi * bp / lambd) * np.sin(theta_rad_sym)
with np.errstate(divide='ignore', invalid='ignore'):
  F_CE_sym = np.where(
      np.abs(arg_E_sym) < 1e-9, 1.0, np.abs(np.sin(arg_E_sym) / arg_E_sym)
  )
FE_sym = F_CE_sym * F_I_sym

# Площина H (симетрична)
arg_H_sym = (np.pi * ap / lambd) * np.sin(theta_rad_sym)
denom_H_sym = 1.0 - (2.0 * ap / lambd * np.sin(theta_rad_sym)) ** 2
with np.errstate(divide='ignore', invalid='ignore'):
  F_CH_sym = np.where(
      np.abs(denom_H_sym) < 1e-6, np.pi / 4.0, np.abs(np.cos(arg_H_sym) / denom_H_sym)
  )
FH_sym = F_CH_sym * F_I_sym

# Полярна ДС у площині E
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.plot(theta_rad_sym, FE_sym, 'b-', linewidth=2.0, label=r'$|F_E(\Theta)|$')
ax.set_theta_zero_location('N')  # Нуль зверху
ax.set_theta_direction(-1)     # За годинниковою стрілкою
ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.set_rmax(1.0)
ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)
ax.set_title(r'Полярна ДС пірамідального рупора у площині E', va='bottom', fontsize=12)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('DS_E_plane_polar_Variant13.png', dpi=300)
plt.show()

# Полярна ДС у площині H
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.plot(theta_rad_sym, FH_sym, 'r-', linewidth=2.0, label=r'$|F_H(\Theta)|$')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.set_rmax(1.0)
ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)
ax.set_title(r'Полярна ДС пірамідального рупора у площині H', va='bottom', fontsize=12)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('DS_H_plane_polar_Variant13.png', dpi=300)
plt.show()