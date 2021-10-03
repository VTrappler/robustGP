#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Imports
import numpy as np
import matplotlib.pyplot as plt
import dill
import seaborn as sns
import robustGP.tools as tools

# for Palatino and other serif fonts use:
plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    'image.cmap': u'viridis',
    'figure.figsize': [5.748031686730317, 3.552478950810724],
    'savefig.dpi': 200
})
plt.rc('text.latex', preamble=r"\usepackage{amsmath} \usepackage{amssymb}")
graphics_folder = '/home/victor/collab_article/adaptive/figures/'


x, y = np.linspace(0, 1, 20), np.linspace(0, 1, 20)
(XY, (xmg, ymg)) = tools.pairify((x, y))

with open('/home/victor/robustGP/IMSE_alpha.data', 'rb') as f:
    IMSE_alpha = dill.load(f)

with open('/home/victor/robustGP/branin_aIMSE.data', 'rb') as f:
    aIMSE_data = dill.load(f)

aIMSE_alpha = aIMSE_data['aIMSE_alpha']

IMSE_alpha['adjusted']
color_corresp = {'0.0': sns.color_palette()[0],
                 '1.0': sns.color_palette()[2],
                 '2.0': sns.color_palette()[3]}


plt.subplot(1, 2, 2)
for (k, v) in IMSE_alpha['adjusted'].items():
    plt.plot(v, color=color_corresp[k], alpha=0.3)
    plt.plot(v.mean(1), color=color_corresp[k], lw=2, label=r'$\alpha$={}'.format(k))
plt.title('IMSE\n Prediction variance \n and adjustment')
plt.plot(aIMSE_alpha['2.0'], label=r'aIMSE')
plt.legend()
plt.ylabel(r'IMSE')
plt.xlabel(r'Iterations')
plt.yscale('log')
plt.subplot(1, 2, 1)
for (k, v) in IMSE_alpha['vanilla'].items():
    plt.plot(v, color=color_corresp[k], alpha=0.3)
    plt.plot(v.mean(1), color=color_corresp[k], lw=2, label=r'$\alpha$={}'.format(k))
plt.legend()
plt.title('IMSE \n Maximization of the prediction variance\n ' + r'of $Z$')
plt.yscale('log')
plt.ylabel(r'IMSE')
plt.xlabel(r'Iterations')
plt.tight_layout()
# plt.savefig(graphics_folder + 'IMSE_predictionvariance_Delta_adjustment.png')
plt.show()


with open('/home/victor/robustGP/aIMSE_alpha.data', 'rb') as f:
    augmented_IMSE_alpha = dill.load(f)
branin_ph = augmented_IMSE_alpha['branin']



plt.figure(figsize=(6, 6))
for j, k in enumerate(['0.0', '1.0', '2.0', '3.0'], 1):
    plt.subplot(2, 2, j)
    plt.contourf(xmg, ymg, augmented_IMSE_alpha[k].reshape(20, 20))
    cb = plt.colorbar()
    cb.ax.set_yticklabels(["{:.2}".format(i) for i in cb.get_ticks()]) # set ticks of your format
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_fontsize(8)
    plt.scatter(branin_ph.gp.X_train_[:, 0], branin_ph.gp.X_train_[:, 1], c="w", s=2)
    plt.title(r'a-IMSE, $\alpha={}$'.format(k))
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$u$')
plt.suptitle(r'Expected augmented IMSE (a-IMSE) of $\Delta_{\alpha}$')
plt.tight_layout()
plt.savefig(graphics_folder + 'aIMSE_diff_alpha.png')
plt.show()




with open("/home/victor/robustGP/branin_aIMSE.data", "rb") as f:
    data = dill.load(f)

branin_aIMSE = data['branin']
alpha = 2.0



ax1 = plt.subplot(2, 2, 1)
ax1.set_title(r"Prediction: $m_\Delta$")
pred, var = branin_aIMSE.predict_GPdelta_product(x, y, alpha=alpha)
ax1.contourf(xmg, ymg, pred.reshape(20, 20))
ax1.scatter(branin_aIMSE.gp.X_train_[:, 0], branin_aIMSE.gp.X_train_[:, 1], c="b")
ax1.set_aspect("equal")
ax2 = plt.subplot(2, 2, 2)
ax2.contourf(xmg, ymg, var.reshape(20, 20))
ax2.scatter(branin_aIMSE.gp.X_train_[:-1, 0], branin_aIMSE.gp.X_train_[:-1, 1], c="b")
ax2.scatter(branin_aIMSE.gp.X_train_[-1, 0], branin_aIMSE.gp.X_train_[-1, 1], c="r")
ax2.set_aspect("equal")
ax2.set_title(r"Prediction variance of $\Delta$: $\sigma_\Delta$")
ax3 = plt.subplot(2, 2, 3)
ax3.set_title(r"Prediction: $m_Z$")
mZ, sZ = branin_aIMSE.predict(XY, return_std=True)
ax3.contourf(xmg, ymg, mZ.reshape(20, 20))
ax3.scatter(branin_aIMSE.gp.X_train_[:-1, 0], branin_aIMSE.gp.X_train_[:-1, 1], c="b")
ax3.scatter(branin_aIMSE.gp.X_train_[-1, 0], branin_aIMSE.gp.X_train_[-1, 1], c="r")
ax3.set_aspect("equal")
ax4 = plt.subplot(2, 2, 4)
ax4.contourf(xmg, ymg, sZ.reshape(20, 20))
ax4.set_aspect("equal")
ax4.set_title(r"Prediction variance $\sigma^2_Z$")
plt.tight_layout()
# plt.savefig(f"/home/victor/robustGP/robustGP/dump/predvar_{i+shift:02d}.png")
# with open("/home/victor/robustGP/robustGP/dump/imse_ivpc.txt", "a+") as f:
#     f.write(f"{i+shift}, {var.mean()}, {ivpc.mean()}\n")
# plt.savefig(graphics_folder + 'prediction_mZ_mDelta.png')
plt.show()

plt.subplot(2, 2, 1)
plt.contourf(xmg, ymg, sZ.reshape(20, 20))
plt.title(r"$\sigma^2_Z$")
plt.subplot(2, 2, 2)
plt.contourf(xmg, ymg, var.reshape(20, 20))
plt.scatter(branin_aIMSE.gp.X_train_[:-1, 0], branin_aIMSE.gp.X_train_[:-1, 1], c="b")
plt.scatter(branin_aIMSE.gp.X_train_[-1, 0], branin_aIMSE.gp.X_train_[-1, 1], c="r")
plt.title(r"$\sigma^2_\Delta$")
plt.subplot(2, 2, 3)
plt.contourf(xmg, ymg, aa.reshape(20, 20))
plt.scatter(branin_aIMSE.gp.X_train_[:, 0], branin_aIMSE.gp.X_train_[:, 1], c="b")
plt.title(r'augmented IMSE')
plt.tight_layout()
for ax in plt.gcf().axes:
    ax.set_aspect('equal')
plt.show()





# EOF ----------------------------------------------------------------------
