import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

def js_gaussians(sig1, sig2, logbase=np.e):
    p = lambda x: norm.pdf(x, 0.0, sig1)
    q = lambda x: norm.pdf(x, 0.0, sig2)
    m = lambda x: 0.5 * (p(x) + q(x))

    def integrand_p(x):
        px = p(x)
        mx = m(x)
        return 0.5 * px * np.log((px + 1e-8) / (mx + 1e-8))

    def integrand_q(x):
        qx = q(x)
        mx = m(x)
        return 0.5 * qx * np.log((qx + 1e-8) / (mx + 1e-8))

    val_p, _ = quad(integrand_p, -np.inf, np.inf, limit=200)
    val_q, _ = quad(integrand_q, -np.inf, np.inf, limit=200)
    js = val_p + val_q

    if logbase == 2:
        js /= np.log(2.0)

    return js

ratios = np.logspace(-2, 2, 100)
vals_nats = [js_gaussians(r, 1.0) for r in ratios]
vals_bits = [v / np.log(2.0) for v in vals_nats]

plt.figure(figsize=(6,4))
plt.semilogx(ratios, vals_bits)
plt.axhline(1.0, ls='--')
plt.axvline(1.0, ls='--')
plt.xlabel("sigma_p / sigma_q")
plt.ylabel("JS divergence (bits)")
plt.title("JS between N(0, sigma_p^2) and N(0, sigma_q^2)")
plt.show()
