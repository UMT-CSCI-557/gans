import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_gan(X_tensor,sample_list):
    fig, ax = plt.subplots()
    ax.plot(*X_tensor.T,'r.',alpha=0.5)
    line, = ax.plot(*sample_list[0].T,'k.')
    ax.set_ylim(-4, 4)
    ax.set_xlim(-4, 4)
    fig.set_size_inches(12,12)

    def update(frame):
        line.set_data(sample_list[frame][:,0],sample_list[frame][:,1])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(sample_list), interval=100, blit=True, repeat=False, repeat_delay=5000)
    plt.show()

def plot_discriminator_vector_field(D, real, fake, xlim=(-2, 2), ylim=(-2, 2), n=40, device="cpu"):
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    x = torch.tensor(pts, dtype=torch.float32, device=device, requires_grad=True)

    scores = D(x)                    # shape (N,1) or (N,)
    scores_flat = scores.reshape(-1)

    grads = torch.autograd.grad(
        outputs=scores_flat.sum(),
        inputs=x,
        create_graph=False
    )[0]

    S = scores_flat.detach().cpu().numpy().reshape(n, n)
    G = grads.detach().cpu().numpy().reshape(n, n, 2)

    U = G[:, :, 0]
    V = G[:, :, 1]

    plt.figure(figsize=(7, 7))
    plt.contourf(XX, YY, S, levels=30, alpha=0.6)
    plt.quiver(XX, YY, U, V, angles="xy", scale_units="xy", scale=20, width=0.002)

    #real_np = real.detach().cpu().numpy()
    #fake_np = fake.detach().cpu().numpy()

    plt.scatter(real[:, 0], real[:, 1], s=8, alpha=0.4, label="real")
    #plt.scatter(fake_np[:, 0], fake_np[:, 1], s=8, alpha=0.4, label="fake")
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().set_aspect("equal")
    plt.title("Discriminator score contours and gradient field")
    plt.show()



