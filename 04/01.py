import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams["font.size"] = 13
plt.rcParams["figure.figsize"] = 16, 12

class MultiNewton(object):
    def __init__(self, f, dx0_f, dx1_f, dx2_f, grad_f, hesse):
        self.f = f
        self.dx0_f = dx0_f
        self.dx1_f = dx1_f
        self.dx2_f = dx2_f
        self.grad_f = grad_f
        self.hesse = hesse

    def _compute_dx(self, bar_x):
        grad = self.grad_f(bar_x)
        hesse = self.hesse(bar_x)
        dx = np.linalg.solve(hesse, -grad)
        return dx

    def solve(self, init_x, n_iter=100, tol=0.01, step_width=1.0):
        self.hist = np.zeros(n_iter)
        bar_x = init_x
        for i in range(n_iter):
            dx = self._compute_dx(bar_x)
            # update
            x = bar_x + dx*step_width
            print("x = [{0:.2f} {1:.2f}]".format(x[0], x[1]))

            bar_x = x
            norm_dx = np.linalg.norm(dx)
            self.hist[i] += norm_dx
            if norm_dx < tol:
                self.hist = self.hist[:i]
                break
        return x

def _main():
    f = lambda x: x[0]**2 + x[1]**2 - x[2]**2 +4*x[0]*x[2] + 4*x[1]*x[2] - 3*x[0] + 2*x[1] +x[2] - 6
    dx0_f = lambda x: 2*x[0] + 4*x[2] + 3
    dx1_f = lambda x: 2*x[1] + 4*x[2] + 2
    dx2_f = lambda x: -2*x[2] + 4*x[0] + 4*x[1] + 1
    grad_f = lambda x: np.array([dx0_f(x), dx1_f(x), dx2_f(x)])
    hesse = lambda x: np.array([[2, 0, 4],[0, 2, 4],[4, 4, -2]])

    init_x = np.array([10, 8])
    solver = MultiNewton(f, dx0_f, dx1_f, dx2_f, grad_f, hesse)
    res = solver.solve(init_x=init_x, n_iter=100, step_width=1.0)
    print("Solution is x = [{0:.2f} {1:.2f} {2:.2f}]".format(res[0], res[1], res[2]))

    errors = solver.hist
    epochs = np.arange(0, errors.shape[0])

    plt.plot(epochs, errors)
    plt.tight_layout()
    plt.savefig('error_mult.png')

if __name__ == "__main__":
    _main()
