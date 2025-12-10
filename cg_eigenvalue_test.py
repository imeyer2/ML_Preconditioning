import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
import time

def generate_spd_from_eigenvalues(eigenvalues):
    """
    Generates a random SPD matrix with the specified eigenvalues.
    A = Q * diag(eigenvalues) * Q.T
    """
    N = len(eigenvalues)
    # Generate a random orthogonal matrix using QR decomposition of a random matrix
    H = np.random.randn(N, N)
    Q, _ = np.linalg.qr(H)
    
    L = np.diag(eigenvalues)
    A = Q @ L @ Q.T
    return A

def run_cg_tracking(A, b, tol=1e-8, maxiter=1000):
    """
    Runs CG and tracks the relative residual norm at each iteration.
    """
    residuals = []
    
    # Initial residual
    x0 = np.zeros_like(b)
    r0 = b - A @ x0
    norm_r0 = np.linalg.norm(r0)
    residuals.append(norm_r0)
    
    def callback(xk):
        # Calculate residual r = b - Ax manually for accuracy
        r = b - A @ xk
        norm_r = np.linalg.norm(r)
        residuals.append(norm_r)

    # Run CG
    cg(A, b, x0=x0, tol=tol, maxiter=maxiter, callback=callback)
    
    # Normalize
    return np.array(residuals) / residuals[0]

def main():
    N = 200 # Matrix size
    num_trials = 10 # Number of matrices per class to average over
    condition_numbers = [100, 1000, 10000]
    num_outliers = 190 # Number of large outliers
    
    plt.figure(figsize=(18, 8))
    plt.suptitle("Conjugate Gradient Convergence: Effect of Eigenvalue Clustering vs. Spread\n"
                 "Even with the same Condition Number ($\kappa$), clustered eigenvalues (Blue) converge much faster than spread eigenvalues (Red).", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Define mixing levels
    # 0.0 = Fully Clustered (Blue)
    # 1.0 = Fully Spread (Red)
    mix_ratios = [0.0, 0.33, 0.66, 1.0]
    colors = ['blue', 'purple', 'green', 'red']
    labels = ['Clustered (Favorable)', 'Mix 33% (Transition)', 'Mix 66% (Transition)', 'Spread (Unfavorable)']

    for idx, kappa in enumerate(condition_numbers):
        print(f"Testing Condition Number: {kappa}")
        
        histories_by_ratio = {r: [] for r in mix_ratios}
        
        for _ in range(num_trials):
            # Base Clustered Eigenvalues (Sorted)
            evals_clustered = np.ones(N)
            # Add small jitter to the cluster (e.g., +/- 0.05)
            evals_clustered[:-num_outliers] += np.random.uniform(-0.05, 0.05, N-num_outliers)
            # Ensure they stay positive and roughly around 1
            evals_clustered = np.abs(evals_clustered) 
            # Set the outliers to achieve condition number kappa (approx)
            evals_clustered[-num_outliers:] = float(kappa)
            evals_clustered = np.sort(evals_clustered)
            
            # Base Spread Eigenvalues (Sorted)
            evals_spread = np.linspace(1.0, float(kappa), N)
            
            for ratio in mix_ratios:
                # Interpolate eigenvalues
                # ratio 0 -> clustered, ratio 1 -> spread
                evals = (1 - ratio) * evals_clustered + ratio * evals_spread
                
                A = generate_spd_from_eigenvalues(evals)
                b = np.random.randn(N)
                b = b / np.linalg.norm(b)
                
                hist = run_cg_tracking(A, b)
                histories_by_ratio[ratio].append(hist)

        # Plotting for this kappa
        plt.subplot(1, 3, idx+1)
        
        # Helper to plot average curve
        def plot_avg_and_individual(histories, color, label):
            max_l = max(len(h) for h in histories)
            # Plot individuals
            for h in histories:
                plt.plot(h, color=color, alpha=0.15)
            
            # Compute average
            # We extend converged runs with their last value for averaging
            sum_arr = np.zeros(max_l)
            for h in histories:
                padded = np.pad(h, (0, max_l - len(h)), 'edge')
                sum_arr += padded
            avg_curve = sum_arr / len(histories)
            
            plt.plot(avg_curve, color=color, label=label, linewidth=2.5)

        for i, ratio in enumerate(mix_ratios):
            plot_avg_and_individual(histories_by_ratio[ratio], colors[i], labels[i])

        plt.yscale('log')
        plt.title(rf"Condition Number $\kappa \approx {kappa}$" + "\n(Higher $\kappa$ implies harder problem)")
        plt.xlabel("CG Iteration Count")
        plt.ylabel(r"Relative Residual Error ($\|r\|/\|r_0\|$)")
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout to make room for suptitle
    print("Saving plot to cg_convergence_comparison.png...")
    plt.savefig('cg_convergence_comparison.png')
    plt.show()

if __name__ == "__main__":
    main()
