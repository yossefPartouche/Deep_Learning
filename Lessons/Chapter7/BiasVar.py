"""
Bias-Variance Visualization Utilities
"""
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, IntSlider, FloatSlider

class BiasVarianceVisualizer:
    """Interactive visualization of bias-variance tradeoff"""
    
    def __init__(self, A=1.0, phi0=0, phi1=2*np.pi):
        """
        Initialize visualizer with ground truth function parameters
        
        Args:
            A: Amplitude of sine wave
            phi0: Phase offset
            phi1: Frequency parameter
        """
        self.A = A
        self.phi0 = phi0
        self.phi1 = phi1
    
    def true_function(self, x):
        """Ground truth function"""
        return self.A * np.sin(self.phi0 + self.phi1 * x)
    
    def piecewise_linear_model(self, x, x_train, y_train, n_segments):
        """
        Fit piecewise linear model with n_segments
        
        Args:
            x: Test points
            x_train: Training input points
            y_train: Training output points
            n_segments: Number of linear segments
            
        Returns:
            Predicted values at x
        """
        boundaries = np.linspace(0, 1, n_segments + 1)
        y_pred = np.zeros_like(x)
        
        for i in range(len(x)):
            seg_idx = np.searchsorted(boundaries[1:], x[i])
            seg_idx = min(seg_idx, n_segments - 1)
            
            left, right = boundaries[seg_idx], boundaries[seg_idx + 1]
            mask = (x_train >= left) & (x_train <= right)
            
            if np.sum(mask) > 0:
                x_seg = x_train[mask]
                y_seg = y_train[mask]
                if len(x_seg) > 1:
                    y_pred[i] = np.interp(x[i], x_seg, y_seg)
                else:
                    y_pred[i] = y_seg[0]
            else:
                nearest_idx = np.argmin(np.abs(x_train - x[i]))
                y_pred[i] = y_train[nearest_idx]
        
        return y_pred
    
    def plot(self, n_segments=3, noise_level=0.2, n_train=20, seed=42):
        """
        Plot model fits with uncertainty regions
        
        Args:
            n_segments: Model complexity (number of segments)
            noise_level: Standard deviation of noise
            n_train: Number of training samples
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        
        x_test = np.linspace(0, 1, 200)
        y_true = self.true_function(x_test)
        
        # Train multiple models
        predictions = []
        for s in range(20):
            np.random.seed(seed + s)
            x_train = np.sort(np.random.uniform(0, 1, n_train))
            y_train = self.true_function(x_train) + np.random.normal(0, noise_level, n_train)
            y_pred = self.piecewise_linear_model(x_test, x_train, y_train, n_segments)
            predictions.append(y_pred)
        
        y_mean = np.mean(predictions, axis=0)
        y_std = np.std(predictions, axis=0)
        bias_sq = np.mean((y_mean - y_true) ** 2)
        var = np.mean(y_std ** 2)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Uncertainty band (variance)
        ax.fill_between(x_test, y_mean - y_std, y_mean + y_std, 
                         alpha=0.3, color='orange', label=f'±1 Std Dev (Variance={var:.3f})')
        
        ax.plot(x_test, y_true, 'k-', linewidth=3, label='True Function', alpha=0.8)
        ax.plot(x_test, y_mean, 'r--', linewidth=2.5, label=f'Average Model (Bias²={bias_sq:.3f})')
        
        # Sample training data
        np.random.seed(seed)
        x_train = np.sort(np.random.uniform(0, 1, n_train))
        y_train = self.true_function(x_train) + np.random.normal(0, noise_level, n_train)
        ax.scatter(x_train, y_train, s=60, alpha=0.7, color='blue', 
                   edgecolors='black', linewidths=1.5, label='Training Data', zorder=5)
        
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('y', fontsize=13)
        ax.set_title(f'Model Complexity: {n_segments} segments | Total Error = {bias_sq + var + noise_level**2:.3f}', 
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.8, 1.8])
        
        plt.tight_layout()
        plt.show()
    
    def interactive_plot(self):
        """Create interactive widget"""
        return interactive(
            self.plot,
            n_segments=IntSlider(min=1, max=20, value=3, description='Complexity:', 
                                continuous_update=False),
            noise_level=FloatSlider(min=0.0, max=0.5, step=0.05, value=0.2, description='Noise:', 
                                   continuous_update=False),
            n_train=IntSlider(min=10, max=100, step=5, value=20, description='Samples:', 
                             continuous_update=False),
            seed=IntSlider(min=0, max=100, value=42, description='Seed:', 
                          continuous_update=False)
        )