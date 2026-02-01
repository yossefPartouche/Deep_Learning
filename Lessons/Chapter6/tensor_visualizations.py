import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import ipywidgets as widgets

# ==================== SCALAR DERIVATIVES ====================

def visualize_scalar_wrt_vector_simple(n=3):
    """Visualization for scalar w.r.t. vector."""
    
    def create_frame(step):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
        
        # Input vector
        ax1.set_xlim(-0.5, 1.5)
        ax1.set_ylim(-0.5, n + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Input: $\\mathbf{v} \\in \\mathbb{R}^3$', fontsize=14)
        
        for i in range(n):
            color = 'gold' if i == step and step < n else 'lightgray'
            rect = patches.Rectangle((0, n-1-i), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax1.add_patch(rect)
            if i == step and step < n:
                ax1.text(1.3, n-1-i+0.4, '←', fontsize=20, va='center', color='red')
        
        # Scalar output
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Output: $s \\in \\mathbb{R}$', fontsize=14)
        
        scalar_box = patches.Rectangle((0, 0), 1, 0.8, 
                                        linewidth=2, edgecolor='black', 
                                        facecolor='lightcoral')
        ax2.add_patch(scalar_box)
        ax2.text(0.5, 0.4, 's', fontsize=16, ha='center', va='center', fontweight='bold')
        
        # Gradient
        ax3.set_xlim(-0.5, 1.5)
        ax3.set_ylim(-0.5, n + 0.5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < n:
            ax3.set_title(f'Step {step+1}/{n}: $\\frac{{\\partial s}}{{\\partial v_{{{step+1}}}}}$', 
                         fontsize=14, fontweight='bold')
        else:
            ax3.set_title('Gradient: $\\frac{\\partial s}{\\partial \\mathbf{v}}$ Complete!', 
                         fontsize=14, fontweight='bold', color='green')
        
        for i in range(n):
            color = 'lightblue' if i <= step and step < n else ('lightblue' if step >= n else 'lightgray')
            rect = patches.Rectangle((0, n-1-i), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax3.add_patch(rect)
            if i == step and step < n:
                ax3.text(-0.3, n-1-i+0.4, '→', fontsize=20, va='center', color='blue')
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, step=widgets.IntSlider(min=0, max=n, value=0, 
                                description='Step:'))

def visualize_scalar_wrt_matrix_simple(m=2, n=3):
    """Visualization for scalar w.r.t. matrix."""
    
    def create_frame(step):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        total_steps = m * n
        
        # Input matrix
        ax1.set_xlim(-0.5, n + 0.5)
        ax1.set_ylim(-0.5, m + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Input: $\\mathbf{{A}} \\in \\mathbb{{R}}^{{{m} \\times {n}}}$', fontsize=14)
        
        for i in range(m):
            for j in range(n):
                is_current = (i * n + j == step and step < total_steps)
                color = 'gold' if is_current else 'lightgray'
                rect = patches.Rectangle((j, m-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax1.add_patch(rect)
        
        # Scalar output
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Output: $s \\in \\mathbb{R}$', fontsize=14)
        
        scalar_box = patches.Rectangle((0, 0), 1, 0.8, 
                                        linewidth=2, edgecolor='black', 
                                        facecolor='lightcoral')
        ax2.add_patch(scalar_box)
        ax2.text(0.5, 0.4, 's', fontsize=16, ha='center', va='center', fontweight='bold')
        
        # Gradient matrix
        ax3.set_xlim(-0.5, n + 0.5)
        ax3.set_ylim(-0.5, m + 0.5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            row_idx = step // n + 1
            col_idx = step % n + 1
            ax3.set_title(f'Step {step+1}/{total_steps}: $\\frac{{\\partial s}}{{\\partial A_{{{row_idx}{col_idx}}}}}$', 
                         fontsize=14, fontweight='bold')
        else:
            ax3.set_title('Gradient: $\\frac{\\partial s}{\\partial \\mathbf{A}}$ Complete!', 
                         fontsize=14, fontweight='bold', color='green')
        
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                color = 'lightblue' if idx <= step and step < total_steps else ('lightblue' if step >= total_steps else 'lightgray')
                rect = patches.Rectangle((j, m-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax3.add_patch(rect)
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, step=widgets.IntSlider(min=0, max=m*n, value=0, 
                                description='Step:'))

def visualize_scalar_wrt_tensor_simple(m=2, n=2, p=3):
    """Visualization for scalar w.r.t. tensor with 3D perspective."""
    
    def create_frame(step):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        total_steps = m * n * p
        depth_offset = 0.3
        
        # Input tensor - 3D perspective
        ax1.set_xlim(-0.5, n + p*depth_offset + 0.5)
        ax1.set_ylim(-0.5, m + p*depth_offset + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Input: $\\mathcal{{T}} \\in \\mathbb{{R}}^{{{m} \\times {n} \\times {p}}}$', fontsize=14)
        
        for k in range(p):
            x_offset = k * depth_offset
            y_offset = k * depth_offset
            
            for i in range(m):
                for j in range(n):
                    idx = k * (m * n) + i * n + j
                    is_current = (idx == step and step < total_steps)
                    color = 'gold' if is_current else 'lightgray'
                    alpha = 1.0 - (k * 0.15)
                    
                    rect = patches.Rectangle((j + x_offset, m-1-i + y_offset), 0.9, 0.9, 
                                              linewidth=2, edgecolor='black', 
                                              facecolor=color, alpha=alpha,
                                              zorder=p-k)
                    ax1.add_patch(rect)
            
            ax1.text(n/2 - 0.5 + x_offset, -0.5 + y_offset, f'k={k+1}', 
                    fontsize=9, ha='center', zorder=p-k)
        
        # Draw connecting lines
        for i in range(m):
            for j in range(n):
                x_start = j
                y_start = m-1-i
                x_end = j + (p-1) * depth_offset
                y_end = m-1-i + (p-1) * depth_offset
                ax1.plot([x_start, x_end], [y_start, y_end], 
                        'k-', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Scalar output
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Output: $s \\in \\mathbb{R}$', fontsize=14)
        
        scalar_box = patches.Rectangle((0, 0), 1, 0.8, 
                                        linewidth=2, edgecolor='black', 
                                        facecolor='lightcoral')
        ax2.add_patch(scalar_box)
        ax2.text(0.5, 0.4, 's', fontsize=16, ha='center', va='center', fontweight='bold')
        
        # Gradient tensor - 3D perspective
        ax3.set_xlim(-0.5, n + p*depth_offset + 0.5)
        ax3.set_ylim(-0.5, m + p*depth_offset + 0.5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            ax3.set_title(f'Step {step+1}/{total_steps}', fontsize=14, fontweight='bold')
        else:
            ax3.set_title('Gradient Complete!', fontsize=14, fontweight='bold', color='green')
        
        for k in range(p):
            x_offset = k * depth_offset
            y_offset = k * depth_offset
            
            for i in range(m):
                for j in range(n):
                    idx = k * (m * n) + i * n + j
                    color = 'lightblue' if idx <= step and step < total_steps else ('lightblue' if step >= total_steps else 'lightgray')
                    alpha = 1.0 - (k * 0.15)
                    
                    rect = patches.Rectangle((j + x_offset, m-1-i + y_offset), 0.9, 0.9, 
                                              linewidth=2, edgecolor='black', 
                                              facecolor=color, alpha=alpha,
                                              zorder=p-k)
                    ax3.add_patch(rect)
            
            ax3.text(n/2 - 0.5 + x_offset, -0.5 + y_offset, f'k={k+1}', 
                    fontsize=9, ha='center', zorder=p-k)
        
        # Draw connecting lines
        for i in range(m):
            for j in range(n):
                x_start = j
                y_start = m-1-i
                x_end = j + (p-1) * depth_offset
                y_end = m-1-i + (p-1) * depth_offset
                ax3.plot([x_start, x_end], [y_start, y_end], 
                        'k-', linewidth=0.5, alpha=0.3, zorder=0)
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, step=widgets.IntSlider(min=0, max=m*n*p, value=0, 
                                description='Step:'))

# ==================== VECTOR DERIVATIVES ====================

def visualize_vector_wrt_scalar_simple(m=3):
    """Visualization for vector w.r.t. scalar."""
    
    def create_frame(step):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
        
        # Scalar input
        ax1.set_xlim(-0.5, 1.5)
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Input: $s \\in \\mathbb{R}$', fontsize=14)
        
        scalar_box = patches.Rectangle((0, 0), 1, 0.8, 
                                        linewidth=2, edgecolor='black', 
                                        facecolor='lightcoral')
        ax1.add_patch(scalar_box)
        ax1.text(0.5, 0.4, 's', fontsize=16, ha='center', va='center', fontweight='bold')
        
        # Vector output
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(-0.5, m + 0.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title('Output: $\\mathbf{a} \\in \\mathbb{R}^3$', fontsize=14)
        
        for i in range(m):
            color = 'gold' if i == step and step < m else 'lightgray'
            rect = patches.Rectangle((0, m-1-i), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax2.add_patch(rect)
            if i == step and step < m:
                ax2.text(1.3, m-1-i+0.4, '←', fontsize=20, va='center', color='red')
        
        # Gradient
        ax3.set_xlim(-0.5, 1.5)
        ax3.set_ylim(-0.5, m + 0.5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < m:
            ax3.set_title(f'Step {step+1}/{m}: $\\frac{{\\partial a_{{{step+1}}}}}{{\\partial s}}$', 
                         fontsize=14, fontweight='bold')
        else:
            ax3.set_title('Gradient: $\\frac{\\partial \\mathbf{a}}{\\partial s}$ Complete!', 
                         fontsize=14, fontweight='bold', color='green')
        
        for i in range(m):
            color = 'lightblue' if i <= step and step < m else ('lightblue' if step >= m else 'lightgray')
            rect = patches.Rectangle((0, m-1-i), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax3.add_patch(rect)
            if i == step and step < m:
                ax3.text(-0.3, m-1-i+0.4, '→', fontsize=20, va='center', color='blue')
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, step=widgets.IntSlider(min=0, max=m, value=0, 
                                description='Step:'))

def visualize_vector_wrt_vector_simple(m=3, n=3):
    """Visualization for vector w.r.t. vector (Jacobian)."""
    
    def create_frame(step):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        total_steps = m * n
        
        # Input vector
        ax1.set_xlim(-0.5, 1.5)
        ax1.set_ylim(-0.5, n + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Input: $\\mathbf{{v}} \\in \\mathbb{{R}}^{n}$', fontsize=14)
        
        for j in range(n):
            is_current = (step % n == j and step < total_steps)
            color = 'gold' if is_current else 'lightgray'
            rect = patches.Rectangle((0, n-1-j), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax1.add_patch(rect)
        
        # Output vector
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(-0.5, m + 0.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Output: $\\mathbf{{a}} \\in \\mathbb{{R}}^{m}$', fontsize=14)
        
        for i in range(m):
            is_current = (step // n == i and step < total_steps)
            color = 'lightcoral' if is_current else 'lightgray'
            rect = patches.Rectangle((0, m-1-i), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax2.add_patch(rect)
        
        # Jacobian matrix
        ax3.set_xlim(-0.5, n + 0.5)
        ax3.set_ylim(-0.5, m + 0.5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            row_idx = step // n + 1
            col_idx = step % n + 1
            ax3.set_title(f'Step {step+1}/{total_steps}: $\\frac{{\\partial a_{{{row_idx}}}}}{{\\partial v_{{{col_idx}}}}}$', 
                         fontsize=14, fontweight='bold')
        else:
            ax3.set_title('Jacobian: $\\frac{\\partial \\mathbf{a}}{\\partial \\mathbf{v}}$ Complete!', 
                         fontsize=14, fontweight='bold', color='green')
        
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                color = 'lightblue' if idx <= step and step < total_steps else ('lightblue' if step >= total_steps else 'lightgray')
                rect = patches.Rectangle((j, m-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax3.add_patch(rect)
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, step=widgets.IntSlider(min=0, max=m*n, value=0, 
                                description='Step:'))

def visualize_vector_wrt_matrix_simple(m=2, n=2, p=3):
    """Visualization for vector w.r.t. matrix."""
    
    def create_frame(step):
        fig = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        total_steps = m * n * p
        
        # Input matrix
        ax1.set_xlim(-0.5, p + 0.5)
        ax1.set_ylim(-0.5, n + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Input: $\\mathbf{{X}} \\in \\mathbb{{R}}^{{{n} \\times {p}}}$', fontsize=14)
        
        for i in range(n):
            for j in range(p):
                is_current = ((step // (n * p)) * n * p + i * p + j == step and step < total_steps)
                color = 'gold' if is_current else 'lightgray'
                rect = patches.Rectangle((j, n-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax1.add_patch(rect)
        
        # Output vector
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(-0.5, m + 0.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Output: $\\mathbf{{a}} \\in \\mathbb{{R}}^{m}$', fontsize=14)
        
        for i in range(m):
            is_current = (step // (n * p) == i and step < total_steps)
            color = 'lightcoral' if is_current else 'lightgray'
            rect = patches.Rectangle((0, m-1-i), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax2.add_patch(rect)
        
        # Gradient tensor (showing as slices)
        ax3.set_xlim(-0.5, p*m + m)
        ax3.set_ylim(-0.5, n + 0.5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            i_idx = step // (n * p)
            remainder = step % (n * p)
            j_idx = remainder // p
            k_idx = remainder % p
            ax3.set_title(f'Step {step+1}/{total_steps}: $\\frac{{\\partial a_{{{i_idx+1}}}}}{{\\partial X_{{{j_idx+1}{k_idx+1}}}}}$', 
                         fontsize=13, fontweight='bold')
        else:
            ax3.set_title('Gradient Complete!', fontsize=14, fontweight='bold', color='green')
        
        for i_slice in range(m):
            for j in range(n):
                for k in range(p):
                    x_pos = i_slice * (p + 0.5) + k
                    idx = i_slice * (n * p) + j * p + k
                    color = 'lightblue' if idx <= step and step < total_steps else ('lightblue' if step >= total_steps else 'lightgray')
                    rect = patches.Rectangle((x_pos, n-1-j), 0.9, 0.9, 
                                              linewidth=2, edgecolor='black', 
                                              facecolor=color)
                    ax3.add_patch(rect)
            ax3.text(i_slice * (p + 0.5) + p/2 - 0.5, -0.3, f'$a_{i_slice+1}$', fontsize=10, ha='center')
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, step=widgets.IntSlider(min=0, max=m*n*p, value=0, 
                                description='Step:'))

def visualize_vector_wrt_tensor_simple(m=2, n=2, p=2, q=2):
    """Visualization for vector w.r.t. 3D tensor with 3D perspective."""
    
    total_steps = m * n * p * q
    
    def create_frame(step):
        fig = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        depth_offset = 0.3
        
        # Input tensor with 3D perspective
        ax1.set_xlim(-0.5, p + q*depth_offset + 0.5)
        ax1.set_ylim(-0.5, n + q*depth_offset + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Input: $\\mathcal{{T}} \\in \\mathbb{{R}}^{{{n} \\times {p} \\times {q}}}$', fontsize=14)
        
        for k in range(q):
            x_offset = k * depth_offset
            y_offset = k * depth_offset
            
            for i in range(n):
                for j in range(p):
                    idx_in_tensor = i * p * q + j * q + k
                    is_current = (step % (n * p * q) == idx_in_tensor and step < total_steps)
                    color = 'gold' if is_current else 'lightgray'
                    alpha = 1.0 - (k * 0.15)
                    
                    rect = patches.Rectangle((j + x_offset, n-1-i + y_offset), 0.9, 0.9, 
                                              linewidth=2, edgecolor='black', 
                                              facecolor=color, alpha=alpha,
                                              zorder=q-k)
                    ax1.add_patch(rect)
            
            ax1.text(p/2 - 0.5 + x_offset, -0.5 + y_offset, f'k={k+1}', 
                    fontsize=9, ha='center', zorder=q-k)
        
        # Draw connecting lines
        for i in range(n):
            for j in range(p):
                x_start = j
                y_start = n-1-i
                x_end = j + (q-1) * depth_offset
                y_end = n-1-i + (q-1) * depth_offset
                ax1.plot([x_start, x_end], [y_start, y_end], 
                        'k-', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Output vector
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylim(-0.5, m + 0.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Output: $\\mathbf{{a}} \\in \\mathbb{{R}}^{m}$', fontsize=14)
        
        for i in range(m):
            is_current = (step // (n * p * q) == i and step < total_steps)
            color = 'lightcoral' if is_current else 'lightgray'
            rect = patches.Rectangle((0, m-1-i), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax2.add_patch(rect)
        
        # Gradient (4D tensor - show progress)
        ax3.set_xlim(-0.5, 2)
        ax3.set_ylim(-0.5, 2)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            progress = (step + 1) / total_steps * 100
            ax3.set_title(f'Step {step+1}/{total_steps} ({progress:.1f}%)', 
                         fontsize=14, fontweight='bold')
            ax3.text(1, 1, f'{step+1}/{total_steps}\nentries\ncomputed', 
                    fontsize=16, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax3.set_title('Gradient Complete!', fontsize=14, fontweight='bold', color='green')
            ax3.text(1, 1, f'All {total_steps}\nentries\ncomputed!', 
                    fontsize=16, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, 
                               step=widgets.IntSlider(min=0, max=total_steps, value=0, 
                                                     description='Step:'))

# ==================== MATRIX DERIVATIVES ====================

def visualize_matrix_wrt_scalar_simple(m=2, n=2):
    """Visualization for matrix w.r.t. scalar."""
    
    def create_frame(step):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        
        total_steps = m * n
        
        # Scalar input
        ax1.set_xlim(-0.5, 1.5)
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title('Input: $x \\in \\mathbb{R}$', fontsize=14)
        
        scalar_box = patches.Rectangle((0, 0), 1, 0.8, 
                                        linewidth=2, edgecolor='black', 
                                        facecolor='lightcoral')
        ax1.add_patch(scalar_box)
        ax1.text(0.5, 0.4, 'x', fontsize=16, ha='center', va='center', fontweight='bold')
        
        # Matrix output
        ax2.set_xlim(-0.5, n + 0.5)
        ax2.set_ylim(-0.5, m + 0.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Output: $\\mathbf{{A}} \\in \\mathbb{{R}}^{{{m} \\times {n}}}$', fontsize=14)
        
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                is_current = (idx == step and step < total_steps)
                color = 'gold' if is_current else 'lightgray'
                rect = patches.Rectangle((j, m-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax2.add_patch(rect)
        
        # Gradient matrix
        ax3.set_xlim(-0.5, n + 0.5)
        ax3.set_ylim(-0.5, m + 0.5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            row_idx = step // n + 1
            col_idx = step % n + 1
            ax3.set_title(f'Step {step+1}/{total_steps}: $\\frac{{\\partial A_{{{row_idx}{col_idx}}}}}{{\\partial x}}$', 
                         fontsize=14, fontweight='bold')
        else:
            ax3.set_title('Gradient: $\\frac{\\partial \\mathbf{A}}{\\partial x}$ Complete!', 
                         fontsize=14, fontweight='bold', color='green')
        
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                color = 'lightblue' if idx <= step and step < total_steps else ('lightblue' if step >= total_steps else 'lightgray')
                rect = patches.Rectangle((j, m-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax3.add_patch(rect)
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, step=widgets.IntSlider(min=0, max=m*n, value=0, 
                                description='Step:'))

def visualize_matrix_wrt_vector_simple(m=2, n=2, p=3):
    """Visualization for matrix w.r.t. vector."""
    
    def create_frame(step):
        fig = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        total_steps = m * n * p
        
        # Input vector
        ax1.set_xlim(-0.5, 1.5)
        ax1.set_ylim(-0.5, p + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Input: $\\mathbf{{x}} \\in \\mathbb{{R}}^{p}$', fontsize=14)
        
        for k in range(p):
            is_current = (step % p == k and step < total_steps)
            color = 'gold' if is_current else 'lightgray'
            rect = patches.Rectangle((0, p-1-k), 1, 0.8, 
                                      linewidth=2, edgecolor='black', 
                                      facecolor=color)
            ax1.add_patch(rect)
        
        # Output matrix
        ax2.set_xlim(-0.5, n + 0.5)
        ax2.set_ylim(-0.5, m + 0.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Output: $\\mathbf{{A}} \\in \\mathbb{{R}}^{{{m} \\times {n}}}$', fontsize=14)
        
        for i in range(m):
            for j in range(n):
                idx = i * n + j
                is_current = (step // p == idx and step < total_steps)
                color = 'lightcoral' if is_current else 'lightgray'
                rect = patches.Rectangle((j, m-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax2.add_patch(rect)
        
        # Gradient tensor (3D - showing as slices)
        ax3.set_xlim(-0.5, p*m*n + m*n)
        ax3.set_ylim(-0.5, 1.5)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            A_idx = step // p
            i_idx = A_idx // n + 1
            j_idx = A_idx % n + 1
            k_idx = step % p + 1
            ax3.set_title(f'Step {step+1}/{total_steps}: $\\frac{{\\partial A_{{{i_idx}{j_idx}}}}}{{\\partial x_{{{k_idx}}}}}$', 
                         fontsize=13, fontweight='bold')
        else:
            ax3.set_title('Gradient Complete! (3D Tensor)', fontsize=14, fontweight='bold', color='green')
        
        for A_idx in range(m * n):
            for k in range(p):
                x_pos = A_idx * (p + 0.5) + k
                idx = A_idx * p + k
                color = 'lightblue' if idx <= step and step < total_steps else ('lightblue' if step >= total_steps else 'lightgray')
                rect = patches.Rectangle((x_pos, 0), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax3.add_patch(rect)
            i = A_idx // n + 1
            j = A_idx % n + 1
            ax3.text(A_idx * (p + 0.5) + p/2 - 0.5, -0.3, f'$A_{{{i}{j}}}$', fontsize=9, ha='center')
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, step=widgets.IntSlider(min=0, max=m*n*p, value=0, 
                                description='Step:'))

def visualize_matrix_wrt_matrix_simple(m=2, n=2, p=2, q=2):
    """Visualization for matrix w.r.t. matrix (4D tensor)."""
    
    total_steps = m * n * p * q
    
    def create_frame(step):
        fig = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        # Input matrix
        ax1.set_xlim(-0.5, q + 0.5)
        ax1.set_ylim(-0.5, p + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Input: $\\mathbf{{X}} \\in \\mathbb{{R}}^{{{p} \\times {q}}}$', fontsize=14)
        
        for k in range(p):
            for l in range(q):
                X_idx = k * q + l
                is_current = (step % (p * q) == X_idx and step < total_steps)
                color = 'gold' if is_current else 'lightgray'
                rect = patches.Rectangle((l, p-1-k), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax1.add_patch(rect)
        
        # Output matrix
        ax2.set_xlim(-0.5, n + 0.5)
        ax2.set_ylim(-0.5, m + 0.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Output: $\\mathbf{{A}} \\in \\mathbb{{R}}^{{{m} \\times {n}}}$', fontsize=14)
        
        for i in range(m):
            for j in range(n):
                A_idx = i * n + j
                is_current = (step // (p * q) == A_idx and step < total_steps)
                color = 'lightcoral' if is_current else 'lightgray'
                rect = patches.Rectangle((j, m-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax2.add_patch(rect)
        
        # Gradient (4D tensor - show progress)
        ax3.set_xlim(-0.5, 2)
        ax3.set_ylim(-0.5, 2)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            progress = (step + 1) / total_steps * 100
            A_idx = step // (p * q)
            i = A_idx // n + 1
            j = A_idx % n + 1
            X_idx = step % (p * q)
            k = X_idx // q + 1
            l = X_idx % q + 1
            ax3.set_title(f'Step {step+1}/{total_steps} ({progress:.1f}%)', 
                         fontsize=14, fontweight='bold')
            ax3.text(1, 1, f'$\\frac{{\\partial A_{{{i}{j}}}}}{{\\partial X_{{{k}{l}}}}}$\n\n{step+1}/{total_steps} entries', 
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax3.set_title('Gradient Complete!', fontsize=14, fontweight='bold', color='green')
            ax3.text(1, 1, f'4D Tensor\n$({m},{n},{p},{q})$\n\nAll {total_steps}\nentries computed!', 
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, 
                               step=widgets.IntSlider(min=0, max=total_steps, value=0, 
                                                     description='Step:'))

def visualize_matrix_wrt_tensor_simple(m=2, n=2, p=2, q=2, r=2):
    """Visualization for matrix w.r.t. 3D tensor with 3D perspective."""
    
    total_steps = m * n * p * q * r
    
    def create_frame(step):
        fig = plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        
        depth_offset = 0.3
        
        # Input tensor with 3D perspective
        ax1.set_xlim(-0.5, q + r*depth_offset + 0.5)
        ax1.set_ylim(-0.5, p + r*depth_offset + 0.5)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f'Input: $\\mathcal{{T}} \\in \\mathbb{{R}}^{{{p} \\times {q} \\times {r}}}$', fontsize=14)
        
        for s in range(r):
            x_offset = s * depth_offset
            y_offset = s * depth_offset
            
            for k in range(p):
                for l in range(q):
                    T_idx = k * q * r + l * r + s
                    is_current = (step % (p * q * r) == T_idx and step < total_steps)
                    color = 'gold' if is_current else 'lightgray'
                    alpha = 1.0 - (s * 0.15)
                    
                    rect = patches.Rectangle((l + x_offset, p-1-k + y_offset), 0.9, 0.9, 
                                              linewidth=2, edgecolor='black', 
                                              facecolor=color, alpha=alpha,
                                              zorder=r-s)
                    ax1.add_patch(rect)
            
            ax1.text(q/2 - 0.5 + x_offset, -0.5 + y_offset, f's={s+1}', 
                    fontsize=9, ha='center', zorder=r-s)
        
        # Draw connecting lines
        for k in range(p):
            for l in range(q):
                x_start = l
                y_start = p-1-k
                x_end = l + (r-1) * depth_offset
                y_end = p-1-k + (r-1) * depth_offset
                ax1.plot([x_start, x_end], [y_start, y_end], 
                        'k-', linewidth=0.5, alpha=0.3, zorder=0)
        
        # Output matrix
        ax2.set_xlim(-0.5, n + 0.5)
        ax2.set_ylim(-0.5, m + 0.5)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Output: $\\mathbf{{A}} \\in \\mathbb{{R}}^{{{m} \\times {n}}}$', fontsize=14)
        
        for i in range(m):
            for j in range(n):
                A_idx = i * n + j
                is_current = (step // (p * q * r) == A_idx and step < total_steps)
                color = 'lightcoral' if is_current else 'lightgray'
                rect = patches.Rectangle((j, m-1-i), 0.9, 0.9, 
                                          linewidth=2, edgecolor='black', 
                                          facecolor=color)
                ax2.add_patch(rect)
        
        # Gradient (5D tensor - show progress)
        ax3.set_xlim(-0.5, 2)
        ax3.set_ylim(-0.5, 2)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        if step < total_steps:
            progress = (step + 1) / total_steps * 100
            ax3.set_title(f'Step {step+1}/{total_steps} ({progress:.1f}%)', 
                         fontsize=14, fontweight='bold')
            ax3.text(1, 1, f'5D Tensor\n$({m},{n},{p},{q},{r})$\n\n{step+1}/{total_steps}\nentries', 
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        else:
            ax3.set_title('Gradient Complete!', fontsize=14, fontweight='bold', color='green')
            ax3.text(1, 1, f'5D Tensor\n$({m},{n},{p},{q},{r})$\n\nAll {total_steps}\nentries computed!', 
                    fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def show_step(step):
        fig = create_frame(step)
        plt.show()
        plt.close()
    
    interact = widgets.interact(show_step, 
                               step=widgets.IntSlider(min=0, max=total_steps, value=0, 
                                                     description='Step:'))