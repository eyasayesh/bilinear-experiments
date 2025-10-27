from models.bilinear_adder import BilinearAdder
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from configs import bilinear_adder_config_b as config
from datasets import get_mod_add_dataloaders

model = BilinearAdder(d_input=config.P*2,
                      d_hidden=config.d_hidden,
                      P=config.P)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("checkpoints/bilinear_adder_b.pth", map_location=device))

#evaluate accuracy
model.to(device)
model.eval()
correct = 0
total = 0

train_loader, test_loader = get_mod_add_dataloaders(modulus = config.P)
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test Accuracy of the model on the {total} test images: {100 * correct / total} %')

def compute_spectrum(vectors):
    """Compute normalized FFT magnitudes for a batch of eigenvectors (rows)."""
    fft_vals = np.fft.fft(vectors, axis=1)
    mag = np.abs(fft_vals[:, :vectors.shape[1] // 2])  # keep positive freqs
    mag /= (np.max(mag, axis=1, keepdims=True) + 1e-8)
    return mag

def cosine(x, A, f, phi, C):
    return A * np.cos(2 * np.pi * f * x + phi) + C


model.cpu()

fig_dir = "./figures/bilinear_adder_b/"

#random digits between 1 and 112
digits = np.random.randint(1, 113, size=17)

for digit in digits:
    analysis = model.analyze_mod_digit(digit)
    interaction_matrix = analysis['interaction_matrix']
    
    eigenvalues = analysis['eigenvalues']
    eigenvectors = analysis['eigenvectors']

    print("total number of eigen vectors is", eigenvectors.shape)

    digit_dir = f'{fig_dir}/{digit}'
    os.makedirs(digit_dir, exist_ok=True)

    figure = plt.figure(figsize=(12, 5))
    plt.imshow(interaction_matrix, cmap='bwr')
    plt.colorbar()
    plt.title(f'Interaction Matrix for Digit {digit}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Embedding Dimension')
    plt.axis('off')
    plt.savefig(f'{digit_dir}/interaction_matrix_{digit}.png')
    plt.close(figure)

    n = 25
    min_opacity = 0.25

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ax = axes[0, 0]
    ax.plot(np.arange(1, n+1), eigenvalues[:n], color='red', linewidth=2.5, zorder=2)
    ax.scatter(np.arange(1, 5), eigenvalues[:4], color='red', s=60, zorder=3)
    for y in eigenvalues[:4]:
        ax.axhline(y=y, color='grey', linestyle='-', linewidth=2, alpha=0.4, zorder=1)
    ax.set_yticks(eigenvalues[:4])
    ax.set_yticklabels([f"{y:.2f}" for y in eigenvalues[:4]], color='grey', fontsize=9)
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax.set_xlim(0.5, n+0.5)
    ax.set_title(f"Top {n} Positive Eigenvalues", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    top_effective_contribution = np.sum(eigenvalues[:n])/np.sum(np.abs(eigenvalues))
    ax.set_xlabel(f"Effective contribution = {top_effective_contribution:.3f}", fontsize=10)

    
    ax_bottom = axes[1, 0]
    bottom_vals = eigenvalues[-n:][::-1]  
    ax_bottom.plot(np.arange(1, n+1), bottom_vals, color='blue', linewidth=2.5, zorder=2)
    highlight_vals = eigenvalues[-4:][::-1]
    ax_bottom.scatter(np.arange(1, 5), highlight_vals, color='blue', s=70, zorder=3)
    for y in highlight_vals:
        ax_bottom.axhline(y=y, color='grey', linestyle='-', linewidth=2, alpha=0.4, zorder=1)
    ax_bottom.set_yticks(highlight_vals,)
    ax_bottom.set_yticklabels([f"{y:.2f}" for y in highlight_vals], color='grey', fontsize=9)
    ax_bottom.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax_bottom.set_xlim(0.5, n+0.5)
    ax_bottom.set_title(f"Top {n} Negative Eigenvalues", fontsize=12)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)

    bottom_effective_contribution = -np.sum(eigenvalues[-n:])/np.sum(np.abs(eigenvalues))
    ax_bottom.set_xlabel(f"Effective contribution = {bottom_effective_contribution:.3f}", fontsize=10)

    # Visualize eigenvectors as images
    # top 50 eigen vectors
    eigenvec = eigenvectors[:, :n].T
    eigenvec_a = eigenvec[:, :config.P]
    eigenvec_b = eigenvec[:, config.P:]

    eigenvals = eigenvalues[:n]
    eigenvals_norm = (eigenvals - eigenvals.min()) / (eigenvals.max() - eigenvals.min() + 1e-8)
    eigenvals_norm = min_opacity + 0.5 * eigenvals_norm

    # Scale each row of eigenvec by its eigenvalue
    scaled_a = eigenvec_a * eigenvals_norm[:, None]  # (50, d)
    scaled_b = eigenvec_b * eigenvals_norm[:, None]  # (50, d)

    # Compute frequency-domain magnitudes
    spectrum_a = compute_spectrum(scaled_a)
    spectrum_b = compute_spectrum(scaled_b)


    # --- Bottom 50 eigenvectors ---
    eigenvec_bottom = eigenvectors[:, -n:].T  # last 50 eigenvectors
    eigenvec_bottom_a = eigenvec_bottom[:, :config.P]
    eigenvec_bottom_b = eigenvec_bottom[:, config.P:]

    eigenvals_bottom = np.abs(eigenvalues[-n:])
    # Normalize and possibly reverse for better visual order
    eigenvals_bottom_norm = (eigenvals_bottom - eigenvals_bottom.min()) / (eigenvals_bottom.max() - eigenvals_bottom.min() + 1e-8)
    eigenvals_bottom_norm = min_opacity + 0.5 * eigenvals_norm

    # Reverse both eigenvalues and vectors if you want "most negative on top"
    eigenvals_bottom_norm = eigenvals_bottom_norm[::-1]
    eigenvec_bottom_a = eigenvec_bottom_a[::-1]
    eigenvec_bottom_b = eigenvec_bottom_b[::-1]


    scaled_bottom_a = eigenvec_bottom_a * eigenvals_bottom_norm[:, None]
    scaled_bottom_b = eigenvec_bottom_b * eigenvals_bottom_norm[:, None]

    spectrum_bottom_a = compute_spectrum(scaled_bottom_a)
    spectrum_bottom_b = compute_spectrum(scaled_bottom_b)

    ax = axes[0, 1]
    ax.imshow(scaled_a, cmap='bwr', aspect='auto')
    ax.set_title(f'Top {n} eigenvectors (input a)', fontsize=12)
    ax.axis('off')
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Eigenvector Index")

    ax = axes[0, 2]
    ax.imshow(scaled_b, cmap='bwr', aspect='auto')
    ax.set_title(f'Top {n} eigenvectors (input b)', fontsize=12)
    ax.axis('off')
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Eigenvector Index")

    # Bottom row: most negative eigenvalues
    ax = axes[1, 1]
    ax.imshow(scaled_bottom_a, cmap='bwr', aspect='auto')
    ax.set_title(f'Bottom {n} eigenvectors (input a)', fontsize=12)
    ax.axis('off')
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Eigenvector Index")

    ax = axes[1, 2]
    ax.imshow(scaled_bottom_b, cmap='bwr', aspect='auto')
    ax.set_title(f'Bottom {n} eigenvectors (input b)', fontsize=12)
    ax.axis('off')
    ax.set_xlabel("Input Feature")
    ax.set_ylabel("Eigenvector Index")

    plt.colorbar(plt.cm.ScalarMappable(cmap='bwr'), ax=axes.ravel().tolist(), shrink=0.6)
    plt.suptitle(f'Bilinear Modular Addition - Output {digit} Analysis - Effective contibution of {2*n} eigenvalues is {bottom_effective_contribution + top_effective_contribution:.3f}', fontsize=16)
    plt.savefig(f'{digit_dir}/digit_{digit}_analysis.png')
    plt.close(fig)

    print("The effective rank for digit", digit, "is:", analysis['effective_rank'])
    print()


    # ----- Plot -----
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    ax = axes[0, 0]
    im = ax.imshow(spectrum_a, cmap='magma', aspect='auto')
    ax.set_title(f'Top {n} eigenvectors (Input A)', fontsize=12)
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Eigenvector Index")
    ax.axis('on')

    ax = axes[0, 1]
    ax.imshow(spectrum_b, cmap='magma', aspect='auto')
    ax.set_title(f'Top {n} eigenvectors (Input B)', fontsize=12)
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Eigenvector Index")
    ax.axis('on')

    ax = axes[1, 0]
    ax.imshow(spectrum_bottom_a, cmap='magma', aspect='auto')
    ax.set_title(f'Bottom {n} eigenvectors (Input A)', fontsize=12)
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Eigenvector Index")
    ax.axis('on')

    ax = axes[1, 1]
    ax.imshow(spectrum_bottom_b, cmap='magma', aspect='auto')
    ax.set_title(f'Bottom {n} eigenvectors (Input B)', fontsize=12)
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("Eigenvector Index")
    ax.axis('on')

    # Shared colorbar across all subplots
    plt.colorbar(plt.cm.ScalarMappable(cmap='magma'), ax=axes.ravel().tolist(), shrink=0.6)
    plt.suptitle(
        f"Bilinear Modular Addition - Output {digit} - Top {n} Frequency Analysis",
        fontsize=16
    )
    plt.savefig(f"{digit_dir}/digit_{digit}_frequency_analysis.png")
    plt.close(fig)

    print(f"✅ Frequency domain visualization saved for digit {digit}")



    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Helper function for normalization
    def normalize(v):
        v = v - np.mean(v)
        v = v / (np.max(np.abs(v)) + 1e-8)
        return v

    # ======== TOP 3 EIGENVECTORS - INPUT A ========
    ax = axes[0, 0]
    offsets = [0, 2, 4]  # vertical offsets
    for i, (style, alpha, offset) in enumerate(zip(['-', '--', '-.'], [1.0, 0.7, 0.5], offsets)):
        vec = normalize(eigenvec_a[i, :])
        ax.plot(np.arange(1, int(len(eigenvalues)/2)+1),
                vec + offset, linestyle=style, color='red', linewidth=2.5, alpha=alpha,
                label=f"Top {i+1}")
    ax.set_title('Top 3 Eigenvectors - Input A', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Normalized Eigenvector")
    ax.set_yticks([])

    # ======== TOP 3 EIGENVECTORS - INPUT B ========
    ax = axes[0, 1]
    for i, (style, alpha, offset) in enumerate(zip(['-', '--', '-.'], [1.0, 0.7, 0.5], offsets)):
        vec = normalize(eigenvec_b[i, :])
        ax.plot(np.arange(1, int(len(eigenvalues)/2)+1),
                vec + offset, linestyle=style, color='red', linewidth=2.5, alpha=alpha,
                label=f"Top {i+1}")
    ax.set_title('Top 3 Eigenvectors - Input B', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Normalized Eigenvector")
    ax.set_yticks([])

    # ======== BOTTOM 3 EIGENVECTORS - INPUT A ========
    ax = axes[1, 0]
    for i, (style, alpha, offset) in enumerate(zip(['-', '--', '-.'], [1.0, 0.7, 0.5], offsets)):
        vec = normalize(eigenvec_bottom_a[-(i+1), :])
        ax.plot(np.arange(1, int(len(eigenvalues)/2)+1),
                vec + offset, linestyle=style, color='blue', linewidth=2.5, alpha=alpha,
                label=f"Bottom {i+1}")
    ax.set_title('Bottom 3 Eigenvectors - Input A', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Normalized Eigenvector")
    ax.set_yticks([])

    # ======== BOTTOM 3 EIGENVECTORS - INPUT B ========
    ax = axes[1, 1]
    for i, (style, alpha, offset) in enumerate(zip(['-', '--', '-.'], [1.0, 0.7, 0.5], offsets)):
        vec = normalize(eigenvec_bottom_b[-(i+1), :])
        ax.plot(np.arange(1, int(len(eigenvalues)/2)+1),
                vec + offset, linestyle=style, color='blue', linewidth=2.5, alpha=alpha,
                label=f"Bottom {i+1}")
    ax.set_title('Bottom 3 Eigenvectors - Input B', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlabel("Input Dimension")
    ax.set_ylabel("Normalized Eigenvector")
    ax.set_yticks([])

    # ======== FIGURE SETTINGS ========
    plt.suptitle(f'Bilinear Modular Addition - Output {digit} - Top and Bottom 3 Eigenvectors', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{digit_dir}/digit_{digit}_top_bottom_3_eigenvectors.png')
    plt.close(fig)





    

