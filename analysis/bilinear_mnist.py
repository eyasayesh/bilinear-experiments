from models.bilinear_classifier import BilinearClassifier
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

model = BilinearClassifier(d_embed=512,
                            d_hidden=512,
                            d_out=10,
                            input_noise=0.5)    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("checkpoints/bilinear_mnist.pth", map_location=device))

model.cpu()

fig_dir = "./figures/bilinear_mnist/"

for digit in range(10):
    analysis = model.analyze_digit(digit)
    interaction_matrix = analysis['interaction_matrix']
    
    eigenvalues = analysis['eigenvalues']
    eigenvectors = analysis['eigenvectors']

    digit_dir = f'{fig_dir}/{digit}'
    os.makedirs(digit_dir, exist_ok=True)

    figure = plt.figure(figsize=(12, 5))
    plt.imshow(interaction_matrix, cmap='RdBu')
    plt.colorbar()
    plt.title(f'Interaction Matrix for Digit {digit}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Embedding Dimension')
    plt.axis('off')
    plt.savefig(f'{digit_dir}/interaction_matrix_{digit}.png')
    plt.close(figure)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    ax = axes[0, 0]
    ax.plot(np.arange(1, 21), eigenvalues[:20], color='blue', linewidth=2.5, zorder=2)
    ax.scatter(np.arange(1, 5), eigenvalues[:4], color='blue', s=60, zorder=3)
    for y in eigenvalues[:4]:
        ax.axhline(y=y, color='grey', linestyle='-', linewidth=2, alpha=0.4, zorder=1)
    ax.set_yticks(eigenvalues[:4])
    ax.set_yticklabels([f"{y:.2f}" for y in eigenvalues[:4]], color='grey', fontsize=9)
    ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax.set_xlim(0.5, 20.5)
    ax.set_title("Positive Eigenvalues", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax_bottom = axes[1, 0]
    bottom_vals = eigenvalues[-20:]   # the smallest 20 eigenvalues
    ax_bottom.plot(np.arange(1, 21), bottom_vals, color='red', linewidth=2.5, zorder=2)
    ax_bottom.scatter(np.arange(17, 21), eigenvalues[-4:], color='red', s=70, zorder=3)
    for y in eigenvalues[-4:]:
        ax_bottom.axhline(y=y, color='grey', linestyle='-', linewidth=2, alpha=0.4, zorder=1)
    ax_bottom.set_yticks(eigenvalues[-4:],)
    ax_bottom.set_yticklabels([f"{y:.2f}" for y in eigenvalues[-4:]], color='grey', fontsize=9)
    ax_bottom.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)
    ax_bottom.set_xlim(0.5, 20.5)
    ax_bottom.set_title("Negative Eigenvalues", fontsize=12)
    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)

    E = model.embed[1].weight.data.numpy() 
    for i in range(1,5):
        ax = axes[0, i]
        eigenvec = eigenvectors[:, i-1]
        img = E.T @ eigenvec
        img = img.reshape(28, 28)
        ax.imshow(img, cmap='RdBu')
        ax.set_title(f'Eigenvector {i}', fontsize=12)
        ax.axis('off')
    
        ax_bottom = axes[1, i]
        eigenvec = eigenvectors[:, -i]
        img = E.T @ eigenvec
        img = img.reshape(28, 28)
        ax_bottom.imshow(img, cmap='RdBu')
        ax_bottom.set_title(f'Eigenvector {-i}', fontsize=12)
        ax_bottom.axis('off')

    plt.colorbar(plt.cm.ScalarMappable(cmap='RdBu'), ax=axes.ravel().tolist(), shrink=0.6)
    plt.suptitle(f'Bilinear Classifier - Digit {digit} Analysis', fontsize=16)
    plt.savefig(f'{digit_dir}/digit_{digit}_analysis.png')
    plt.close(fig)

    print("The effective rank for digit", digit, "is:", analysis['effective_rank'])
    print()



    

