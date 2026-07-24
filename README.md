# Multi-task Optical Classification with Single-layer Differential D2NN

Code and trained models reproducing all results of the paper.

## Environment
Python 3.9.24; install deps:
```
pip install -r requirements.txt
```
Datasets (MNIST / Fashion-MNIST / EMNIST-Letters / USPS) download automatically via torchvision on first run.

## Reproduce
Open `d2nn_full_experiments.ipynb`, run `Kernel -> Restart & Run All`.
- Trained models: `saved_models/<config>_seed<seed>.pt`
- To reproduce Table 1 without retraining, load a checkpoint:
```python
ck = torch.load('saved_models/B_seed42.pt', weights_only=False)
```
- Figures regenerate into `paper_figures/` from the saved models and `all_results.json`.

## Protocol
- Fixed 30 epochs, Adam lr=0.005, StepLR(5,0.5), batch 32/task, no early stopping, no validation-based selection.
- Reported accuracy = blind test accuracy of the final-epoch model on the standard test sets.
- Multi-task training cycles the smaller datasets so each task gets equal gradient steps per epoch.
- T=0.1, energy weight lambda_E=1e-3, epsilon=1e-6; single layer = 40000 parameters.
- Wavelength pairs (nm): MNIST 1064/532, Fashion 780/633, EMNIST 850/450, USPS 980/488.
- EMNIST uses the 10-class A-J subset to share the 10-window detector layout.

## Files
- `all_results.json` — accuracies & training histories, all configs & seeds
- `robustness_results.json` — noise / fabrication / gain / wavelength scans
- `height_profile_B_seed42_um.npy` — optimized DOE height map (micron)
- `saved_samples/` — detector-plane intensities for Fig. 2

## License
Code: MIT. Datasets belong to their respective owners.
