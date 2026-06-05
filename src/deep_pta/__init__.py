"""Deep PTA — neural interpreter of pressure transient tests.

Subpackages
-----------
engine
    Analytical PTA engine (Laplace-space solutions + Stehfest inversion).
data
    Synthetic dataset generator (sampling, realism, Bourdet derivative, representation).
models
    Neural architectures (ResNet-1D, hand-built PatchTST) and losses.
train
    Training loops, hyperparameter optimization, and model comparison.
app
    Interactive Gradio app and optional LLM narrator.
"""

__version__ = "0.1.0"
