"""Gradio app: upload a ``(t, p)`` CSV and get a PTA diagnosis with an overlaid fit.

The app reuses the full pipeline (Bourdet derivative, representation, model, engine) and
adds no new physics. ``build_demo`` returns the Gradio ``Blocks`` so it can be imported
and smoke-tested without launching a server; ``main`` launches it.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from deep_pta.app.inference import diagnose, load_model, reconstruct_derivative
from deep_pta.data.real_cases import load_real_case


def _format_diagnosis(result: dict[str, object]) -> str:
    params = cast("dict[str, float]", result["params"])
    confidence = cast(float, result["confidence"])
    lines = [
        f"Reservoir model : {result['reservoir']}",
        f"Boundary        : {result['boundary']}",
        f"Confidence      : {confidence:.0%}",
        "",
        "Estimated parameters:",
    ]
    for key, value in params.items():
        lines.append(f"  {key:8s} = {value:.4g}")
    if confidence < 0.4:
        lines.append("")
        lines.append("Low confidence: the test may fall outside the known taxonomy.")
    return "\n".join(lines)


def _diagnose_csv(file_obj: Any, checkpoint: str) -> tuple[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, dp = load_real_case(file_obj.name if hasattr(file_obj, "name") else file_obj)
    model = load_model(checkpoint)
    result = diagnose(model, t, dp)
    x = result["x"]
    assert isinstance(x, np.ndarray)
    recon = reconstruct_derivative(
        cast(int, result["reservoir_class"]),
        cast(int, result["boundary_class"]),
        cast("dict[str, float]", result["params"]),
    )

    grid = np.arange(x.shape[1])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, x[1], label="data derivative (normalized)", lw=2)
    ax.plot(grid, recon, "--", label="fitted model derivative", lw=2)
    ax.set_xlabel("log-time grid index")
    ax.set_ylabel("standardized log derivative")
    ax.set_title("Bourdet derivative: data vs fitted model")
    ax.legend()
    fig.tight_layout()
    return _format_diagnosis(result), fig


def build_demo(checkpoint: str = "models/cnn_baseline.pt") -> Any:
    """Build the Gradio Blocks app.

    Parameters
    ----------
    checkpoint : str, optional
        Path to the trained checkpoint, by default ``models/cnn_baseline.pt``.

    Returns
    -------
    gradio.Blocks
        The (unlaunched) app.
    """
    import gradio as gr

    with gr.Blocks(title="Deep PTA") as demo:
        gr.Markdown("# Deep PTA\nUpload a `(t, p)` CSV to diagnose the pressure test.")
        with gr.Row():
            file_in = gr.File(label="Pressure CSV (t, p)", file_types=[".csv"])
            text_out = gr.Textbox(label="Diagnosis", lines=12)
        plot_out = gr.Plot(label="Derivative overlay")
        btn = gr.Button("Diagnose")
        btn.click(
            fn=lambda f: _diagnose_csv(f, checkpoint),
            inputs=file_in,
            outputs=[text_out, plot_out],
        )
    return demo


def main() -> None:
    """Launch the Gradio app."""
    build_demo().launch()


if __name__ == "__main__":
    main()
