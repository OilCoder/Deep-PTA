"""Optional LLM narrator that explains a diagnosis in interpreter language.

Given a diagnosis from :func:`deep_pta.app.inference.diagnose`, produce an
engineer-style narrative (e.g., "radial stabilization followed by a doubling of the
derivative -> sealing fault at ~300 ft"). Uses the Anthropic SDK with the most capable
Claude model; falls back to a deterministic template when no API key is configured.
"""

from __future__ import annotations

import os
from typing import cast

_SYSTEM = (
    "You are a senior well-test interpreter. Given a pressure transient diagnosis, "
    "explain it in two or three sentences using flow-regime language (radial "
    "stabilization, linear/bilinear flow, derivative doubling for a sealing fault, "
    "a drop for constant pressure, unit slope for a closed boundary). Be concise and "
    "physical."
)


def _template_narration(diagnosis: dict[str, object]) -> str:
    params = diagnosis.get("params", {})
    assert isinstance(params, dict)
    param_str = ", ".join(f"{k}={v:.3g}" for k, v in params.items())
    return (
        f"The derivative is read as a {diagnosis['reservoir']} reservoir with a "
        f"{diagnosis['boundary']} boundary (confidence "
        f"{cast(float, diagnosis['confidence']):.0%}). Estimated parameters: {param_str}."
    )


def narrate(diagnosis: dict[str, object], model: str = "claude-opus-4-8") -> str:
    """Narrate a diagnosis in interpreter language.

    Parameters
    ----------
    diagnosis : dict
        Output of :func:`deep_pta.app.inference.diagnose`.
    model : str, optional
        Anthropic model id, by default ``"claude-opus-4-8"``.

    Returns
    -------
    str
        A short narrative. Falls back to a template if the SDK or API key is absent.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _template_narration(diagnosis)
    try:
        import anthropic
    except ImportError:
        return _template_narration(diagnosis)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=200,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _template_narration(diagnosis)}],
    )
    parts = [block.text for block in message.content if block.type == "text"]
    return "\n".join(parts) if parts else _template_narration(diagnosis)
