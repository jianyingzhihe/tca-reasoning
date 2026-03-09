import html
import torch
from IPython.display import HTML, display

# ---------- helpers ----------

def print_topk_token_predictions(sentence, original_logits, new_logits, tokenizer, k: int = 5):
    """
    Print top-k next-token predictions in plain text (terminal-friendly).
    Works for both text-only and VLM logits.
    """

    def topk(logits):
        # Handle shape: [B,T,V] or [T,V] or [V]
        x = logits
        if x.ndim == 3:
            x = x.squeeze(0)
        if x.ndim == 2:
            x = x[-1]  # last position
        probs = torch.softmax(x, dim=-1)
        vals, idxs = torch.topk(probs, k)
        return [(tokenizer.decode([i]), float(v)) for i, v in zip(idxs, vals)]

    orig = topk(original_logits)
    new  = topk(new_logits)

    print(f"\n🧾 Input: {sentence}\n")

    print("🔹 Original top-k predictions:")
    for tok, prob in orig:
        print(f"   {tok!r:>15s}: {prob:.3f}")

    print("\n🔸 New (after intervention):")
    for tok, prob in new:
        print(f"   {tok!r:>15s}: {prob:.3f}")

    print("-" * 60)



def _normalize_logits_to_TV(logits: torch.Tensor) -> torch.Tensor:
    """
    Normalize logits to shape [T, V].
    Accepts [V], [T, V], or [B, T, V] (B==1).
    """
    x = logits
    if x.ndim == 1:
        # [V] -> pretend T=1
        x = x.unsqueeze(0)
    elif x.ndim == 3:
        # [B, T, V] -> squeeze batch
        assert x.size(0) == 1, f"Expected batch=1, got {x.size()}"
        x = x.squeeze(0)
    elif x.ndim != 2:
        raise ValueError(f"Unsupported logits shape {tuple(x.shape)}; need [V], [T,V], or [1,T,V]")
    return x  # [T, V]


def _detect_image_token_ids(tokenizer) -> list[int]:
    """
    Best-effort detection of image special token IDs for a VLM tokenizer.
    Returns a possibly empty list if no markers are present.
    """
    ids = []
    vocab = tokenizer.get_vocab()
    candidates = [
        "<start_of_image>", "<image_start>", "<image>", "<image_token>", "<image_soft_token>"
    ]
    for t in candidates:
        if t in vocab:
            ids.append(tokenizer.convert_tokens_to_ids(t))
    # model.cfg.image_token_id might exist in your class; if so, you can add it too.
    return list(dict.fromkeys(ids))


def _last_text_position(input_ids: torch.Tensor, tokenizer, image_token_ids: list[int] | None = None) -> int:
    """
    Pick the last text token position (ignoring image marker tokens).
    input_ids: [T] or [1, T]
    """
    ids = input_ids.squeeze(0) if input_ids.ndim == 2 else input_ids
    if image_token_ids is None:
        image_token_ids = _detect_image_token_ids(tokenizer)

    if image_token_ids:
        img_mask = torch.zeros_like(ids, dtype=torch.bool)
        for iid in image_token_ids:
            img_mask |= (ids == iid)
        text_pos = (~img_mask).nonzero(as_tuple=True)[0]
        if len(text_pos) > 0:
            return int(text_pos[-1].item())

    # fallback: last token
    return int(ids.numel() - 1)


def _topk_at_position(logits_TV: torch.Tensor, pos: int, tokenizer, k: int = 5):
    """
    Compute top-k at a given position in [T, V] logits.
    """
    pos = pos if pos >= 0 else (logits_TV.size(0) - 1)
    probs = torch.softmax(logits_TV[pos], dim=-1)
    topk = torch.topk(probs, k)
    return [(tokenizer.decode([topk.indices[i]]), topk.values[i].item()) for i in range(k)]


# ---------- main display: works for text-only *and* VLM ----------

def display_topk_token_predictions(
    sentence_or_prompt,
    original_logits: torch.Tensor,
    new_logits: torch.Tensor | None,
    tokenizer,
    k: int = 5,
    *,
    # If you pass input_ids, we'll auto-select the last *text* token (ignoring image tokens)
    input_ids: torch.Tensor | None = None,
    # You can override where to look (position index in the sequence)
    position: int | None = None,
    # If you already know image token ids, pass them; otherwise we try to detect
    image_token_ids: list[int] | None = None,
):
    """
    Visualize top-k next-token predictions at a chosen position.
    - For text-only, you can omit `input_ids` and `position` (defaults to last token).
    - For VLM, pass `input_ids` from the model's processor batch so we can pick the
      last text token by default (ignoring image markers).
    - Set `position=` explicitly to override either case.
    """

    # Normalize logits to [T, V]
    orig_TV = _normalize_logits_to_TV(original_logits)
    new_TV  = _normalize_logits_to_TV(new_logits) if new_logits is not None else None

    # Decide which position to visualize
    if position is not None:
        pos = position
    elif input_ids is not None:
        pos = _last_text_position(input_ids, tokenizer, image_token_ids=image_token_ids)
    else:
        pos = orig_TV.size(0) - 1  # last token by default

    original_tokens = _topk_at_position(orig_TV, pos, tokenizer, k)
    new_tokens = _topk_at_position(new_TV, pos, tokenizer, k) if new_TV is not None else []

    # ---- HTML (your original styling, lightly kept) ----
    html_out = f"""
    <style>
    .token-viz {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        margin-bottom: 10px; max-width: 700px;
    }}
    .token-viz .header {{
        font-weight: bold; font-size: 14px; margin-bottom: 3px; padding: 4px 6px; border-radius: 3px; color: white; display: inline-block;
    }}
    .token-viz .sentence {{
        background-color: rgba(200, 200, 200, 0.2); padding: 4px 6px; border-radius: 3px; border: 1px solid rgba(100, 100, 100, 0.5);
        font-family: monospace; margin-bottom: 8px; font-weight: 500; font-size: 14px;
    }}
    .token-viz table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px; font-size: 13px; table-layout: fixed; }}
    .token-viz th {{ text-align: left; padding: 4px 6px; font-weight: bold; border: 1px solid rgba(150, 150, 150, 0.5); background-color: rgba(200, 200, 200, 0.3); }}
    .token-viz td {{ padding: 3px 6px; border: 1px solid rgba(150, 150, 150, 0.5); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .token-viz .token-col {{ width: 20%; }}
    .token-viz .prob-col {{ width: 15%; }}
    .token-viz .dist-col {{ width: 65%; }}
    .token-viz .monospace {{ font-family: monospace; }}
    .token-viz .bar-container {{ display: flex; align-items: center; }}
    .token-viz .bar {{ height: 12px; min-width: 2px; }}
    .token-viz .bar-text {{ margin-left: 6px; font-weight: 500; font-size: 12px; }}
    .token-viz .even-row {{ background-color: rgba(240, 240, 240, 0.1); }}
    .token-viz .odd-row  {{ background-color: rgba(255, 255, 255, 0.1); }}
    </style>

    <div class="token-viz">
        <div class="header" style="background-color: #555555;">Input:</div>
        <div class="sentence">{html.escape(str(sentence_or_prompt))}</div>

        <div class="header" style="background-color: #2471A3;">Original Top {k} @ pos {pos}</div>
        <table>
            <thead>
                <tr><th class="token-col">Token</th><th class="prob-col" style="text-align: right;">Probability</th><th class="dist-col">Distribution</th></tr>
            </thead>
            <tbody>
    """

    # Scaling against a shared max for both columns (if new_logits provided)
    max_prob = max([p for _, p in original_tokens] + ([p for _, p in new_tokens] if new_tokens else [1e-12]))

    def _rows(tokens, color):
        rows = ""
        for i, (token, prob) in enumerate(tokens):
            bar_width = int(prob / max_prob * 100)
            row_class = "even-row" if i % 2 == 0 else "odd-row"
            rows += f"""
                <tr class="{row_class}">
                    <td class="monospace token-col" title="{html.escape(token)}">{html.escape(token)}</td>
                    <td class="prob-col" style="text-align: right;">{prob:.3f}</td>
                    <td class="dist-col">
                        <div class="bar-container">
                            <div class="bar" style="background-color: {color}; width: {bar_width}%;"></div>
                            <span class="bar-text">{prob * 100:.1f}%</span>
                        </div>
                    </td>
                </tr>
            """
        return rows

    html_out += _rows(original_tokens, "#2471A3")

    html_out += """
            </tbody>
        </table>
    """

    if new_TV is not None:
        html_out += f"""
        <div class="header" style="background-color: #27AE60;">New Top {k} @ pos {pos}</div>
        <table>
            <thead>
                <tr><th class="token-col">Token</th><th class="prob-col" style="text-align: right;">Probability</th><th class="dist-col">Distribution</th></tr>
            </thead>
            <tbody>
        """
        html_out += _rows(new_tokens, "#27AE60")
        html_out += """
            </tbody>
        </table>
        """

    html_out += "</div>"
    display(HTML(html_out))


# --------- generations comparison stays the same ---------

def display_generations_comparison(original_text, pre_intervention_gens, post_intervention_gens):
    escaped_original = html.escape(original_text)
    html_content = """
    <style>
    .generations-viz { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; margin-bottom: 12px; font-size: 13px; max-width: 700px; }
    .generations-viz .section-header { font-weight: bold; font-size: 14px; margin: 10px 0 5px 0; padding: 4px 6px; border-radius: 3px; color: white; display: block; }
    .generations-viz .pre-intervention-header { background-color: #2471A3; }
    .generations-viz .post-intervention-header { background-color: #27AE60; }
    .generations-viz .generation-container { margin-bottom: 8px; padding: 3px; border-left: 3px solid rgba(100, 100, 100, 0.5); }
    .generations-viz .generation-text { background-color: rgba(200, 200, 200, 0.2); padding: 6px 8px; border-radius: 3px; border: 1px solid rgba(100, 100, 100, 0.5); font-family: monospace; font-weight: 500; white-space: pre-wrap; line-height: 1.2; font-size: 13px; overflow-x: auto; }
    .generations-viz .base-text { color: rgba(100, 100, 100, 0.9); }
    .generations-viz .new-text  { background-color: rgba(255, 255, 0, 0.25); font-weight: bold; padding: 1px 0; border-radius: 2px; }
    .generations-viz .pre-intervention-item { border-left-color: #2471A3; }
    .generations-viz .post-intervention-item { border-left-color: #27AE60; }
    .generations-viz .generation-number { font-weight: bold; margin-bottom: 3px; color: rgba(70, 70, 70, 0.9); font-size: 12px; }
    </style>
    <div class="generations-viz">
    <div class="section-header pre-intervention-header">Pre-intervention generations:</div>
    """
    for i, gen_text in enumerate(pre_intervention_gens):
        if gen_text.startswith(original_text):
            base_part = html.escape(original_text)
            new_part = html.escape(gen_text[len(original_text):])
            formatted_text = f'<span class="base-text">{base_part}</span><span class="new-text">{new_part}</span>'
        else:
            formatted_text = html.escape(gen_text)
        html_content += f"""
        <div class="generation-container pre-intervention-item">
            <div class="generation-number">Generation {i + 1}</div>
            <div class="generation-text">{formatted_text}</div>
        </div>
        """
    html_content += '<div class="section-header post-intervention-header">Post-intervention generations:</div>'
    for i, gen_text in enumerate(post_intervention_gens):
        if gen_text.startswith(original_text):
            base_part = html.escape(original_text)
            new_part = html.escape(gen_text[len(original_text):])
            formatted_text = f'<span class="base-text">{base_part}</span><span class="new-text">{new_part}</span>'
        else:
            formatted_text = html.escape(gen_text)
        html_content += f"""
        <div class="generation-container post-intervention-item">
            <div class="generation-number">Generation {i + 1}</div>
            <div class="generation-text">{formatted_text}</div>
        </div>
        """
    html_content += "</div>"
    display(HTML(html_content))
