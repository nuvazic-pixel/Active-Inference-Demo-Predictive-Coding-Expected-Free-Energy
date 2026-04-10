# Notes

## Conceptual progression

### `mini_pc_pytorch.py`
Start here if you want the cleanest predictive-coding intuition:
- latent inference,
- top-down reconstruction,
- local error minimization.

### `mini_pc_active_inference.py`
This adds the first real active-inference ingredients:
- precision weighting,
- nonlinear generative mapping,
- policy scoring,
- simple ablations.

### `mnist_foveated_active_inference_lite.py`
This is the first benchmark that is easy to explain to other people:
- same image,
- same glimpse budget,
- guided sampling vs random sampling.

If guided glimpses reliably beat random glimpses, the policy is earning its keep.

## Recommended benchmark statement

> Under the same observation budget, uncertainty-guided glimpses improve classification accuracy and/or reconstruction compared with random glimpses.

## Practical advice

- Change **one knob at a time**.
- Always test **across seeds** before trusting a gain.
- Add sanity checks directly into the code, not only in your notebook.
- Prefer clean baselines over bigger models.
