# Example: SAM3 text-prompt segmentation

Reference files for the SAM3 hierarchical pipeline (`sam3_process.py` / `sam3_ui.py`).
Concepts here are deliberately generic (a room with a few people and objects) —
adapt the prompts and names to your own scene.

## Files

| File | Purpose |
|---|---|
| `concepts_example.json` | Concept definitions passed to `sam3_process.py --concepts`. Each entry: a `name` (label used on disk / in the UI) and a `text_prompt` (what SAM3 actually searches for). `detection_frame` is optional (default 0) and only affects the UI preview, not what gets detected. |
| `concepts_vocabulary_example.json` | Optional. Maps each concept name to a list of suggested **instance** names, shown as a dropdown when you rename detected instances in `sam3_ui.py` (Vocabulary → Load Vocabulary File). You can always type a custom name instead. |

A concept (e.g. `person`) can produce several instances; the vocabulary is where
you list the names you expect (`adult`, `child`, `experimenter`, ...). Concepts
that only ever have one instance (`floor`, `door`) can just repeat the name.

## Usage

```bash
# 1. Run detection + propagation for every concept in the file
python sam3_process.py \
    --video   example/your_video.mp4 \
    --project example/your_project \
    --concepts example/concepts_example.json

# 2. Review / rename / refine in the UI
python sam3_ui.py example/your_project
#    Vocabulary menu -> Load Vocabulary File -> concepts_vocabulary_example.json

# 3. Re-run refinement clicks made in the UI
python sam3_process.py --project example/your_project --refine

# 4. Export an overlay video
python sam3_process.py --project example/your_project --export example/output.mp4
```

A ready-to-run sample (1 minute of egocentric video with gaze data, concepts and vocabulary) is in [`sample_data/child_1min/`](../sample_data/child_1min/README.md).
