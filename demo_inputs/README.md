# Demo inputs

Two dates of one synthetic 512×512 scene. Upload both to the UI and ask about change;
upload one and ask about land cover.

| file | scene |
|---|---|
| `demo_input_tif_1.tif` | reservoir at 60 px radius |
| `demo_input_tif_2.tif` | the same scene, reservoir grown to 92 px |

4 bands (`B04`, `B03`, `B08`, `B11`), EPSG:32643, **10 m GSD** written into the affine
transform — which is what lets the answer be in hectares rather than pixels.

## Ground truth

Synthetic on purpose: you already know the right answer, so you can tell whether the
system computed it or guessed.

| quantity | truth |
|---|---|
| water, date 1 | 1,127,700 m² (112.77 ha) |
| water, date 2 | 2,656,100 m² (265.61 ha) |
| water, change | +1,528,400 m² (+152.84 ha, +135.5%) |
| built-up, both dates | 1,260,000 m² — **change must be 0** |

Measured through the running system: water 1,127,900 → 2,656,300, delta 1,528,400 m²
(+135.5%); built-up delta exactly 0. The few-hundred-m² offsets on the endpoints are the
sensor noise the generator adds, and they cancel in the difference.

The built-up row is the one that matters. NDBI is high over open water as well as
concrete, so before the water mask was subtracted this scene reported the growing lake as
a construction boom. If a change ever reappears there, that bug is back.

## Questions to try

- "How has the water body changed between these two dates?" → *water increased by 1.53 km²*
- "How much has the built-up area changed?" → *built up is unchanged*
- "Describe the land cover in this image" (one file) → proportions, with a warning that
  the indices overlap and need not sum to 100

Regenerate with `python scripts/make_demo_inputs.py`. It prints the ground truth again.

## What will not work

Any question needing a learned model — VQA, captioning, object counting, grounding,
semantic change. Those tools are unbound because their weights do not exist yet, and the
UI will say so rather than answer.

An RGB screenshot (PNG or JPEG) is refused: it has no near-infrared band, so no spectral
index applies.
