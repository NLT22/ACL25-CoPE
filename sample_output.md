# SynCPR to Fashion-IQ Sample

This file shows one concrete example of how a SynCPR sample is converted into the Fashion-IQ-compatible format used by this repository.

## 1. Original SynCPR sample

```json
{
  "reference_caption": "The young woman with black hair is wearing an ebony black blouse, a navy blue skirt, and black heeled sandals. She is holding a silver clutch.",
  "target_caption": "The young woman with black hair is wearing an ebony black blouse, a light gray skirt, and black heeled sandals. She is carrying a large black leather handbag.",
  "reference_image_path": "test2/sub_img/img_left/10732-1_left.png",
  "target_image_path": "test2/sub_img/img_right/10732-1_right.png",
  "edit_caption": "Wearing light gray skirt, carrying a large black leather handbag.",
  "cpr_id": 0
}
```

## 2. Staged extracted images

After the `extract` step, the referenced images are expected to exist under the staging directory with the same relative paths:

```text
<staging-dir>/
└── test2/
    └── sub_img/
        ├── img_left/
        │   └── 10732-1_left.png
        └── img_right/
            └── 10732-1_right.png
```

Example:

```text
E:\datasets\syncpr-staging\test2\sub_img\img_left\10732-1_left.png
E:\datasets\syncpr-staging\test2\sub_img\img_right\10732-1_right.png
```

## 3. Normalized image names

The converter normalizes image names by:

- removing `.png`
- preserving the relative path
- joining path parts with `__`

So the two image paths become:

```text
test2/sub_img/img_left/10732-1_left.png
-> test2__sub_img__img_left__10732-1_left

test2/sub_img/img_right/10732-1_right.png
-> test2__sub_img__img_right__10732-1_right
```

## 4. Caption conversion

The repository's Fashion-IQ loader expects:

- `candidate`: reference image name
- `target`: target image name
- `captions`: a list with exactly 2 text edits

The SynCPR field:

```text
Wearing light gray skirt, carrying a large black leather handbag.
```

is converted into:

```json
[
  "Wearing light gray skirt",
  "carrying a large black leather handbag"
]
```

## 5. Output entry in cap.dress.<split>.json

One converted sample in `captions/cap.dress.train.json` or `captions/cap.dress.val.json` or `captions/cap.dress.test.json` will look like this:

```json
{
  "candidate": "test2__sub_img__img_left__10732-1_left",
  "target": "test2__sub_img__img_right__10732-1_right",
  "captions": [
    "Wearing light gray skirt",
    "carrying a large black leather handbag"
  ]
}
```

Which split it lands in depends on the deterministic hash-based split logic.

## 6. Output image files in Fashion-IQ tree

The final Fashion-IQ-compatible output will contain copied images like:

```text
<output-dir>/
├── images/
│   ├── test2__sub_img__img_left__10732-1_left.png
│   └── test2__sub_img__img_right__10732-1_right.png
├── captions/
│   ├── cap.dress.train.json
│   ├── cap.dress.val.json
│   └── cap.dress.test.json
└── image_splits/
    ├── split.dress.train.json
    ├── split.dress.val.json
    └── split.dress.test.json
```

## 7. Output image_splits entry

The corresponding image names are also added to the split file for that split:

```json
[
  "test2__sub_img__img_left__10732-1_left",
  "test2__sub_img__img_right__10732-1_right"
]
```

In the real output, each split file contains many image names, not just these two.

## 8. Summary mapping

```text
reference_image_path -> candidate + copied reference image
target_image_path    -> target + copied target image
edit_caption         -> captions[0], captions[1]
cpr_id               -> not written into Fashion-IQ output
reference_caption    -> not written into Fashion-IQ output
target_caption       -> not written into Fashion-IQ output
```
