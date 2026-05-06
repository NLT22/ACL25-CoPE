import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple


def iter_top_level_array(json_path: Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Stream objects from a top-level JSON array."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False

    with json_path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break

            buffer += chunk
            while True:
                buffer = buffer.lstrip()

                if not started:
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ValueError(f"{json_path} is not a JSON array.")
                    started = True
                    buffer = buffer[1:]
                    continue

                if not buffer:
                    break

                if buffer[0] == "]":
                    finished = True
                    buffer = buffer[1:]
                    break

                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue

                try:
                    value, index = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break

                yield value
                buffer = buffer[index:]

    if not started or not finished:
        buffer = buffer.lstrip()
        if buffer == "]":
            finished = True

    if not started or not finished:
        raise ValueError(f"Could not fully parse JSON array from {json_path}.")


def split_edit_caption(edit_caption: str) -> List[str]:
    text = " ".join(str(edit_caption).strip().split())
    if not text:
        return ["No change", "No change"]

    comma_parts = [part.strip(" ,.;") for part in text.split(",") if part.strip(" ,.;")]
    if len(comma_parts) >= 2:
        return [comma_parts[0], ", ".join(comma_parts[1:])]

    lower_text = text.lower()
    token = " and "
    if token in lower_text:
        index = lower_text.find(token)
        left = text[:index].strip(" ,.;")
        right = text[index + len(token):].strip(" ,.;")
        if left and right:
            return [left, right]

    normalized = text.strip(" ,.;")
    return [normalized, normalized]


def build_image_name(relative_path: str) -> str:
    return "__".join(Path(relative_path).with_suffix("").parts)


def choose_split(group_key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.md5(group_key.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < train_ratio:
        return "train"
    if score < train_ratio + val_ratio:
        return "val"
    return "test"


def collect_needed_images(
    input_json: Path, limit: int, progress_interval: int = 50000
) -> Tuple[Dict[str, Set[str]], int]:
    needed: Dict[str, Set[str]] = {}
    sample_count = 0

    for index, sample in enumerate(iter_top_level_array(input_json)):
        if limit and index >= limit:
            break

        sample_count += 1
        if sample_count == 1 or sample_count % progress_interval == 0:
            print(
                f"[index] scanned {sample_count} samples from {input_json}",
                flush=True,
            )
        for key in ("reference_image_path", "target_image_path"):
            relative_path = sample[key]
            parent = str(Path(relative_path).parent)
            needed.setdefault(parent, set()).add(Path(relative_path).name)

    return needed, sample_count


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def maybe_copy_file(source_path: Path, target_path: Path, overwrite: bool) -> bool:
    if target_path.exists() and not overwrite:
        return False
    ensure_parent(target_path)
    shutil.copy2(source_path, target_path)
    return True


def extract_needed_archives(
    source_root: Path,
    staging_dir: Path,
    needed_images: Dict[str, Set[str]],
    overwrite: bool,
) -> Tuple[int, int, int]:
    zip_count = 0
    extracted_images = 0
    skipped_images = 0

    for relative_parent, filenames in sorted(needed_images.items()):
        source_dir = source_root / relative_parent
        staging_subdir = staging_dir / relative_parent

        if not source_dir.exists():
            raise FileNotFoundError(f"Missing source directory referenced by SynCPR.json: {source_dir}")

        unresolved = set(filenames)

        for image_name in list(unresolved):
            source_image = source_dir / image_name
            if source_image.exists():
                target_image = staging_subdir / image_name
                changed = maybe_copy_file(source_image, target_image, overwrite)
                extracted_images += int(changed)
                skipped_images += int(not changed)
                unresolved.remove(image_name)

        zip_paths = sorted(source_dir.glob("batch_*.zip"))
        for zip_index, zip_path in enumerate(zip_paths, start=1):
            if not unresolved:
                break

            zip_count += 1
            print(
                f"[extract] scanning {zip_path} "
                f"({zip_index}/{len(zip_paths)} in {source_dir.name}, remaining {len(unresolved)})",
                flush=True,
            )
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    archive_members = {
                        Path(member).name: member
                        for member in archive.namelist()
                        if not member.endswith("/")
                    }
                    matched_names = sorted(unresolved.intersection(archive_members.keys()))
                    for image_name in matched_names:
                        target_image = staging_subdir / image_name
                        if target_image.exists() and not overwrite:
                            skipped_images += 1
                            unresolved.remove(image_name)
                            continue

                        ensure_parent(target_image)
                        with archive.open(archive_members[image_name]) as source_handle:
                            with target_image.open("wb") as target_handle:
                                shutil.copyfileobj(source_handle, target_handle)
                        extracted_images += 1
                        unresolved.remove(image_name)
            except zipfile.BadZipFile as exc:
                raise RuntimeError(f"Corrupted zip archive: {zip_path}") from exc

        if unresolved:
            missing_list = ", ".join(sorted(unresolved)[:5])
            raise FileNotFoundError(
                f"Could not resolve {len(unresolved)} images under {source_dir}. "
                f"Examples: {missing_list}"
            )

    return zip_count, extracted_images, skipped_images


def convert_dataset(
    input_json: Path,
    image_root: Path,
    output_dir: Path,
    category: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    limit: int,
    overwrite: bool,
) -> Tuple[Dict[str, int], int]:
    captions_dir = output_dir / "captions"
    image_splits_dir = output_dir / "image_splits"
    images_dir = output_dir / "images"
    captions_dir.mkdir(parents=True, exist_ok=True)
    image_splits_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    split_samples: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    split_image_names: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    copied_images = 0
    counts = {"train": 0, "val": 0, "test": 0}
    copied_name_to_source: Dict[str, Path] = {}

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0.")

    for index, sample in enumerate(iter_top_level_array(input_json)):
        if limit and index >= limit:
            break

        reference_rel = sample["reference_image_path"]
        target_rel = sample["target_image_path"]
        reference_src = image_root / reference_rel
        target_src = image_root / target_rel

        if not reference_src.exists():
            cpr_id = sample.get("cpr_id", "unknown")
            raise FileNotFoundError(
                f"Missing extracted reference image for sample index {index}, cpr_id {cpr_id}: {reference_src}"
            )
        if not target_src.exists():
            cpr_id = sample.get("cpr_id", "unknown")
            raise FileNotFoundError(
                f"Missing extracted target image for sample index {index}, cpr_id {cpr_id}: {target_src}"
            )

        reference_name = build_image_name(reference_rel)
        target_name = build_image_name(target_rel)
        captions = split_edit_caption(sample.get("edit_caption", ""))
        pair_key = "||".join(sorted([reference_rel, target_rel]))
        split = choose_split(pair_key, train_ratio, val_ratio)

        split_samples[split].append(
            {"candidate": reference_name, "target": target_name, "captions": captions}
        )
        split_image_names[split].add(reference_name)
        split_image_names[split].add(target_name)
        counts[split] += 1

        copied_name_to_source.setdefault(reference_name, reference_src)
        copied_name_to_source.setdefault(target_name, target_src)

    for split, samples in split_samples.items():
        caption_file = captions_dir / f"cap.{category}.{split}.json"
        split_file = image_splits_dir / f"split.{category}.{split}.json"

        with caption_file.open("w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False, indent=2)
        with split_file.open("w", encoding="utf-8") as handle:
            json.dump(sorted(split_image_names[split]), handle, ensure_ascii=False, indent=2)

    for image_name, source_path in copied_name_to_source.items():
        target_path = images_dir / f"{image_name}.png"
        if target_path.exists() and not overwrite:
            continue
        ensure_parent(target_path)
        shutil.copy2(source_path, target_path)
        copied_images += 1

    return counts, copied_images


def run_extract(args: argparse.Namespace) -> None:
    print(f"[extract] indexing image paths from {args.input_json}", flush=True)
    needed_images, sample_count = collect_needed_images(args.input_json, args.limit)
    print(f"[extract] indexed {sample_count} samples from {args.input_json}", flush=True)
    print(f"[extract] extracting into {args.staging_dir}", flush=True)
    zip_count, extracted_images, skipped_images = extract_needed_archives(
        source_root=args.source_root,
        staging_dir=args.staging_dir,
        needed_images=needed_images,
        overwrite=args.overwrite,
    )
    print(
        "[extract] done. "
        f"archives scanned: {zip_count}, images extracted: {extracted_images}, images skipped: {skipped_images}",
        flush=True,
    )


def run_convert(args: argparse.Namespace) -> None:
    counts, copied_images = convert_dataset(
        input_json=args.input_json,
        image_root=args.staging_dir,
        output_dir=args.output_dir,
        category=args.category,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"[convert] output directory: {args.output_dir}", flush=True)
    print(
        "[convert] samples - "
        f"train: {counts['train']}, val: {counts['val']}, test: {counts['test']}",
        flush=True,
    )
    print(f"[convert] images copied into Fashion-IQ tree: {copied_images}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SynCPR archives into a staging directory and convert them into Fashion-IQ format."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run", "extract", "convert"],
        default="run",
        help="Pipeline step to execute. Default: run",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(r"E:\SynCPR\SynCPR.json"),
        help="Path to SynCPR.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"E:\SynCPR"),
        help="Root directory containing SynCPR zip archives.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
        help="Directory where extracted SynCPR images will be staged.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where Fashion-IQ-formatted data will be written.",
    )
    parser.add_argument(
        "--category",
        choices=["dress", "shirt", "toptee"],
        default="dress",
        help="Fashion-IQ category name to emit.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of samples to process for quick testing. 0 means all samples.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction and only run conversion when command=run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing staged images and Fashion-IQ image files.",
    )

    args = parser.parse_args()
    if args.command in {"run", "convert"} and args.output_dir is None:
        parser.error("--output-dir is required for 'run' and 'convert'.")
    return args


def main() -> None:
    args = parse_args()

    if args.command == "extract":
        run_extract(args)
        return

    if args.command == "convert":
        run_convert(args)
        return

    if not args.skip_extract:
        run_extract(args)
    else:
        print(
            f"[extract] skipped. Using existing staged images in {args.staging_dir}",
            flush=True,
        )

    run_convert(args)


if __name__ == "__main__":
    main()
