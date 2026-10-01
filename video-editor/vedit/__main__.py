"""Command line:  python -m vedit <command> ...

  new <video> [--name N]        create projects/N, copy the video, ingest it
  transcribe <project>          word timestamps (whisper.cpp + ivrit.ai) -> work/words.md
  words <project>               print the words table
  mask <project>                person mask for every frame -> work/masks.npy
  render <project> [--name v1]  render edit.py -> out/v1.mp4 (+ --preview for a fast draft)
  frame <project> <t>           render one output moment to PNG
  contact <project> <video>     contact sheet: 6 frames per second, times on each
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2

from . import mask, media, transcribe
from .project import load

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vedit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("new")
    a.add_argument("video")
    a.add_argument("--name")
    a.add_argument("--fps")
    a = sub.add_parser("transcribe")
    a.add_argument("project")
    a.add_argument("--model")
    a.add_argument("--prompt")
    a.add_argument("--lang", default="he")
    for c in ("words", "mask"):
        sub.add_parser(c).add_argument("project")
    a = sub.add_parser("render")
    a.add_argument("project")
    a.add_argument("--name", default="v1")
    a.add_argument("--preview", action="store_true")
    a.add_argument("--variant", help="passed to build(p, variant=...) in edit.py")
    a.add_argument("--workers", type=int)
    a = sub.add_parser("frame")
    a.add_argument("project")
    a.add_argument("t", type=float)
    a.add_argument("--out")
    a = sub.add_parser("contact")
    a.add_argument("project")
    a.add_argument("video")
    a.add_argument("--per-sec", type=int, default=6)
    args = ap.parse_args(argv)

    if args.cmd == "new":
        proj = ROOT / "projects" / (args.name or Path(args.video).stem)
        (proj / "work").mkdir(parents=True, exist_ok=True)
        src = proj / ("source" + Path(args.video).suffix.lower())
        if not src.exists():
            shutil.copy2(args.video, src)
        print(json.dumps(media.ingest(src, proj / "work", fps=args.fps), indent=1, ensure_ascii=False))
        if not (proj / "edit.py").exists():
            shutil.copy(ROOT / "projects" / "_template" / "edit.py", proj / "edit.py")
        print(proj)
    elif args.cmd == "transcribe":
        print(transcribe.table(transcribe.transcribe(Path(args.project) / "work", args.model, args.lang, args.prompt)))
    elif args.cmd == "words":
        print((Path(args.project) / "work" / "words.md").read_text())
    elif args.cmd == "mask":
        print(mask.compute(Path(args.project) / "work"))
    elif args.cmd == "render":
        p = load(args.project, **({"variant": args.variant} if args.variant else {}))
        print(p.render(args.name, preview=args.preview, workers=args.workers))
        print(json.dumps(p.report, ensure_ascii=False, default=str)[:3000])
    elif args.cmd == "frame":
        p = load(args.project)
        p.out.mkdir(parents=True, exist_ok=True)
        out = args.out or str(p.out / f"frame_{args.t:.2f}.png")
        cv2.imwrite(out, p.frame_at(args.t))
        print(out)
    elif args.cmd == "contact":
        print(*load(args.project).contact(args.video, args.per_sec), sep="\n")


if __name__ == "__main__":
    main()
