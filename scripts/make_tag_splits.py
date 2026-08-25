"""Split train/val/test into 4-way (day/night × clear/rain) tag subsets.

Reads scene_tags.json (produced upstream from scene.json + CLIP for test)
and writes 12 new split files:
  {train,val,test}_{day_clear,day_rain,night_clear,night_rain}.txt

Each output file preserves the original "sample_id\\tabs_path" format so it
drops in as a direct replacement for the base split path in dataset cfg.

Untagged sample_ids (should be 0 with our extended tag file) are dropped
with a warning per split.
"""
import json
import os
from collections import Counter


SPLITS_DIR = "/data/public/nuScenes/derived/splits"
TAGS_JSON = f"{SPLITS_DIR}/scene_tags.json"
SPLIT_NAMES = ("train", "val", "test")
TAG_NAMES = ("day_clear", "day_rain", "night_clear", "night_rain")


def tag_key(t):
    return f"{'night' if t['night'] else 'day'}_{'rain' if t['rain'] else 'clear'}"


def main():
    tags = json.load(open(TAGS_JSON))
    print(f"[load] scene_tags.json → {len(tags)} tagged samples")

    for split in SPLIT_NAMES:
        src = f"{SPLITS_DIR}/{split}.txt"
        out_files = {t: open(f"{SPLITS_DIR}/{split}_{t}.txt", "w") for t in TAG_NAMES}
        stats = Counter()
        with open(src) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line: continue
                sid = line.split("\t", 1)[0]
                if sid not in tags:
                    stats["<untagged>"] += 1
                    continue
                k = tag_key(tags[sid])
                out_files[k].write(line + "\n")
                stats[k] += 1
        for f in out_files.values():
            f.close()
        total = sum(stats.values())
        print(f"\n[{split}]  total={total}")
        for t in TAG_NAMES:
            print(f"  {t:<12}  {stats[t]:>6}  ({stats[t]/total:.1%})")
        if stats.get("<untagged>", 0):
            print(f"  <untagged>   {stats['<untagged>']}  (dropped)")


if __name__ == "__main__":
    main()
