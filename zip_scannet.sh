
#!/bin/bash

SCANS_DIR="/Users/siddharthraj/classes/cv/cv_final/data/scans"
OUT_DIR="/Users/siddharthraj/classes/cv/cv_final/data/scannet_zips"

mkdir -p "$OUT_DIR"

SCENES=(
    scene0000_00 scene0001_00 scene0002_00 scene0003_00
    scene0004_00 scene0005_00 scene0006_00 scene0007_00
    scene0008_00 scene0009_00 scene0010_00
)

for scene in "${SCENES[@]}"; do
    src="$SCANS_DIR/$scene/exported"
    out="$OUT_DIR/${scene}.zip"

    if [ ! -d "$src" ]; then
        echo "SKIP $scene — exported/ not found"
        continue
    fi

    echo "Zipping $scene..."
    start=$(date +%s)
    # Run from SCANS_DIR so zip preserves scene/exported/ path structure
    (cd "$SCANS_DIR" && zip -0 -r -q "$out" "$scene/exported/")
    end=$(date +%s)
    echo "  Done in $((end - start))s — $(du -sh "$out" | cut -f1)"
done

echo ""
echo "All zips in: $OUT_DIR"
echo "Upload that folder to MyDrive/final_proj/scannet_zips/ on Google Drive"
