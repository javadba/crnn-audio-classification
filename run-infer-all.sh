#!/usr/bin/bash

bdir='/data/mike-coughs/audio'
fcnt=0
for d in bdir; do
  echo $d
  for f in $d; do
    echo "$f-$d"
     ./run.py -c config.mike-coughs23.json --cfg crnn.mike-coughs.cfg  -r "$bdir/saved_cv/0620_131620/checkpoints/model_best.pth" $f
    fcnt=$((fcnt + 1))
    if [[ "$fcnt" -gt 3 ]]; then
      exit 1
    fi
  done
done
