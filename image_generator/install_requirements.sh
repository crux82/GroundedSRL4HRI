#!/bin/bash
if [ ! -d "GroundingDINO" ]; then
  git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
  mv Grounded-Segment-Anything/GroundingDINO .
  rm -rf Grounded-Segment-Anything
fi
pip install -q -r requirements.txt
pip install -q ./GroundingDINO/
pip install -q supervision==0.21.0
