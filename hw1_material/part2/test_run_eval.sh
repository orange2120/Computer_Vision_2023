#!/bin/bash
for i in {1..10}
do
    python3 eval.py 2>&1 | grep "Time"
done