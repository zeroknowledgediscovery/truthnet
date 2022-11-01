#!/bin/bash

pdoc --html ../truthnet/ -o docs/ -c latex_math=True -f --template-dir docs/dark_templates

cp -r docs/emergenet/* docs
