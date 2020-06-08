#!/usr/bin/env bash
#### HOW TO USE ####
# `bash script/convert.sh %filepath`
# This translates to 
# `convert %filepath -resize 1024x600\! %filepath`
####

FILEPATH=$1
DEFAULT_WIDTH=1024
DEFAULT_HEIGHT=600

# echo "the PWD is : ${pwd}"
convert $FILEPATH -resize ${DEFAULT_WIDTH}x${DEFAULT_HEIGHT}\! $FILEPATH

