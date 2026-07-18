#!/usr/bin/env bash

src="$1"
dst="$2"

mkdir -p "$dst"

i=0
for file in "$src"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    [[ -e "$file" ]] || continue
    ((i++))

    if (( i % 4 == 0 )); then
        cp -- "$file" "$dst/"
    fi
done
