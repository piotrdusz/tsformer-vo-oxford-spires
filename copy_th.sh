#!/usr/bin/env bash

src="oxford/church_02/sequence/"
dst="oxford/church_02_5th/sequence/"

mkdir -p "$dst"

i=0
for file in "$src"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    [[ -e "$file" ]] || continue
    ((i++))

    if (( i % 4 == 0 )); then
        cp -- "$file" "$dst/"
    fi
done
