#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: ./reencode.sh [input dir] [output dir]"
fi

indir=$1
outdir=$2

mkdir ${outdir}
if [[ ! -d "${outdir}" ]]; then
  echo "${outdir} doesn't exist. Creating it.";
  mkdir -p ${outdir}
fi

#for inname in $(ls ${indir}/*mp4)
#do
#	class_path="$(dirname "$inname")"
#	class_name="${class_path##*/}"
#
#	outname="${outdir}/${class_name}/${inname##*/}"
#	outname="${outname%.*}.mp4"

#	mkdir -p "$(dirname "$outname")"
#	ffmpeg -i ${inname} -vf setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo ${outname}

#done

for c in $(ls ${indir})
do
    for inname in $(ls ${indir}/${c}/*mp4)
    do
            class_path="$(dirname "$inname")"
            class_name="${class_path##*/}"

            outname="${outdir}/${class_name}/${inname##*/}"
            outname="${outname%.*}.mp4"

            mkdir -p "$(dirname "$outname")"
            ffmpeg -i ${inname} -vf scale=iw:ih,setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo ${outname}
    done
done

