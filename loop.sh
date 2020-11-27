#!/bin/bash
for((i = 0; i < 50; i++))
do
	 echo -e "Amostra $i "
	 sudo iw wlp3s0 scan | grep 'wlp3s0\|signal' > teste && awk -f preprocessing.awk teste >> bancodedados
done


