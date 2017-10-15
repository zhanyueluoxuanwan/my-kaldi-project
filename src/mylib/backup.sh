#!/bin/bash

[ -d ./backup ] || mkdir -p ./backup

for x in `find . |grep -E "*\.cc$"`
do
	cp ${x} ./backup
done
for x in `find . |grep -E "*\.h$"`
do
	cp ${x} ./backup
done
[ -f Makefile ] && cp Makefile ./backup
