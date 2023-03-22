#!/bin/bash

if [ $# -eq 0 ] ; then
	echo "" ;
	echo 'Usage: ./archive-experiment.sh <EXPERIMENT DIR>' ;
	echo "" ;
	exit ;
fi

basename=`echo $1 | sed 's/\///'`

tar cf ${basename}.tar ${basename} ;
md5sum ${basename}.tar > ${basename}.md5 ;
echo "Archived ${basename}" ;
