#!/bin/bash

if [[ $# -ne 1 ]] ;
	then echo "Usage: ./dataset_dimension.sh <path/to/images>"
	exit 2
fi

fname=`basename $1`.txt

identify $1/*.tiff | cut -d' ' -f3 | sed 's/x/, /' > $fname ;

echo " Resultado escrito em $fname" ;
