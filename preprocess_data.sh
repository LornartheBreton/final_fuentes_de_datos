#! /bin/bash

xsv select 2-4 $1 > $2

./train.sh
