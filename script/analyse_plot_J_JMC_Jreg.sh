#/bin/bash



repertoire=$1
FILE_IN="Iteration.dat"
FILE_OUT="J.eps"
OPTIONS_PLOTTING=""



cmd_plot="plot \"$repertoire/$FILE_IN\" using 1:2 $OPTIONS_PLOTTING  with linespoints title  'J' "
#cmd_plot="plot "
cmd_plot="$cmd_plot, \"$repertoire/$FILE_IN\" using 1:3 $OPTIONS_PLOTTING  with linespoints title  'J_{MC}'"
cmd_plot="$cmd_plot, \"$repertoire/$FILE_IN\" using 1:4 $OPTIONS_PLOTTING  with linespoints title  'J_{reg}'"



echo $cmd_plot



cmd_output="set output \""$FILE_OUT"\""
echo $cmd_output
cmd_xlabel="set xlabel \""num iteration"\""
echo $cmd_ylabel
cmd_ylabel="set ylabel \""J"\""
echo $cmd_ylabel


#eam_iter.eps
gnuplot << EOF
set terminal postscript landscape
set terminal svg enhanced background rgb 'white'
set encoding iso_8859_1
set nologscale; set logscale y
$cmd_output
$cmd_xlabel
$cmd_ylabel
set key box
#set key 60,0.36
$cmd_plot


EOF

display "$FILE_OUT"&

