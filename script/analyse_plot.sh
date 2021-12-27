#/bin/bash



AXIS_X="$1"
AXIS_Y="$2"
FILE_IN="$3"
FILE_OUT="$4"
NUM_A="$5"
NUM_B="$6"
REPS="$7"
TITLES="$8"
OPTIONS_PLOTTING="$9"


echo "$REPS"

i=0
for rep2 in  $REPS
do 
i=`expr $i + 1`
echo $i
cmd1_awk="echo \"$REPS\" |awk '{print \$$i}'"
echo $cmd1_awk
repertoire=`eval "$cmd1_awk"`
cmd2_awk="echo \"$TITLES\" |awk '{print \$$i}'"
echo $cmd2_awk
titre=`eval "$cmd2_awk"`

if [ $i -eq 1 ]
then
cmd_plot="plot \"$repertoire/$FILE_IN\" using $NUM_A:$NUM_B $OPTIONS_PLOTTING  with linespoints title  '$titre' "
else cmd_plot="$cmd_plot ,\"$repertoire/$FILE_IN\" using $NUM_A:$NUM_B $OPTIONS_PLOTTING  with linespoints title  '$titre' "
fi

done

echo $cmd_plot

cmd_plot=`echo $cmd_plot | sed 's/0,/0\./g'` 
echo $cmd_plot


cmd_output="set output \""$FILE_OUT"\""
echo $cmd_output
cmd_xlabel="set xlabel \""$AXIS_X"\""
echo $cmd_ylabel
cmd_ylabel="set ylabel \""$AXIS_Y"\""
echo $cmd_ylabel


#eam_iter.eps
gnuplot << EOF
set terminal postscript landscape
set terminal svg enhanced background rgb 'white'
set encoding iso_8859_1
$cmd_output
$cmd_xlabel
$cmd_ylabel
set key box
#set key 60,0.36
$cmd_plot


EOF

display "$FILE_OUT"&

