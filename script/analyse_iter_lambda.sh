#/bin/bash

OPTIONS_PLOTTING='every 10::1'
REPS=""
TITLES=""
for lambda in "0.1000" "0.0100" "0.0010" "0.0001" "0.0000"
do REPS="$REPS bruit$1_lambda"$lambda"00"
TITLES="$TITLES MCRQ_lambda="$lambda""
echo $REPS
echo $TITLES
done




#####################################################################
# J_reg_J.eps
#####################################################################

AXIS_X="iteration"
AXIS_Y="J_reg/J"
FILE_IN="crit.dat"
FILE_OUT="J_reg_J.eps"
NUM_A="1"
NUM_B="6"


analyse_plot.sh "$AXIS_X" "$AXIS_Y" "$FILE_IN"  "$FILE_OUT" "$NUM_A" "$NUM_B"  "$REPS" "$TITLES" "$OPTIONS_PLOTTING"


#####################################################################
# step.eps
#####################################################################

AXIS_X="iteration"
AXIS_Y="step"
FILE_IN="crit.dat"
FILE_OUT="step.eps"
NUM_A="1"
NUM_B="7"


analyse_plot.sh "$AXIS_X" "$AXIS_Y" "$FILE_IN"  "$FILE_OUT" "$NUM_A" "$NUM_B"  "$REPS" "$TITLES" "$OPTIONS_PLOTTING"



#####################################################################
# J.eps
#####################################################################

AXIS_X="iteration"
AXIS_Y="J"
FILE_IN="crit.dat"
FILE_OUT="J.eps"
NUM_A="1"
NUM_B="2"



analyse_plot.sh "$AXIS_X" "$AXIS_Y" "$FILE_IN"  "$FILE_OUT" "$NUM_A" "$NUM_B"  "$REPS" "$TITLES" "$OPTIONS_PLOTTING"


#####################################################################
# eam.eps
#####################################################################


AXIS_X="iteration"
AXIS_Y="distance L1"
FILE_IN="eam.dat"
FILE_OUT="eam.eps"
NUM_A="1"
NUM_B="3"



analyse_plot.sh "$AXIS_X" "$AXIS_Y" "$FILE_IN"  "$FILE_OUT" "$NUM_A" "$NUM_B"  "$REPS" "$TITLES" "$OPTIONS_PLOTTING"
