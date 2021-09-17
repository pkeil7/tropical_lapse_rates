
# Calculations done with cdo version 1.9.6

# Get monmeans, interpolate and tropical means from ICON experiments

# This does not work if you redo some of the files and don't delete them beofre from the WORK_DIR
# do 'rm $WORK_DIR/${memb}_${varname}*'

# ----- Specifify variables etc

DATA_DIR=../icon-aes/experiments
WORK_DIR=../data/icon-param-exp
TEMP_DIR=#some directory where files can go temporarliy

#members=('ppk0002' 'ppk0010' 'ppk0011' 'ppk0012' 'ppk0013' 'ppk0014' 'ppk0015' 'ppk0016' 'ppk0017' 'ppk0018' 'ppk0019' 'ppk0022' 'ppk0023' 'ppk0024' 'ppk0025' 'ppk0026' 'ppk0027' 'ppk0030' 'ppk0031' 'ppk0032' 'ppk0033' 'ppk0034' 'ppk0035' 'ppk0040' 'ppk0041' 'ppk0042' 'ppk0043' 'ppk0044' 'ppk0045' 'ppk0046' 'ppk0050' 'ppk0051' 'ppk0052' 'ppk0053' 'ppk0054' 'ppk0055' 'ppk0056' 'ppk0057' 'ppk0058' 'ppk0060' 'ppk0061' 'ppk0062' 'ppk0063' 'ppk0064' 'ppk0065' 'ppk0066' 'ppk0067' 'ppk0068' 'ppk0069' 'ppk0070' 'ppk0071' 'ppk0072' 'ppk0073' 'ppk0074')

members=('ppk0002')
pressure_lvls="100000,85000,70000,60000,50000,40000,30000,25000,20000,15000,10000,5000,2500,1500,1000"

#3d vars always need pressure for interpolation, var=??,pfull,ps
var=tend_ta_dyn,pfull
varname=tend_ta_dyn
stream=phy_3d_ml
tropical_mean=true
last_year=2014

# ------ Run:

for memb in ${members[@]}; do

    # clean bofore we start:
    rm -f ${WORK_DIR}/${memb}_${varname}_*1979-${last_year}_mm.nc
    rm -f ${TEMP_DIR}/${memb}_${varname}_*mm.nc


	# get relevant variables
	last_year_cdo=$(( $last_year + 1 ))
	for yr in $( seq 1979 $last_year_cdo ); do

		echo
		echo $yr
		cdo -selname,${var} \
		${DATA_DIR}/${memb}/${memb}_${stream}_${yr}0101T000000Z.nc \
		${TEMP_DIR}/${memb}_${varname}_${yr}_mm.nc
	done


	#ls ${TEMP_DIR}/${memb}_${varname}_????.nc
	cdo cat ${TEMP_DIR}/${memb}_${varname}_????_mm.nc ${WORK_DIR}/${memb}_${varname}_1979-${last_year}_mm.nc


	# horizontal and verical remapping

	infile=${WORK_DIR}/${memb}_${varname}_1979-${last_year}_mm.nc
	outfile=${WORK_DIR}/${memb}_${varname}_remap63_1979-${last_year}_mm.nc

	if [ \( $stream = 'atm_3d_ml' \) -o \( $stream = 'phy_3d_ml' \) ]; then

		#remapdis: Distance weighted average remapping
		cdo -P 10 remapdis,t63grid -ap2pl,${pressure_lvls} $infile $outfile

	else

		cdo -P 10 remapdis,t63grid $infile $outfile

	fi

	echo

	if [ $tropical_mean = true ]; then

		lat=20
		lon1=0
 		lon2=360

		infile=${WORK_DIR}/${memb}_${varname}_remap63_1979-${last_year}_mm.nc
		outfile=${WORK_DIR}/${memb}_${varname}_remap63_${lat}latmean_${lon1}-${lon2}lonmean_1979-${last_year}_mm.nc

		cdo fldmean -sellonlatbox,${lon1},${lon2},-${lat},${lat} ${infile} ${outfile}

	fi
done
