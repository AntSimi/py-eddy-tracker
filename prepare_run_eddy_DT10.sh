#!/bin/bash

ROOT_DIR="/work/sentinel3/pva_axp/Eddy_Tracking/"
#DATA_DIR="${ROOT_DIR}/data/sla/DT10/"
DATA_DIR="${ROOT_DIR}/data/sla/DT10/test/"
INSTALL_DIR="${ROOT_DIR}/process"
yaml_template="eddy_tracker_configuration_DT10.yaml"

echo "Looking at data file in ${DATA_DIR}"
echo "Template yaml configuration file is: ${yaml_template}"
# Prepare the list of yaml files
i=0
for file in $(ls ${DATA_DIR}); do
    i=$(($i+1))
    filename=$(basename ${file})
    datestr=${file:31:8}
    outyaml="${INSTALL_DIR}/production/sh/eddy_tracker_configuration_${datestr}.yaml"
    echo "Preparing yaml file for date ${datestr}: ${outyaml}"
    sed "s/XXXXXXXX/${datestr}/g" ${yaml_template} > ${outyaml}
    outsh="production/sh/eddy_job_${i}.sh"
    cmd="python -u ${INSTALL_DIR}/make_eddy_track_AVISO.py ${outyaml}"
    echo "#!/bin/bash" > ${outsh}
    echo "${cmd}" >> ${outsh}
    #echo "echo ${outyaml}" >> ${outsh}
    chmod u+x ${outsh}
done

echo "Found ${i} files: preparing array of ${i} jobs"

# Prepare  the PBS script

cat > run_eddy.pbs << EOF 

#!/bin/bash
#PBS -N pbs_eddy
#PBS -J 1-${i}
#PBS -l select=1:ncpus=1:mem=16000mb 
#PBS -l walltime=24:00:00
#PBS -m bea
#PBS -M evanmason@gmail.com

# repertoire d execution
module load python/2.7.5
cd /work/sentinel3/pva_axp/Eddy_Tracking/process

/work/sentinel3/pva_axp/Eddy_Tracking/process/production/sh/eddy_job_\${PBS_ARRAY_INDEX}.sh

EOF

echo "wrote run_eddy.pbs".
echo "Run command: qsub run_eddy.pbs"


    




 
