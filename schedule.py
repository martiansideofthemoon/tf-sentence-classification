import subprocess
base_script = \
"""#!/bin/bash

#SBATCH -p speech-gpu
#SBATCH -o {0}

cd /share/data/lang/users/kalpesh/constraint-learning
export LD_PRELOAD="/share/data/speech/Software/tcmalloc/lib/libtcmalloc.so"
/share/data/speech/Software/anaconda/bin/python \
/share/data/lang/users/kalpesh/constraint-learning/train.py \
--seed {1} \
--config_file {2} \
--job_id {3}
"""

seeds = [0,1,2,3,4,6,8]#[i for i in range(4, 10)]
modes = ["rand"]

for m in modes:
    for s in seeds:
        script = base_script.format("logs/%s_%s.log" % (m, s), s, "config/%s.yml" % m, m + "_" + str(s))
        with open('schedulers/%s_%s.sh' % (m, s), 'w') as f:
            f.write(script)
        command = "sbatch " + 'schedulers/%s_%s.sh' % (m, s)
        print(subprocess.check_output(command, shell=True))
