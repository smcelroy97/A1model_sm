from netpyne.batchtools.search import search
import numpy as np

params = {'EEGain' : np.linspace(0.5, 1.5, 2),
          'IEGain' : np.linspace(0.5, 1.5, 2),
}

# use batch_shell_config if running directly on the machine
shell_config = {'command': 'mpiexec -np 4 nrniv -python -mpi init.py',}

run_config = shell_config

search(job_type = 'sh',
       comm_type = 'socket',
       label = 'grid',
       params = params,
       output_path = '../grid_batch',
       checkpoint_path = '../ray',
       run_config = run_config,
       num_samples = 1,
       metric = 'loss',
       mode = 'min',
       algorithm = "variant_generator",
       max_concurrent = 9)
