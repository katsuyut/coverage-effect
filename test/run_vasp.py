# import os
# exitcode = os.system('mpirun -np 16 vasp_std')

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler
from custodian.vasp.jobs import VaspJob

handlers = [VaspErrorHandler(), UnconvergedErrorHandler()]
jobs = [VaspJob(['mpirun -np 16 vasp_std'])]
c = Custodian(handlers, jobs, max_errors=10)
c.run()
exitcode = 0
