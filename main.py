# public
from os.path import join, dirname
import importlib
import shutil
import traceback

# private
import trainer
from options import Options
options_handler = Options()
options = options_handler.parse()

if __name__ == "__main__":

    print('resume from {}'.format(options.resume))
    options = options_handler.update_opt_from_json(join(dirname(options.resume), 'flags.json'), options)

    tr = trainer.Trainer(options)
    print(tr.opt.phase, '-->', tr.opt.runsPath)
    tr.test()