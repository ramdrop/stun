from os.path import dirname, join

import trainer
from options import Options

options_handler = Options()
options = options_handler.parse()

if __name__ == "__main__":

    if options.phase in ['test_tea', 'test_stu', 'train_stu']:
        print(f'resume from {options.resume}')
        options = options_handler.update_opt_from_json(join(dirname(options.resume), 'flags.json'), options)
        tr = trainer.Trainer(options)
        print(tr.opt.phase, '-->', tr.opt.runsPath)
    elif options.phase in ['train_tea']:
        tr = trainer.Trainer(options)
        print(tr.opt.phase, '-->', tr.opt.runsPath)

    if options.phase in ['train_tea']:
        tr.train()
    elif options.phase in ['train_stu']:
        tr.train_student()
    elif options.phase in ['test_tea', 'test_stu']:
        tr.test()