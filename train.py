import os
import numpy as np
import torch
import torch.nn.functional as F
import time

from base_container import BaseContainer
from utils.summary import TensorboardSummary
from utils.logger import logger, set_logger_path
from utils.utils import Saver
from utils.utils import acc_cal
from tqdm import tqdm
import queue


class Trainer(BaseContainer):
    def __init__(self):
        super().__init__()
        now_time = time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time()))
        logger_path = os.path.join(
            self.args.training.save_dir,
            self.args.logging.log_dir,
            '%s.log' % now_time
        )
        set_logger_path(self.args, logger_path)
        self.saver = Saver(self.args)
        self.summary = TensorboardSummary()
        self.writer = self.summary.create_summary(self.saver.directory, self.args.models)

        self.init_training_container()
        self.batchsize = self.args.training.batch_size
        self.acc_cal = acc_cal()
        self.best = 0.0
        self.q_maxsize = 5
        self.q = queue.Queue(self.q_maxsize)
        self.halt = False

        logger.debug('\n TRAINING PARAMS: ')
        for p in self.model.named_parameters():
            if p[1].requires_grad:
                logger.debug(p[0])
        logger.debug('\n')

    def training(self):
        self.model.train()
        logger.info('\n THE MODEL START TRAINGING!')

        max_iter = self.args.training.max_iter
        it = self.start_it

        while not self.halt:
            for samples in tqdm(self.train_loader):
                torch.cuda.empty_cache()
                samples = to_cuda(samples)

                # VALIDATION
                val_iter = self.args.validation.get('val_iter', -1)
                if val_iter > 0 and it % val_iter == 0 and it >= self.args.validation.get('start_val_iter', 15000):
                    self.validation(it, 'val')
                    self.model.train()

                if it % 100 == 0:
                    logger.info('\n===> Iteration  %d/%d' % (it, max_iter))

                if it >= self.args.training.warm_up and self.args.training.get('weight_update_iter', -1) > 0 and it % self.args.training.get('weight_update_iter', -1) == 0:
                    self.model.eqm.flush()
                    logger.info('\n WEIGHTS HAVE BEEN REFRESH!')

                self.optimizer.zero_grad()
                outputs = self.model(samples, iter=it)
                losses = self.criterion(outputs)
                loss = losses['loss']
                loss.backward()
                self.optimizer.step()

                # LOG TRAINGING LOSS
                if it % 100 == 0:
                    loss_log_str = '=>LOSS: %.4f  |  ' % (loss.item())
                    for loss_name in losses.keys():
                        if loss_name != 'loss':
                            if loss_name.startswith("trip"):
                                if losses[loss_name] == None:
                                    loss_log_str += '%s: None' % (loss_name)
                                else:
                                    loss_log_str += '%s: %.4f' % (loss_name, losses[loss_name])
                            else:
                                loss_log_str += '%s: %.4f' % (loss_name, losses[loss_name])
                                self.writer.add_scalar('TRAINING/%s_iter' % loss_name, losses[loss_name], it)
                    logger.info(loss_log_str)
                    self.writer.add_scalar('TRAINING:TOTAL_LOSS', loss.item(), it)

                # LEARNING RATE
                lr_decay_iter = self.args.training.optimizer.get('lr_decay_iter', None)
                if lr_decay_iter is not None:
                    for i in range(len(lr_decay_iter)):
                        if it == lr_decay_iter[i]:
                            lr = self.args.training.optimizer.lr * (self.args.training.optimizer.lr_decay ** (i+1))
                            logger.info('\nTHE LEARNING RATE BEGIN TO DROP TO %.6f\n' % (lr))
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = lr
                            break

                it += 1

                # SAVE THE MODEL AND OPTIMIZER PARAMETERS
                if it % self.args.training.save_iter == 0 or it == max_iter or it == 1:
                    logger.info('\n MODEL IS BEING SAVED AUTOMATICALLY. Processing...')
                    optimizer_to_save = self.optimizer.state_dict()
                    self.saver.save_checkpoint({
                        'start_it': it,
                        'stage': self.stage,
                        'state_dict': self.model.state_dict(),
                        'optimizer': optimizer_to_save,
                    }, filename = self.args.training.weights_filename + '%06d.pth.tar' % it)
                    logger.info('Done.')

    def validation(self, it):
        logger.info('\n MODEL VALIDATION IN PROGRESS.')
        self.acc_cal.reset()
        self.model.eval()

        data_loader = self.val_loader
        dist_pos = []
        dist_neg = []
        for i, samples in enumerate(tqdm(data_loader)):
            samples = to_cuda(samples)

            with torch.no_grad():
                outputs = self.model(samples, is_triple=True)
                dist_pos.append(outputs[-1]['dist_pos'].mean().item())
                dist_neg.append(outputs[-1]['dist_neg'].mean().item())

            self.acc_cal.add_batch(outputs[-1]['pred'], outputs[-1]['target'])

        self.writer.add_scalar('VALIDATION/DIST_POS', np.array(dist_pos).mean(), it)
        self.writer.add_scalar('VALIDATION/DIST_NEG', np.array(dist_neg).mean(), it)

        acc = self.acc_cal.get_accuracy()
        self.writer.add_scalar('VALIDATION/ACC', acc, it)
        logger.info('+++++>[ITERATION: %d    VALIDATION/ACC = %.4f    OPTIMAL ACC = %.4f' % (it, acc, self.best))

        if acc > self.best:
            self.best = acc
            logger.info('\n SAVING THE CURRENT OPTIMAL MODEL.')
            optimizer_to_save = self.optimizer.state_dict()
            weight_filename = self.args.validation.optimal_weights_filename + '{}.pth.tar'.format(it)
            self.saver.save_checkpoint({
                'start_it': it,
                'stage': self.stage,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer_to_save,
            }, filename = weight_filename)

        if not self.q.full():
            self.q.put(acc)
        else:
            self.q.get()
            self.q.put(acc)

        if it > self.args.training.max_iter and self.q.full():
            accs = []
            for i in range(self.q_maxsize):
                accs.append(self.q.get())
            for i in range(self.q_maxsize):
                self.q.put(accs[i])

            accs = np.array(accs)
            accs_mismatch = np.delete(accs, 0)
            accs_mismatch = np.append(accs_mismatch, 0)
            inter_diff = accs - accs_mismatch
            inter_diff = np.delete(inter_diff, -1)
            ascend_n = np.count_nonzero(inter_diff < 0)

            max_index = np.argmax(accs)
            if max_index == 0 and ascend_n == 0:
                self.halt = True

def to_cuda(sample):
    if isinstance(sample, list):
        return [to_cuda(i) for i in sample]
    elif isinstance(sample, dict):
        for key in sample.keys():
            sample[key] = to_cuda(sample[key])
        return sample
    elif isinstance(sample, torch.Tensor):
        return sample.cuda()
    else:
        return sample

def main():
    trainer = Trainer()
    trainer.training()
    trainer.writer.close()

if __name__ == "__main__":
    main()
