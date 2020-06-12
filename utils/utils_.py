import math
import time


def save_logfile(log_loss, save_path):
    with open(save_path, 'wt') as f:
        for k, v in log_loss.items():
            w_line = k
            for digit in v:
                w_line += ' %.3f' % digit
            f.write(w_line + '\n')


def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None, current_kld=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('epoch: %2d inner_iter: %3d' % (epoch, inner_iter), end=" ")
    message = '%s niter: %d completed: %3d%%)' % (time_since(start_time, niter_state / total_niters),
                                                niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.3f ' % (k, v)
    if current_kld is not None:
        message += ' current_kld_weight: %f' % (current_kld)
    print(message)
