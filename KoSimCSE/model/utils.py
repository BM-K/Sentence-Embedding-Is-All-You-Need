import os
import torch
import logging
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
writer = SummaryWriter()


class Metric():

    def __init__(self, args):
        self.args = args

    def get_lr(self, optimizer):
        return optimizer.state_dict()['param_groups'][0]['lr']

    def count_parameters(self, model):
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def cal_acc(self, yhat, y):
        with torch.no_grad():
            yhat = yhat.max(dim=-1)[1]
            acc = (yhat == y).float().mean()

        return acc

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def cal_dev_score(self, score, indicator):
        validation_score = score['score'] / score['iter']
        for key, value in indicator.items():
            indicator[key] /= score['iter']

        print("\n\nCosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            indicator['eval_pearson_cosine'], indicator['eval_spearman_cosine']))
        print("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            indicator['eval_pearson_manhattan'], indicator['eval_spearman_manhattan']))
        print("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            indicator['eval_pearson_euclidean'], indicator['eval_spearman_euclidean']))
        print("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}\n".format(
            indicator['eval_pearson_dot'], indicator['eval_spearman_dot']))

        return validation_score

    def update_indicator(self, indicator, score):
        for key, value in indicator.items():
            if key == 'eval_spearman_cosine':
                indicator[key] += score['eval_spearman_cosine']
            elif key == 'eval_pearson_cosine':
                indicator[key] += score['eval_pearson_cosine']
            elif key == 'eval_spearman_manhattan':
                indicator[key] += score['eval_spearman_manhattan']
            elif key == 'eval_pearson_manhattan':
                indicator[key] += score['eval_pearson_manhattan']
            elif key == 'eval_spearman_euclidean':
                indicator[key] += score['eval_spearman_euclidean']
            elif key == 'eval_pearson_euclidean':
                indicator[key] += score['eval_pearson_euclidean']
            elif key == 'eval_spearman_dot':
                indicator[key] += score['eval_spearman_dot']
            elif key == 'eval_pearson_dot':
                indicator[key] += score['eval_pearson_dot']

    def draw_graph(self, cp):
        writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])
        writer.add_scalars('acc_graph', {'train': cp['tma'], 'valid': cp['vma']}, cp['ep'])

    def performance_check(self, cp, config):
        print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Train acc: {cp["tma"]:.4f}==')
        print(f'\t==Valid Loss: {cp["vl"]:.4f} | Valid acc: {cp["vma"]:.4f}==')
        print(f'\t==Epoch latest LR: {self.get_lr(config["optimizer"]):.9f}==\n')

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def move2device(self, sample, device):
        if len(sample) == 0:
            return {}

        def _move_to_device(maybe_tensor, device):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.to(device)
            elif isinstance(maybe_tensor, dict):
                return {
                    key: _move_to_device(value, device)
                    for key, value in maybe_tensor.items()
                    }
            elif isinstance(maybe_tensor, list):
                return [_move_to_device(x, device) for x in maybe_tensor]
            elif isinstance(maybe_tensor, tuple):
                return [_move_to_device(x, device) for x in maybe_tensor]
            else:
                return maybe_tensor

        return _move_to_device(sample, device)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        sorted_path = config['args'].path_to_save + "kosimcse-" + config['args'].model.replace("/", "-") + '.pt'
        if cp['vs'] > pco['best_valid_score']:
            pco['best_valid_score'] = cp['vs']

            state = {'model': config['model'].state_dict(),
                     'optimizer': config['optimizer'].state_dict()}

            torch.save(state, sorted_path)
            print(f'\t## SAVE {sorted_path} |'
                  f' valid_score: {cp["vs"]:.4f} |'
                  f' epochs: {cp["ep"]} |'
                  f' steps: {cp["step"]} ##\n')

def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))
