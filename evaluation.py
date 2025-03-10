import torch
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
            acc = correct_k.mul(100.0 / batch_size)
            res.append(acc)
        return res

def evaluate_model_quantized(model, data_loader, neval_batches=None):
    """Evaluates the model on the provided data loader"""
    model.eval()
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    cnt = 0

    with torch.no_grad():
        for image, target in tqdm(data_loader, desc="Evaluating model"):
            image, target = image.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
            output = model(image)
            cnt += 1

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), image.size(0))
            top5.update(acc5.item(), image.size(0))

            if neval_batches is not None and cnt >= neval_batches:
                break

    return f'top1: {top1.avg:.2f}, top5: {top5.avg:.2f}'
