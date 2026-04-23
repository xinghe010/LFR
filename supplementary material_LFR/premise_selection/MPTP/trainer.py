import torch

def train(epoch, data_loader, model, optimizer, device, recorder, scaler=None):
    recorder.info('------starting {} epoch training------'.format(epoch))
    model.train()
    total = 0
    corrects = 0
    tp = fp = tn = fn = 0
    train_loss = 0.0
    for i, batch in enumerate(data_loader, 1):
        batch = batch.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(batch)
            loss.backward()
            optimizer.step()
        corrects += model.corrects
        tp += model.tp; fp += model.fp; tn += model.tn; fn += model.fn
        total += batch.y.size()[0]
        train_loss += loss.cpu().item()
    acc = corrects / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    log = "train epoch[{}] end! loss: {:.4f} acc: {:.2f}% f1: {:.2f}% recall: {:.2f}% prec: {:.2f}%".format(
        epoch, train_loss / i, acc * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return train_loss / i, acc, f1, recall, precision

def valid(epoch, data_loader, model, device, recorder):
    recorder.info('------starting {} epoch valid------'.format(epoch))
    model.eval()
    total = 0
    corrects = 0
    tp = fp = tn = fn = 0
    valid_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch = batch.to(device)
            loss = model(batch)
            corrects += model.corrects
            tp += model.tp; fp += model.fp; tn += model.tn; fn += model.fn
            total += batch.y.size()[0]
            valid_loss += loss.cpu().item()
    acc = corrects / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    log = "valid epoch[{}] end! loss: {:.4f} acc: {:.2f}% f1: {:.2f}% recall: {:.2f}% prec: {:.2f}%".format(
        epoch, valid_loss / i, acc * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return valid_loss / i, acc, f1, recall, precision

def test(data_loader, model, device, recorder):
    recorder.info('------starting test------')
    model.eval()
    total = 0
    corrects = 0
    tp = fp = tn = fn = 0
    test_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch = batch.to(device)
            loss = model(batch)
            corrects += model.corrects
            tp += model.tp; fp += model.fp; tn += model.tn; fn += model.fn
            total += batch.y.size()[0]
            test_loss += loss.cpu().item()
    acc = corrects / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    log = "test end! loss: {:.4f} acc: {:.2f}% f1: {:.2f}% recall: {:.2f}% prec: {:.2f}%".format(
        test_loss / i, acc * 100, f1 * 100, recall * 100, precision * 100)
    recorder.info(log)
    return test_loss / i, acc, f1, recall, precision
