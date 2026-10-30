import torch
import pandas as pd

def train(epoch, data_loader, model, optimizer, device, recorder):
    recorder.info('------starting {} epoch training------'.format(epoch))
    model.train()
    total = 0
    corrects = 0
    train_loss = 0.0

    for i, batch in enumerate(data_loader, 1):
        batch = batch.to(device)
        optimizer.zero_grad()

        loss = model(batch)
        corrects += model.corrects
        loss.backward()
        optimizer.step()
        total += batch.y.size()[0]
        train_loss += loss.cpu().item()
    log = "train epoch[{}] end! train loss: {:.4f} train accuarcy: {:.2f}%".format(
        epoch, train_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return train_loss / i, corrects / total

def valid(epoch, data_loader, model, device, recorder):
    recorder.info('------starting {} epoch valid------'.format(epoch))
    model.eval()
    total = 0
    corrects = 0
    valid_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch = batch.to(device)
            loss = model(batch)
            corrects += model.corrects
            total += batch.y.size()[0]
            valid_loss += loss.cpu().item()

    log = "valid epoch[{}] end! valid loss: {:.4f} valid accuarcy: {:.2f}%".format(
        epoch, valid_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return valid_loss / i, corrects / total

def test(data_loader, model, device, recorder):
    recorder.info('------starting test------')
    model.eval()
    total = 0
    corrects = 0
    test_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch = batch.to(device)
            loss = model(batch)
            corrects += model.corrects
            total += batch.y.size()[0]
            test_loss += loss.cpu().item()
    log = "test end! test loss: {:.4f} test accuarcy: {:.2f}%".format(
        test_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return test_loss / i, corrects / total

def test_new(data_loader, model, device, recorder):
    recorder.info('------starting test------')
    model.eval()
    total = 0
    corrects = 0
    test_loss = 0.0

    tp, fp, fn, tn = 0, 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader, 1):
            batch = batch.to(device)
            loss = model(batch)
            corrects += model.corrects

            tp += model.tp
            fp += model.fp
            tn += model.tn
            fn += model.fn

            total += batch.y.size()[0]
            test_loss += loss.cpu().item()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    log = "test end! test loss: {:.4f} test accuarcy: {:.2f}% test recall: {:.2f}% test precison: {:.2f}% " \
          "test F1: {:.2f}%".format(
        test_loss / i, (corrects / total) * 100, recall * 100, precision * 100, f1 * 100)
    recorder.info(log)

    return test_loss / i, corrects / total, recall, precision, f1

def test2(train_loader, test_loader, model, device, recorder):
    recorder.info('------starting test------')
    model.eval()
    total = 0
    corrects = 0
    test_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            batch = batch.to(device)
            loss = model(batch)
            corrects += model.corrects
            total += batch.y.size()[0]
            test_loss += loss.cpu().item()

            g_s = model.initial(batch.x_s)
            g_t = model.initial(batch.x_t)
            g_s = model.dag_emb(g_s, batch.term_walk_index_s)
            g_t = model.dag_emb(g_t, batch.term_walk_index_t)
            g_g_s = model.pooling(g_s, batch.x_s_batch)
            g_g_t = model.pooling(g_t, batch.x_t_batch)
            x = torch.cat((g_g_s, g_g_t), 1)
            df1 = pd.DataFrame(x.detach().cpu().numpy())
            df1.to_csv('mydata\\test.data', mode='a', index=False, header=False)
            df2 = pd.DataFrame(batch.y.detach().cpu().numpy())
            df2.to_csv('mydata\\test_y.data', mode='a', index=False, header=False)

    with torch.no_grad():
        for j, batch in enumerate(train_loader, 1):
            batch = batch.to(device)

            g_s = model.initial(batch.x_s)
            g_t = model.initial(batch.x_t)
            g_s = model.dag_emb(g_s, batch.term_walk_index_s)
            g_t = model.dag_emb(g_t, batch.term_walk_index_t)
            g_g_s = model.pooling(g_s, batch.x_s_batch)
            g_g_t = model.pooling(g_t, batch.x_t_batch)
            x = torch.cat((g_g_s, g_g_t), 1)
            df1 = pd.DataFrame(x.detach().cpu().numpy())
            df1.to_csv('mydata\\train.data', mode='a', index=False, header=False)
            df2 = pd.DataFrame(batch.y.detach().cpu().numpy())
            df2.to_csv('mydata\\train_y.data', mode='a', index=False, header=False)

    log = "test end! test loss: {:.4f} test accuarcy: {:.2f}%".format(
        test_loss / i, (corrects / total) * 100)
    recorder.info(log)
    return test_loss / i, corrects / total
