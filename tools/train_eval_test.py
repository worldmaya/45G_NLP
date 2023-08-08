# coding: UTF-8
import torch

def test(config, model, test_iter):
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(config.save_path))
    else:
        model.load_state_dict(torch.load(config.save_path,map_location='cpu'))
    model.eval()
    result = evaluate(config, model, test_iter)
    return result

def evaluate(config, model, data_iter):
    model.eval()
    class_list = [x.strip() for x in open(r'THUCNews/data/data/class.txt', encoding='utf-8').readlines()]
    with torch.no_grad():
        for batch in data_iter:
            batch = tuple(t.to(config.device) for t in batch)
            outputs = model(batch)
            outputs = torch.softmax(outputs,-1)#值和索引
    return outputs
