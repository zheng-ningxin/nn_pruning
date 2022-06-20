import torch
from transformers import Wav2Vec2FeatureExtractor, HubertForSequenceClassification
from transformers.optimization import AdamW
from datasets import load_dataset
import soundfile as sf
from torchaudio.sox_effects import apply_effects_file

# effects = [["channels", "1"], ["rate", "16000"], ["gain", "-3.0"]]
# def map_to_array(example):
#     speech, _ = apply_effects_file(example["file"], effects)
#     example["speech"] = speech.squeeze(0).numpy()
#     return example

processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")
model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks").cuda()

ds_train = load_dataset("superb", "ks", split="train")
ds_test = load_dataset("superb", "ks", split="test")

def map_to_array(example):
    speech_array, sample_rate = sf.read(example["file"])
    example["speech"] = speech_array
    example["sample_rate"] = sample_rate
    return example

ds_train = ds_train.map(map_to_array)
ds_test = ds_test.map(map_to_array)
import pdb; pdb.set_trace()
# import ipython
# ipython.embed()

def evaluate(model, ds):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, speech in enumerate(ds["speech"]):
            input_values = processor(speech, sampling_rate=16000, padding=True, return_tensors='pt').input_values.cuda()
            logits = model(input_values).logits
            predicted_class_ids = torch.argmax(logits, dim=-1)
            if predicted_class_ids == ds['label'][i]:
                correct += 1
        print('ACC: ', correct/len(ds['label']))
# evaluate(model, ds_test)

def finetune(model, ds, batchsize=32, learning_rate=1e-3, epochs=10):
    loss_func = torch.nn.CrossEntropyLoss()

    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    # import pdb; pdb.set_trace()
    dataset_size = len(ds['label']) # 51094
    for epoch in range(epochs):
        model.train()
        for i, speech in enumerate(ds["speech"]):
            optimizer.zero_grad()
            inputs = processor(speech, sampling_rate=16000, padding=True, return_tensors="pt", max_length=160000).input_values.cuda()
            import pdb; pdb.set_trace()

            logits = model(inputs).logits
            loss = loss_func(logits, ds["label"][i])
            print('Loss', loss)
            loss.backward()
            optimizer.step()
if __name__ == '__main__':
    pass
    evaluate(model, ds_test)
    finetune(model, ds_train, 16)
    import pdb; pdb.set_trace()
# loss = model(input_values, labels=labels).loss