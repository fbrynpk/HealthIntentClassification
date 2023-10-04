import torch
import transformers
import speech2text

from torch import nn
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Setup class names
with open("class_names.txt", "r") as f:
    labels = [prompt.strip() for prompt in f.readlines()]


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, len(labels))

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)

        return linear_output


def predict(model, text):
    text.lower()
    text_dict = tokenizer(
        text, padding="max_length", max_length=20, truncation=True, return_tensors="pt"
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    mask = text_dict["attention_mask"].to(device)
    input_id = text_dict["input_ids"].squeeze(1).to(device)

    model.eval()
    with torch.inference_mode():
        output = model(input_id, mask)
        label_id = output.argmax(dim=1).item()
        return label_id


model = BertClassifier()
model.load_state_dict(torch.load("./models/FirstAidClassifier.pth"))

prediction = predict(model, text=speech2text.transcript)
print(
    f"You might be experiencing symptoms for {labels[prediction]}, please seek some medical help!"
)
