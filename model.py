import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def init_hidden(self,batch_size):
        pass 
                
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()  
        # save params
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # define layers
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden2out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, features, captions):
        batch_size = features.shape[0]
        captions = self.word_embeddings(captions[:,:-1])
        embed = torch.cat((features.unsqueeze(1), captions), dim=1)
        lstm_out, _ = self.lstm(embed)
        outputs = self.hidden2out(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        generatedcaption = []
        batch_size = inputs.shape[0]
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden) # 1,1,embedsize
            outputs = self.hidden2out(lstm_out)        # 1,1,vocab_size
            outputs = outputs.squeeze(1)                 # 1,vocab_size
            wordid  = outputs.argmax(dim=1)              # 1
            generatedcaption.append(wordid.item())
            inputs = self.word_embeddings(wordid.unsqueeze(0))  # 1,1->1,1,embed_size
            
        return generatedcaption