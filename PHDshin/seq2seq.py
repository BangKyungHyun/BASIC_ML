import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import re
import pickle
import pandas as pd

#하이퍼 파라미터
hidden_size = 256
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
MAX_LENGTH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text):
    if pd.isna(text):                          # NaN값을 처리
        return ''
    text = text.lower()                        # 소문자로 변경
    text = re.sub(r'\d+', ' ', text)           # 숫자를 공백으로
    text = re.sub(r'([^\w\s])', r' \1 ', text) # 특수문자(마침표) 앞 뒤로 공백 추가
    text = re.sub(r'\s+', ' ', text)           # 두개 이상의 공백은 하나의 공백으로
    text = text.strip()                        # 텍스트 앞 뒤의 공백 제거
    return text

def indexesFromSentence(vocab, sentence):
    return [vocab.get(word, vocab['<UNK>']) for word in sentence.split(' ')]

def tensorFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size, device=device),
                torch.zeros(2, 1, self.hidden_size, device=device))

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return (torch.zeros(2, 1, self.hidden_size, device=device),
                torch.zeros(2, 1, self.hidden_size, device=device))


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()  # backpropagation only 1 line!

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# 학습을 반복해 주는 코드

def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
    print_loss_total = 0

    for iter in range(1, n_iters + 1):
        training_pair = random.choice(pairs)
        input_tensor = tensorFromSentence(word_to_ix, training_pair[0]).to(device)
        target_tensor = tensorFromSentence(word_to_ix, training_pair[1]).to(device)

        loss = train(input_tensor, target_tensor, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print(f'Iteration: {iter}, Loss: {print_loss_avg: .4f}')
            print_loss_total = 0


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(word_to_ix, sentence).to(device)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()
        encoder_hidden = tuple([e.to(device) for e in encoder_hidden])

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoded_words = []  # output sentence

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(ix_to_word[topi.item()])  # 여기는 최종 아웃풋의 인덱스가 들어갑니다
            decoder_input = topi.squeeze().detach()
        return ' '.join(decoded_words)

# 채팅함수
def chat(encoder, decoder, max_length=MAX_LENGTH):
    print("Let's chat! (type 'bye' to exit)")
    while True:
        input_sentence = input("> ")
        if input_sentence == 'bye':
            break
        output_sentence = evaluate(encoder, decoder, input_sentence)
        print('<', output_sentence)

# 데이터 로드 및 기본 전처리 부분을..
df = pd.read_csv('./data/chatbot_dataset.txt', sep='\t', names=['Question', 'Answer'])
df['Encoder Inputs'] = df['Question'].apply(clean_text)
df['Decoder Inputs'] = df['Answer'].apply(clean_text)

df['Decoder Inputs']

input_sentence = [sentence for sentence in df['Encoder Inputs']]
output_sentence = [sentence + "<EOS>" for sentence in df['Decoder Inputs']]

input_sentence[0:5]

output_sentence[0:5]

# 단어 사전 생성
all_words = set(' '.join(df['Encoder Inputs'].tolist()+df['Decoder Inputs'].tolist()).split())
vocab = {'<PAD>': PAD_token, '<SOS>': SOS_token, '<EOS>': EOS_token, '<UNK>': UNK_token}
vocab.update({word: i+4 for i, word in enumerate(all_words)})
vocab_size = len(vocab)
# vocab 변수 저장
with open('./data/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

word_to_ix = vocab
ix_to_word = {i: word for word, i in word_to_ix.items()}

word_to_ix['hello']

ix_to_word[167]

encoder = EncoderLSTM(vocab_size, hidden_size).to(device)
decoder = DecoderLSTM(hidden_size, vocab_size).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.005)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# pairs 리스트를 만들어서 학습 데이터를 준비
pairs = [list(x) for x in zip(df['Encoder Inputs'], df['Decoder Inputs'])]

pairs[1]

#학습실행 def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
trainIters(encoder, decoder, 30000, print_every=1000)

torch.save(encoder.state_dict(), './models/encoder_tmp.pth')
torch.save(decoder.state_dict(), './models/decoder_tmp.pth')

# 평가실행
encoder.eval()
decoder.eval()

chat(encoder, decoder)

# vocab 변수 로드
with open('./data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)
word_to_ix = vocab
ix_to_word = {i: word for word, i in word_to_ix.items()}
encoder = EncoderLSTM(vocab_size, hidden_size).to(device)
decoder = DecoderLSTM(hidden_size, vocab_size).to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.005)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.005)