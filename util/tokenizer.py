import re
from typing import List, Union

# Complete punctuation from string.punctuation: !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~

class Tokenizer:
    NON_ASCII_REGEX = re.compile(r"[^\x00-\x7F\u2013]")
    PUNCTUATIONS = '!"#$%&()*+/;<=>@?[\\]^_`{|}~'
    REAL_SEPARATOR = "'.,:"
    PUNCTUATIONS_REGEX = re.compile(fr"([{PUNCTUATIONS}])")
    REAL_SEPARATOR_REGEX = re.compile(fr"(([{REAL_SEPARATOR}][^a-zA-Z0-9])|([{REAL_SEPARATOR}]$))")
    
    def __init__(self, max_vocab=50000, lower=False, normalize=False, remove_non_ascii=False,
                 pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
        if normalize and remove_non_ascii:
            raise ValueError('You can only choose between normalize/remove_non_ascii!')
        self.max_vocab = max_vocab
        self.lower = lower
        self.normalize = normalize
        self.remove_non_ascii = remove_non_ascii
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.add_words([pad_token, unk_token, bos_token, eos_token])
            
    def train_tokenizer(self, text: Union[str, List[str]]):
        tokenized_text = []
        if isinstance(text, str):
            tokenized_text.extend(self.tokenize(text))
        elif isinstance(text, list):
            for sentence in text:
                tokenized_text.extend(self.tokenize(sentence))
                
        self.add_words(tokenized_text)
        self.word2count = {k: v for k, v in sorted(self.word2count.items(), key=lambda item: item[1], reverse=True)}
        if self.n_words > self.max_vocab:
            print(f'Least frequent words will be removed until n_vocab = {self.max_vocab} (excluding special tokens)')
            words_to_remove = list(self.word2count.keys())[self.max_vocab:]
            self.remove_words(words_to_remove, restructure_index=True)
        
    def encode(self, texts:Union[List[List[str]], List[str]], use_eos_bos_token=False):
        list_of_tokens = []
        for text in texts:
            tokenized_text = self.tokenize(text) if isinstance(text, str) else text
            encoded_token = [self.word2index.get(token) or self.word2index[self.unk_token] for token in tokenized_text]
            list_of_tokens.append(
                encoded_token if not use_eos_bos_token
                else [self.word2index[self.bos_token]] + encoded_token + [self.word2index[self.eos_token]]
            )
        return list_of_tokens
            
    def decode(self, list_of_tokens:List[List[int]], to_string=False):
        output = []
        for tokens in list_of_tokens:
            tokens = [self.index2word[token] for token in tokens]
            if to_string:
                output.append(' '.join(tokens))
            else:
                output.append(tokens)
        return output
    
    def tokenize(self, s):
        if self.lower:
            s = s.lower()
        if self.normalize:
            s = self._unicode_to_ascii(s)
        elif self.remove_non_ascii:
            s = _remove_non_ascii(s)
        s = re.sub(self.PUNCTUATIONS_REGEX, r" \1 ", s)
        s = re.sub(self.REAL_SEPARATOR_REGEX, r" \1", s)
        s = s.split()
        return s

    def add_words(self, list_of_words):
        for word in list_of_words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def remove_words(self, list_of_words, restructure_index=False):
        for word in list_of_words:
            self.remove_word(word)
        if restructure_index:
            self._restructure_index()

    def remove_word(self, word, restructure_index=False):
        if word not in self.word2index:
            raise ValueError(f'{word} does not exist in the dictionary.')
        elif word in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            pass
        else:
            del self.index2word[self.word2index[word]]
            del self.word2index[word]
            del self.word2count[word]
            self.n_words -= 1
        if restructure_index:
            self._restructure_index()
        
    def _unicode_to_ascii(self, s):
        return unidecode(s)

    def _remove_non_ascii(self, s):
        return re.sub(self.NON_ASCII_REGEX, r"", s)

    def _restructure_index(self):
        self.index2word = {}
        i = 0
        for word in self.word2index:
            self.index2word[i] = word
            self.word2index[word] = i            
            i += 1
        assert i == self.n_words
        