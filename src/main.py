from src.parsing import Parser

if __name__ == '__main__':
    parser = Parser.ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    tokenization = parser.get_tokenization()
    print(len(tokenization))
