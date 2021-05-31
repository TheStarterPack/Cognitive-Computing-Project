from src.parsing import Parser

if __name__ == '__main__':
    parser = Parser.ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    action_to_id = parser.get_action_to_id_dict()
