from parsing import Parser, ActionSequence as AS
from models import Word2Vec

if __name__ == '__main__':
    #SETUP PARSER
    myparser = Parser.ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = myparser.read_action_seq_corpus()
    action_to_id = myparser.get_action_to_id_dict()

    # SETUP MODEL
    vocab_size = max(action_to_id.values())+1
    print("max token id + 1 -> vocab size:", vocab_size)
    model = Word2Vec.CustomWord2Vec(vocab_size=vocab_size)
    model.configure_optimizer()

    # SETUP TRAIN DATA
    contexts, centers = AS.generate_contexts(action_sequences)
    np_contexts = AS.actions_to_tokenized_np_arrays(contexts, action_to_id)
    np_centers = AS.actions_to_tokenized_np_arrays(centers, action_to_id)
    data_loader = model.data_loader_from_numpy(np_centers.squeeze(), np_contexts)

    # TRAINING
    model.train(data_loader)