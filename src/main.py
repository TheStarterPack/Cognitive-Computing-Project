from parsing import parser, actionSequence as AS
from models import word2Vec
import argparse

if __name__ == '__main__':
    # SETUP ARGUMENT PARSER
    argpar = argparse.ArgumentParser()
    argpar.add_argument("--epochs", default=30)
    args = argpar.parse_args()  

    # SETUP PARSER
    parser = parser.ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    action_to_id = parser.get_action_to_id_dict()

    # SETUP MODEL
    vocab_size = len(action_to_id)
    print(f"vocab size: {vocab_size}")
    model = word2Vec.CustomWord2Vec(vocab_size=vocab_size)
    model.configure_optimizer()

    # SETUP TRAIN DATA
    contexts, centers = AS.generate_contexts(action_sequences)
    np_contexts = AS.actions_to_tokenized_np_arrays(contexts, action_to_id)
    np_centers = AS.actions_to_tokenized_np_arrays(centers, action_to_id)
    data_loader = model.data_loader_from_numpy(np_centers.squeeze(), np_contexts)

    # TRAINING
    print(f"Start of Training for {args.epochs} epochs")
    model.train(data_loader, epochs=args.epochs)
    model.plot_logs(["loss"])
