from src.parsing import parser, actionSequence as AS
from src.models import word2Vec
import argparse
import os
from src.models.torchUtils import data_set_from_numpy
from src.models.torchUtils import write_embeddings_to_file
from torch.utils.data import random_split, DataLoader

if __name__ == '__main__':
    # SETUP ARGUMENT PARSER
    argpar = argparse.ArgumentParser()
    argpar.add_argument("--epochs", type=int, default=30)
    argpar.add_argument("--dims", type=int, default=64)
    argpar.add_argument("-noload", action="store_true")
    argpar.add_argument("-train", action="store_true")
    args = argpar.parse_args()

    # SETUP PARSER
    parser = parser.ActionSeqParser(include_augmented=False, include_default=True)
    action_sequences = parser.read_action_seq_corpus()
    action_to_id = parser.get_action_to_id_dict()

    # SETUP MODEL
    vocab_size = len(action_to_id)
    print(f"vocab size: {vocab_size}")
    model = word2Vec.CustomWord2Vec(vocab_size=vocab_size, dims=args.dims)
    loaded_model_flag = False
    if not args.noload:
        loaded_model_flag = model.load_model()
    model.configure_optimizer()

    # SETUP DATA
    contexts, centers = AS.generate_contexts(action_sequences)
    np_contexts = AS.actions_to_tokenized_np_arrays(contexts, action_to_id)
    np_centers = AS.actions_to_tokenized_np_arrays(centers, action_to_id)
    dataset = data_set_from_numpy(np_centers.squeeze(), np_contexts)
    train_counts = int(0.9 * len(dataset))
    trainset, testset = random_split(dataset, (train_counts, len(dataset) - train_counts))
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=True)

    # TRAINING
    if not loaded_model_flag or args.train:
        print(f"Start of Training for {args.epochs} epochs")
        model.train(train_loader, test_loader=test_loader, epochs=args.epochs)
        model.plot_logs(["loss"])

    write_embeddings_to_file(model, action_to_id, approach_name='action_target_embedding')
    idx_to_action = {v: k for k, v in action_to_id.items()}

    """
    # TESTING
    # TODO: way to get action from idxs
    # maybe juts this:?
    #print("action for idx 12", idx_to_action(12))

    # TODO then maybe something like this:
    vec12 = model.idx_to_center_vec(12)
    vec15 = model.idx_to_center_vec(15)
    delta_vec = vec12 - vec15

    print(model.get_most_similar_idxs(vec=delta_vec))
    print(model.get_most_similar_idxs(idx=12))
    print(model.get_most_similar_idxs(idx=15))

    # TODO clustering of embedding vectors?
    # cluster(model.centers)
    # cluster(model.contexts)
    """
