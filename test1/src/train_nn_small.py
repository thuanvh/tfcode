import train_nn

train_net = train_nn.NNTraining(
        learning_rate = 0.0001,
        num_steps = 10000,
        batch_size = 128,
        display_step = 100,
        save_iter = 100,
        datafile = "../data/training-data-small.txt.train.shuffle.txt",
        dictfile = "../model/dict-small.npy",
        model_dir = "../model/small-nn-3",
        process_type="train")

train_net.run()