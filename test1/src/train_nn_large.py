import train_nn

# train_net = train_nn.NNTraining(
#         learning_rate = 0.0001,
#         num_steps = 100000,
#         batch_size = 100,
#         display_step = 200,
#         save_iter = 1000,
#         datafile = "../data/training-data-large.txt.train.shuffle.txt",
#         dictfile = "../model/dict-large.npy",
#         model_dir = "../model/large-nn-1",
#         neurons=[512, 256])

train_net = train_nn.NNTraining(
        learning_rate = 0.0001,
        num_steps = 100000,
        batch_size = 100,
        display_step = 200,
        save_iter = 1000,
        datafile = "../data/training-data-large.txt.train.shuffle.txt",
        dictfile = "../model/dict-large-all.npy",
        model_dir = "../model/large-nn-2",
        neurons=[512, 256])

train_net.run()