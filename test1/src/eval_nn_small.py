import train_nn

train_net = train_nn.NNTraining(
        learning_rate = 0.001,
        num_steps = 1000,
        batch_size = 128,
        display_step = 100,
        save_iter = 100,
        datafile = "../data/training-data-small.txt.valid.txt",
        dictfile = "../model/dict-small.npy",
        model_dir = "../model/small-nn-2",
        restore_file= "../model/small-nn-2/model3000.ckpt",
        process_type= "eval" )

train_net.run()