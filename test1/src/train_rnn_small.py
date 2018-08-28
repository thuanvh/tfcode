import train_rnn

train_net = train_rnn.RNNTraining(
        learning_rate = 0.0001,
        num_steps = 10000,
        batch_size = 128,
        display_step = 100,
        save_iter = 100,
        datafile = "../data/training-data-small.txt.train.shuffle.txt",
        dictfile = "../model/dict-small.npy",
        model_dir = "../model/small-rnn-3/",
        series_len = 130,
        num_hidden = 128,
        normalize_data= True,
        process_type="train",
        #restore_file="../model/small-rnn-3/model10000.ckpt"
        )

train_net.run()