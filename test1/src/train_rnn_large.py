import train_rnn

train_net = train_rnn.RNNTraining(
        learning_rate = 0.01,
        num_steps = 20000,
        batch_size = 128,
        display_step = 200,
        save_iter = 1000,
        datafile = "../data/training-data-large.txt.train.shuffle.txt",
        dictfile = "../model/dict-large.npy",
        model_dir = "../model/large-rnn-3/",
        series_len = 1000,
        num_hidden = 512,
        process_type="train",
        normalize_data= True,
        restore_file="../model/large-rnn-3/model5000.ckpt"
        )

train_net.run()