import train_nn

train_net = train_nn.NNTraining(
        learning_rate = 0.001,
        num_steps = 1000,
        batch_size = 128,
        display_step = 100,
        save_iter = 100,
        datafile = "../data/test-data-small.txt",
        dictfile = "../model/dict-small.npy",
        model_dir = "../model/small-nn-trained",
        restore_file= "../model/small-nn-trained/model1000.ckpt",
        process_type= "predict",
        predict_output= "../result/result-small.txt" )

train_net.run()