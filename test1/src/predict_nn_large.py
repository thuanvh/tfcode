import train_nn

train_net = train_nn.NNTraining(
        batch_size = 128,
        datafile = "../data/test-data-large.txt",
        dictfile = "../model/dict-large.npy",
        model_dir = "../model/large-nn-1",
        restore_file= "../model/large-nn-1/model27000.ckpt",
        process_type= "predict",
        neurons=[512,256],
        predict_output= "../result/result-large.txt" )

train_net.run()