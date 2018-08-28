import train_rnn

train_net = train_rnn.RNNTraining(        
        batch_size = 128,       
        series_len = 130,
        num_hidden = 128, 
        datafile = "../data/training-data-small.txt.test.txt",
        dictfile = "../model/dict-small.npy",
        model_dir = "../model/small-rnn-3",
        restore_file= "../model/small-rnn-3/model6200.ckpt",
        process_type= "eval" )


# train_net = train_rnn.RNNTraining(        
#         batch_size = 128,
#         datafile = "../data/training-data-small.txt.train.shuffle.txt",
#         dictfile = "../model/dict-small.npy",
#         model_dir = "../model/small-rnn-3/",
#         series_len = 130,
#         num_hidden = 128,
#         process_type="train",
#         restore_file="../model/small-rnn-3/model10000.ckpt")

train_net.run()