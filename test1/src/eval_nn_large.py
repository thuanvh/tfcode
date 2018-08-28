import train_nn

# train_net = train_nn.NNTraining(        
#         batch_size = 128,
#         datafile = "../data/training-data-large.txt.test.txt",
#         dictfile = "../model/dict-large-all.npy",
#         model_dir = "../model/large-nn-2",
#         restore_file= "../model/large-nn-2/model13000.ckpt",
#         process_type= "eval",
#         neurons=[512, 256] )


# train_net = train_nn.NNTraining(        
#         batch_size = 128,
#         datafile = "../data/training-data-large.txt.test.txt",
#         dictfile = "../model/dict-large.npy",
#         model_dir = "../model/large-nn-1",
#         restore_file= "../model/large-nn-1/model27000.ckpt",
#         process_type= "eval",
#         neurons=[512, 256] )

train_net = train_nn.NNTraining(        
        batch_size = 128,
        datafile = "../data/training-data-large.txt.test.txt",
        dictfile = "../model/dict-large.npy",
        model_dir = "../model/large-nn",
        restore_file= "../model/large-nn/model9000.ckpt",
        process_type= "eval",
        neurons=[256, 256] )



train_net.run()