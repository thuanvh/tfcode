python freeze_graph.py --input_graph=model_100_3/graph.pbtxt --input_checkpoint=model_100_3/model20000.ckpt --input_binary=false --output_graph=bin/model.pb --output_node_names=fc2/add 
