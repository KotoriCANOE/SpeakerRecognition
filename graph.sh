postfix=153
python graph.py --postfix $postfix --num-labels 5994 --embed-size 512 --normalization None
python freeze_graph.py --input_binary False --input_graph model$postfix.tmp/model.graphdef --input_checkpoint model$postfix.tmp/model --output_graph model$postfix.tmp/model.pb --output_node_names Embedding

exit
