cd /d "%~dp0"

FOR %%i IN (152) DO (
	python graph.py --postfix %%i --num-labels 5994 --embed-size 512 --normalization None
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Embedding
)

pause

python graph.py --out-channels 5994 --postfix 129 --embed-size 512
python graph.py --out-channels 5994 --postfix 130 --embed-size 512
python graph.py --out-channels 5994 --postfix 131 --embed-size 64
python graph.py --out-channels 5994 --postfix 132 --embed-size 64

FOR %%i IN (147) DO (
	python graph.py --out-channels 5994 --postfix %%i --embed-size 64
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Embedding
)
FOR %%i IN (148) DO (
	python graph.py --out-channels 5994 --postfix %%i --embed-size 64
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Embedding
)

FOR %%i IN (149) DO (
	python graph.py --out-channels 5994 --postfix %%i --embed-size 512
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Embedding
)
FOR %%i IN (150) DO (
	python graph.py --out-channels 5994 --postfix %%i --embed-size 4096 --model-file model_0250000
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Embedding
)
FOR %%i IN (201) DO (
	python graph.py --out-channels 1022 --postfix %%i --embed-size 512
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Embedding
)

FOR %%i IN (153) DO (
	python graph.py --postfix %%i --num-labels 5994 --embed-size 512 --normalization None --model-file model_0510000
	python freeze_graph.py --input_graph model%%i.tmp\model.graphdef --input_checkpoint model%%i.tmp\model --output_graph model%%i.tmp\model.pb --output_node_names Embedding
)
