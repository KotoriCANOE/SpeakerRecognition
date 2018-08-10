cd /d "%~dp0"

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
