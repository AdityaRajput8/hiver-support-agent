.PHONY: setup data ingest cluster index baselines eval demo clean

setup:
	python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
	cp .env.example .env

data:
	python src/ingest/filter_brand.py --input data/raw/twcs.csv --brand Uber_Support --out data/processed/uber_subset.csv
	python src/ingest/thread_builder.py --input data/processed/uber_subset.csv --out data/processed/uber_threads.parquet

cluster:
	python src/intents/cluster_explore.py --input data/processed/uber_threads.parquet --k 10

index:
	python src/retrieval/build_index.py

baselines:
	python src/classify/baseline_tfidf.py --train

golden:
	python eval/build_golden_sample.py --n 200

eval:
	python eval/run_eval.py

demo:
	streamlit run app/streamlit_demo.py

clean:
	rm -rf data/processed/* data/golden/*