Setup (one-time)
# Install Rust: https://www.rust-lang.org/tools/install
pip install -r requirements.txt
./scripts/build_dl_manager.sh

Run the pipeline
python run_pipeline.py              # download from Jira + preprocess
python run_pipeline.py --skip-download   # reuse output/raw_issues.json
