PROFILE ?= prod
TARGET  ?= dev

.PHONY: deploy

deploy:
	databricks bundle validate --target $(TARGET) --profile $(PROFILE) --output json \
	  | tee /tmp/coverpath-validate.json \
	  | python3 scripts/gen_app_yaml.py
	databricks bundle deploy --target $(TARGET) --profile $(PROFILE)
	databricks apps deploy coverpath \
	  --source-code-path "$$(python3 -c 'import json; print(json.load(open("/tmp/coverpath-validate.json"))["workspace"]["file_path"] + "/app")')" \
	  --profile $(PROFILE)
