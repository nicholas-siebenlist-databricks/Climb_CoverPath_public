PROFILE ?= prod

.PHONY: deploy

deploy:
	databricks bundle validate --target dev --profile $(PROFILE) --output json \
	  | python3 scripts/gen_app_yaml.py
	databricks bundle deploy --target dev --profile $(PROFILE)
	databricks apps deploy coverpath \
	  --source-code-path /Workspace/Users/itai@climb.ai/.bundle/coverpath/dev/files/app \
	  --profile $(PROFILE)
