.PHONY: install synth diff bootstrap deploy destroy outputs

VENV := .venv
PY   := $(VENV)/bin/python
# `npx --yes` so the CDK CLI install prompt cannot block a non-interactive run.
CDK  := npx --yes aws-cdk@2

install:  ## Create the infra venv
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt

synth:  ## Render the CloudFormation template (no AWS calls beyond lookups)
	. $(VENV)/bin/activate && $(CDK) synth

diff:  ## What would change against the deployed stack
	. $(VENV)/bin/activate && $(CDK) diff

bootstrap:  ## One-time per account/region
	. $(VENV)/bin/activate && $(CDK) bootstrap

deploy:  ## Provision everything (builds and pushes the image)
	. $(VENV)/bin/activate && $(CDK) deploy --require-approval never

destroy:  ## Tear it all down - nothing is retained
	. $(VENV)/bin/activate && $(CDK) destroy --force

outputs:  ## Show the stack outputs, including the service URL
	aws cloudformation describe-stacks --stack-name EngramStack \
	  --query 'Stacks[0].Outputs' --output table
