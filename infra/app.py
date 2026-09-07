#!/usr/bin/env python3
"""CDK entry point.

    cd infra && make deploy      # after `make install` and a one-time bootstrap
    cd infra && make destroy     # leaves nothing behind

Model ids are context-overridable so a deploy can be pointed at a different
Bedrock model without editing the stack:

    npx aws-cdk@2 deploy -c chat_model=amazon.nova-pro-v1:0
"""

from __future__ import annotations

import os

import aws_cdk as cdk

from engram_infra.stack import EngramStack

app = cdk.App()

EngramStack(
    app,
    "EngramStack",
    chat_model=app.node.try_get_context("chat_model") or "amazon.nova-lite-v1:0",
    reasoning_model=app.node.try_get_context("reasoning_model") or "amazon.nova-pro-v1:0",
    embed_model=app.node.try_get_context("embed_model") or "amazon.titan-embed-text-v2:0",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
    ),
    description="engram - long-term memory for conversational agents",
)

app.synth()
