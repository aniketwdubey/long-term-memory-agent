"""Offline template checks; run with `make -C infra test`."""

from __future__ import annotations

import tempfile
import unittest

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template
from engram_infra.stack import EngramStack


class AuthenticationStackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = tempfile.TemporaryDirectory()
        cls.app = cdk.App(
            outdir=cls.output.name,
            context={
                "availability-zones:account=111111111111:region=us-east-1": [
                    "us-east-1a",
                    "us-east-1b",
                ]
            },
        )
        cls.stack = EngramStack(
            cls.app,
            "EngramStack",
            env=cdk.Environment(account="111111111111", region="us-east-1"),
        )
        cls.template = Template.from_stack(cls.stack)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.output.cleanup()

    def test_accounts_require_verified_email_and_strong_passwords(self) -> None:
        self.template.has_resource_properties(
            "AWS::Cognito::UserPool",
            {
                "AutoVerifiedAttributes": ["email"],
                "UsernameAttributes": ["email"],
                "Policies": {"PasswordPolicy": Match.object_like({"MinimumLength": 12})},
                "MfaConfiguration": "OPTIONAL",
                "EnabledMfas": ["SOFTWARE_TOKEN_MFA"],
            },
        )

    def test_public_client_uses_srp_rotation_and_short_access_tokens(self) -> None:
        self.template.has_resource_properties(
            "AWS::Cognito::UserPoolClient",
            {
                "GenerateSecret": False,
                "ExplicitAuthFlows": ["ALLOW_USER_SRP_AUTH"],
                "EnableTokenRevocation": True,
                "AccessTokenValidity": 5,
                "TokenValidityUnits": Match.object_like({"AccessToken": "minutes"}),
                "RefreshTokenRotation": {"Feature": "ENABLED", "RetryGracePeriodSeconds": 10},
            },
        )

    def test_deployed_api_cannot_default_to_anonymous_access(self) -> None:
        self.template.has_resource_properties(
            "AWS::ECS::ExpressGatewayService",
            {
                "HealthCheckPath": "/health",
                "PrimaryContainer": Match.object_like(
                    {
                        "Environment": Match.array_with(
                            [
                                {"Name": "ENGRAM_AUTH_MODE", "Value": "cognito"},
                                {"Name": "ENGRAM_COGNITO_USER_POOL_ID", "Value": Match.any_value()},
                                {"Name": "ENGRAM_COGNITO_CLIENT_ID", "Value": Match.any_value()},
                            ]
                        )
                    }
                ),
            },
        )
        self.template.has_output("UserPoolId", {"Value": Match.any_value()})
        self.template.has_output("UserPoolClientId", {"Value": Match.any_value()})


if __name__ == "__main__":
    unittest.main()
