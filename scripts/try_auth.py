"""Try a real Cognito sign-up/sign-in and one authenticated API request.

This is a local learning tool. Passwords and tokens stay in this process and are
not written to disk or printed. It does not need AWS IAM credentials.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import ClientError

DEFAULT_POOL_ID = "us-east-1_5CSZjfMZm"
DEFAULT_CLIENT_ID = "5hqph58gj3gfntfbhug23svpak"
DEFAULT_API_URL = "https://en-bcd35655a2874eefa88fcc6f50da39e3.ecs.us-east-1.on.aws"


def list_memories(api_url: str, access_token: str, user_id: str) -> None:
    request = Request(
        f"{api_url.rstrip('/')}/v1/memories/{quote(user_id, safe='')}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
        print(f"API status: {response.status}")
        print(f"Your user ID: {user_id}")
        print(f"Your memories: {result['total']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or use a Cognito account and call the Engram API."
    )
    parser.add_argument(
        "--pool-id", default=os.getenv("ENGRAM_COGNITO_USER_POOL_ID", DEFAULT_POOL_ID)
    )
    parser.add_argument(
        "--client-id", default=os.getenv("ENGRAM_COGNITO_CLIENT_ID", DEFAULT_CLIENT_ID)
    )
    parser.add_argument("--api-url", default=os.getenv("ENGRAM_API_URL", DEFAULT_API_URL))
    args = parser.parse_args()

    try:
        from pycognito.aws_srp import AWSSRP
    except ImportError:
        print('Install the optional sign-in helper: .venv/bin/pip install -e ".[auth-demo]"')
        return 1

    email = input("Your email: ").strip()
    if not email or "@" not in email:
        print("Enter a valid email address.")
        return 1
    password = getpass.getpass("Password (12+ chars, upper/lower, number, symbol; hidden): ")
    client = boto3.client("cognito-idp", region_name=args.pool_id.split("_", 1)[0])

    try:
        client.sign_up(
            ClientId=args.client_id,
            Username=email,
            Password=password,
            UserAttributes=[{"Name": "email", "Value": email}],
        )
        print("Cognito sent a confirmation code to your email.")
        code = input("Confirmation code from the email: ").strip()
        client.confirm_sign_up(ClientId=args.client_id, Username=email, ConfirmationCode=code)
        print("Email confirmed.")
    except client.exceptions.UsernameExistsException:
        print("Account already exists; signing in with your password.")
    except client.exceptions.InvalidPasswordException:
        print("Use at least 12 characters with uppercase, lowercase, a number, and a symbol.")
        return 1
    except ClientError as exc:
        print(f"Cognito error: {exc.response['Error']['Code']}")
        return 1

    try:
        response = AWSSRP(
            username=email,
            password=password,
            pool_id=args.pool_id,
            client_id=args.client_id,
            client=client,
        ).authenticate_user()
        access_token = response["AuthenticationResult"]["AccessToken"]
        user = client.get_user(AccessToken=access_token)
        user_id = next(
            attribute["Value"] for attribute in user["UserAttributes"] if attribute["Name"] == "sub"
        )
        print("Cognito sign-in succeeded. Reading your memories with the access token...")
        list_memories(args.api_url, access_token, user_id)
    except HTTPError as exc:
        print(f"Engram API returned HTTP {exc.code}.")
        return 1
    except ClientError as exc:
        print(f"Cognito sign-in error: {exc.response['Error']['Code']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
