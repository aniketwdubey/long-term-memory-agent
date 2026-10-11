"""Try Cognito sign-in, inspect memories, and optionally chat with Engram.

This is a local learning tool. Passwords and tokens stay in this process and are
not written to disk or printed. It does not need AWS IAM credentials.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import boto3
from botocore.exceptions import ClientError

DEFAULT_POOL_ID = "us-east-1_5CSZjfMZm"
DEFAULT_CLIENT_ID = "5hqph58gj3gfntfbhug23svpak"
DEFAULT_API_URL = "https://en-bcd35655a2874eefa88fcc6f50da39e3.ecs.us-east-1.on.aws"


def list_memories(api_url: str, access_token: str, user_id: str, *, details: bool = False) -> None:
    request = Request(
        f"{api_url.rstrip('/')}/v1/memories/{quote(user_id, safe='')}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
        print(f"API status: {response.status}")
        print(f"Your user ID: {user_id}")
        print(f"Your memories: {result['active']} active, {result['total']} total")
        if details:
            for memory in result["memories"]:
                status = "active" if memory["active"] else "retired"
                print(f"  - [{status}] {memory['text']}")


def send_chat(
    api_url: str, access_token: str, user_id: str, thread_id: str, message: str
) -> dict[str, Any]:
    request = Request(
        f"{api_url.rstrip('/')}/v1/chat",
        data=json.dumps({"user_id": user_id, "thread_id": thread_id, "message": message}).encode(),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        result: dict[str, Any] = json.load(response)
        return result


def chat_loop(api_url: str, access_token: str, user_id: str, thread_id: str) -> None:
    print("Chat ready. Type a message, /memories, /thread NAME, or /quit.")
    print(f"Conversation: {thread_id}")
    while True:
        try:
            message = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            continue
        if message == "/quit":
            return
        if message == "/memories":
            list_memories(api_url, access_token, user_id, details=True)
            continue
        if message.startswith("/thread "):
            thread_id = message.removeprefix("/thread ").strip()
            if not thread_id:
                print("Use /thread NAME, for example /thread second-session.")
                continue
            print(f"Conversation: {thread_id}")
            continue
        if message.startswith("/"):
            print("Commands: /memories, /thread NAME, /quit")
            continue
        try:
            result = send_chat(api_url, access_token, user_id, thread_id, message)
        except HTTPError as exc:
            if exc.code == 401:
                print("Access token expired. Run the command again to sign in.")
                return
            print(f"Engram API returned HTTP {exc.code}.")
            continue
        print(f"Engram> {result['reply']}")
        for label, key in (("Recalled", "recalled"), ("Saved", "written")):
            memories = result[key]
            print(f"{label} ({len(memories)}):")
            for memory in memories:
                print(f"  - {memory['text']}")


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
    parser.add_argument("--chat", action="store_true", help="Chat after signing in")
    parser.add_argument("--thread", default="first-session", help="Conversation name for --chat")
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
        if args.chat:
            chat_loop(args.api_url, access_token, user_id, args.thread)
    except HTTPError as exc:
        print(f"Engram API returned HTTP {exc.code}.")
        return 1
    except ClientError as exc:
        print(f"Cognito sign-in error: {exc.response['Error']['Code']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
