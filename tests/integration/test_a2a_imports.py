import asyncio
import httpx
import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory for common_utils access
parent_dir = Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Correct imports based on SDK structure
from a2a.client.client import A2AClient
from a2a.types import SendMessageRequest, Message, TextPart, MessageSendParams

async def test_imports_and_objects():
    """
    A minimal test to confirm correct import and object creation for the A2A SDK.
    """
    print("--- Testing A2A SDK Imports and Object Creation ---")
    try:
        # 1. Create a simple text part
        text_part = TextPart(text="Test message")
        print("✅ Successfully created TextPart")

        # 2. Create a Message
        message = Message(
            messageId=str(uuid4()),
            role="user",
            parts=[text_part],
        )
        print("✅ Successfully created Message")

        # 3. Create MessageSendParams
        params = MessageSendParams(message=message)
        print("✅ Successfully created MessageSendParams")

        # 4. Create a SendMessageRequest
        request = SendMessageRequest(
            id=str(uuid4()),
            params=params
        )
        print("✅ Successfully created SendMessageRequest")

        print("\n--- All objects created successfully ---")
        return True

    except ImportError as e:
        print(f"❌ ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_imports_and_objects()) 