import requests
from colorama import Fore, Style, init
from config import HF_API_KEY

init(autoreset=True)

DEFAULT_MODEL = "google/pegasus-xsum"


def query(payload, model_name=DEFAULT_MODEL):
    url = f"https://router.huggingface.co/hf-inference/models/{model_name}"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Status {response.status_code}: {response.text}")

    if not response.content:
        raise Exception("Empty response from Hugging Face API")

    try:
        results = response.json()
    except Exception:
        raise Exception(f"Invalid response:\n{response.text}")

    if isinstance(results, dict) and "error" in results:
        raise Exception(results["error"])

    return results


def summarize_text(text, min_length, max_length, model_name=DEFAULT_MODEL):
    payload = {
        "inputs": text,
        "parameters": {
            "min_length": min_length,
            "max_length": max_length
        }
    }

    print(Fore.BLUE + Style.BRIGHT + f"\nSummarizing with model: {model_name}")

    results = query(payload, model_name)

    if isinstance(results, list) and len(results) > 0:
        return results[0].get("summary_text", "No summary generated.")

    return "No summary generated."


def main():
    user_name = input(
        Fore.YELLOW + Style.BRIGHT + "Hello, what is your name?: "
    ).strip()

    print(Fore.GREEN + f"Welcome {user_name}")

    user_text = input(
        Fore.YELLOW + Style.BRIGHT + "Enter text to summarize: "
    ).strip()

    if not user_text:
        print(Fore.RED + "No text provided. Exiting...")
        return

    style_choice = input(
        Fore.YELLOW
        + Style.BRIGHT
        + "Choose style:\n1. Standard\n2. Enhanced\nChoice: "
    ).strip()

    if style_choice == "2":
        min_length, max_length = 80, 200
    else:
        min_length, max_length = 50, 150

    try:
        summary = summarize_text(
            user_text,
            min_length,
            max_length,
            model_name=DEFAULT_MODEL
        )

        print(Fore.GREEN)
        print(f"Summary for {user_name}")
        print("=" * 40)
        print(summary)

    except Exception as e:
        print(Fore.RED + f"Error: {e}")


if __name__ == "__main__":
    main()