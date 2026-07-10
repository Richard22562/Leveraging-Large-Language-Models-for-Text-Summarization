import requests
from colorama import Fore, Style, init
from config import HF_API_KEY
init(autoreset=True)
default_model="google/pegasus-xsum"
def query(payload,model_name=default_model):
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    header=("Authorization",f"Bearer {HF_API_KEY}")
    return requests.post(url,headers=header,json=payload).json()
def summarize_text(text,min_length,max_length,model_name=default_model):
    payload={
        "inputs":text,
        "parameters":{"min_length":max_length,"max_length":max_length},
    }
    print(Fore.BLUE + Style.BRIGHT + f"\n Summarizing with midel: {model_name}")
    results=query(payload,model_name)
    return results[0].get("summary_text") if isinstance(results,list) and results else None
if __name__=="__main__":
    user_name=input(Fore.YELLOW+Style.BRIGHT+"Hello, what is your name?: ").strip()
    print(Fore.GREEN + f"welcome {user_name}")
    user_text=input(Fore.YELLOW+Style.BRIGHT+"Enter text to summarize: ").strip()
    if not user_text:
        print(Fore.RED + "no text provided, Exiting....")
        exit()
    style_choice=input(Fore.YELLOW+Style.BRIGHT+"Choose Style,:\n1.Standard\n2.Enhanced: ").strip()
    min_length,max_length=(80, 200) if style_choice =="2" else (50,150)
    summary=summarize_text(user_text,min_length,max_length,model_name=default_model)
    print(Fore.GREEN + f"Summary for {user_name}:\n============================\n")




