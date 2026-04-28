import requests

def send_to_service(prompt: str):
    response = requests.post(
        "http://localhost:8000/generate",
        json={"prompt": prompt}
    )
    return response.json()

if __name__ == "__main__":
    test = "WINNER! You have won a free iPhone! Click here to claim."
    result = send_to_service(test)
    print(result)