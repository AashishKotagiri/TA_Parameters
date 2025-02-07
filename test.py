import requests

API_URL = "http://127.0.0.1:8000/generate"

def ask_sqlbot(question):
    """Send a request to the FastAPI server and print the response."""
    response = requests.post(API_URL, json={"user_input": question})
    if response.status_code == 200:
        print(f" AI: {response.json()['response']}\n")
    else:
        print(f" Error: {response.text}")

# Test Chat Session
if __name__ == "__main__":
    print("\n=== SQLBot Chat Test ===\n")

    queries = [
        "Generate an SQL query to fetch all employees earning above $70,000.",
        "Modify it to filter only those from the Sales department.",
        "Now order results by salary in descending order."
    ]

    for query in queries:
        print(f" User: {query}")
        ask_sqlbot(query)
